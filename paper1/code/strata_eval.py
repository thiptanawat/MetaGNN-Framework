#!/usr/bin/env python3
"""Where does the structural signal live? Held-out AUROC by reaction class and arm.

If a structural model is partly re-deriving the labeling rule, its advantage should be
concentrated where that rule is simplest: reactions with no gene rule, transport and
exchange reactions, and the network's hubs. Scoring the saved predictions of every arm
inside those strata says where structure wins, where expression wins, and where the
rewired network loses, without retraining anything.

Strata are defined on the held-out reactions of each cell only, so nothing here touches
a reaction the model trained on.
"""
import os, re, glob, json, statistics as st
import numpy as np, pandas as pd, torch, h5py
from sklearn.metrics import roc_auc_score

D = os.path.expanduser("~/metagnn/work/crc_624")
OUTD = os.path.expanduser("~/metagnn/out")
OUT = os.path.join(OUTD, "strata.json")
N = 10600

y = np.asarray(torch.load(D + "/activity_pseudolabels.pt", map_location="cpu", weights_only=False)).astype(int)
AL = json.load(open(os.path.expanduser("~/metagnn/out/recon3d_aligned.json")))["reactions"]; assert len(AL) == N
gpr = np.array([bool(r["gene_sets"]) for r in AL])          # the project's own gene table, canonical order
sub = np.array([(r.get("subsystem") or "") for r in AL])
transport = np.array([s.startswith("Transport") or "xchange" in s for s in sub]) | np.array([r["is_exchange"] for r in AL])
t = torch.load(f"{D}/edge_indices/shared_metabolite.pt", map_location="cpu", weights_only=False).numpy()
deg = np.zeros(N); np.add.at(deg, t[0][t[0] < N], 1)
# expression-bearing mask, as the fold stratification computes it
meta = pd.read_csv(D + "/clinical_metadata.tsv", sep="\t")
from sklearn.model_selection import StratifiedKFold, train_test_split
ids = meta["tcga_barcode"].tolist(); proj = meta["project"].fillna("U").tolist()
tr, te = next(iter(StratifiedKFold(5, shuffle=True, random_state=2024).split(ids, proj)))
trn, _ = train_test_split([ids[i] for i in tr], test_size=0.15, random_state=2024)
has = np.stack([h5py.File(f"{D}/reaction_features/{p}.h5")["X_R"][:, 0] for p in trn[:20]]).max(0) > 0

q = np.quantile(deg, [1/3, 2/3])
STRATA = {
    "gpr_rule": gpr, "no_gpr_rule": ~gpr,
    "expression_bearing": has, "no_expression": ~has,
    "transport_exchange": transport, "metabolic": ~transport,
    "degree_low": deg <= q[0], "degree_mid": (deg > q[0]) & (deg <= q[1]), "degree_high": deg > q[1],
}
PAT = re.compile(r"^gnn_B_mb0\.0_p(?P<pf>\d)_r(?P<rf>\d)(?:_(?P<mode>mean|zero|indicator|rewire\d+))?_preds\.npz$")

R = {"strata_sizes": {k: int(v.sum()) for k, v in STRATA.items()},
     "strata_prevalence": {k: float(y[v].mean()) for k, v in STRATA.items()},
     "arms": {}}
for d in (OUTD + "/dh", OUTD + "/ctrl", OUTD + "/rewire"):
    for f in sorted(glob.glob(os.path.join(d, "*_preds.npz"))):
        m = PAT.match(os.path.basename(f))
        if not m:
            continue
        mode = m.group("mode") or "real"; cell = f"p{m.group('pf')}r{m.group('rf')}"
        z = np.load(f, allow_pickle=True); P = z["scores"]; mk = z["heldout_mask"]
        rec = {}
        for name, s in STRATA.items():
            mm = mk & s
            if mm.sum() < 30 or len(set(y[mm])) < 2:
                continue
            rec[name] = float(np.mean([roc_auc_score(y[mm], P[i, mm]) for i in range(len(P))]))
        R["arms"].setdefault(mode, {})[cell] = rec

# roll up: mean over cells per arm per stratum, with the cell count. The three substitution arms are
# rolled up on their common basis, the cells where all three have run, so that the columns of the
# table are paired; the rewired arm and the presence-only (indicator) arm are rolled up on every cell
# they have, and their paired differences below are taken on their own cells.
BASIS = sorted(set(R["arms"].get("real", {})) & set(R["arms"].get("mean", {})) & set(R["arms"].get("zero", {})))
R["basis"] = BASIS
roll = {}
for mode, cells in R["arms"].items():
    roll[mode] = {}
    use = {c: v for c, v in cells.items() if (mode.startswith("rewire") or mode == "indicator" or c in BASIS)}
    for name in STRATA:
        v = [c[name] for c in use.values() if name in c]
        if v:
            roll[mode][name] = dict(mean=st.mean(v), sd=(st.stdev(v) if len(v) > 1 else None), n=len(v))
R["rollup"] = roll
# paired differences between arms inside each stratum, on the basis cells
R["paired"] = {}
for a, b, lab in (("mean", "zero", "aggregate_expression"), ("real", "mean", "patient_identity"), ("real", "zero", "expression_total")):
    if a in R["arms"] and b in R["arms"]:
        R["paired"][lab] = {}
        for name in STRATA:
            d = [R["arms"][a][c][name] - R["arms"][b][c][name] for c in BASIS
                 if name in R["arms"][a].get(c, {}) and name in R["arms"][b].get(c, {})]
            if d:
                R["paired"][lab][name] = dict(mean=st.mean(d), n=len(d), n_positive=sum(1 for x in d if x > 0))
# the presence-only arm against the zeroed and cohort-mean arms, paired on the indicator arm's own cells
if "indicator" in R["arms"]:
    IB = sorted(set(R["arms"]["indicator"]) & set(R["arms"].get("mean", {})) & set(R["arms"].get("zero", {})))
    R["indicator_basis"] = IB
    for a, b, lab in (("indicator", "zero", "presence_over_zero"), ("mean", "indicator", "magnitude_over_presence"),
                      ("real", "indicator", "own_over_presence")):
        if a in R["arms"] and b in R["arms"]:
            R["paired"][lab] = {}
            for name in STRATA:
                d = [R["arms"][a][c][name] - R["arms"][b][c][name] for c in IB
                     if name in R["arms"][a].get(c, {}) and name in R["arms"][b].get(c, {})]
                if d:
                    R["paired"][lab][name] = dict(mean=st.mean(d), n=len(d), n_positive=sum(1 for x in d if x > 0))
# held-out reactions per stratum, averaged over cells, so the table can say what each AUROC rests on
held = {name: [] for name in STRATA}
for d in (OUTD + "/dh",):
    for f in sorted(glob.glob(os.path.join(d, "gnn_B_mb0.0_p*_r*_preds.npz"))):
        m = PAT.match(os.path.basename(f))
        if not m or m.group("mode"):
            continue
        mk = np.load(f, allow_pickle=True)["heldout_mask"]
        for name, s in STRATA.items():
            held[name].append(int((mk & s).sum()))
R["strata_heldout_mean"] = {name: (float(np.mean(v)) if v else None) for name, v in held.items()}

print("%-20s %6s %6s | %-8s %-8s %-8s %-8s %-8s" % ("stratum", "n_rxn", "prev", "real", "mean", "zero", "indic.", "rewire7"))
print("-" * 86)
for name in STRATA:
    row = [roll.get(m, {}).get(name, {}).get("mean") for m in ("real", "mean", "zero", "indicator", "rewire7")]
    print("%-20s %6d %6.3f | %s" % (name, R["strata_sizes"][name], R["strata_prevalence"][name],
          " ".join(("%.4f  " % x) if x is not None else "   n/a   " for x in row)))
json.dump(R, open(OUT, "w"), indent=1)
print("\nwrote", OUT)

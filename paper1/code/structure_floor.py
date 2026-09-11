#!/usr/bin/env python3
"""A structure-only floor for the double hold-out: the first, exploratory version.

Before crediting a graph model's accuracy to transcriptomics, ask how much of it is available from
the network alone. The activity labels are membership in a union of tINIT-style reconstructions, and
those algorithms decide membership largely from connectivity, so topology may predict the label
directly. This fits logistic regression on purely structural per-reaction features, no patient data
of any kind, on the identical reaction folds as the graph model, and reports held-out AUROC.

This script is kept for provenance and is NOT the source of any number in the manuscript. The floor
the manuscript reports is computed by naive_baselines_allfolds.py, which runs the same idea on every
patient fold, tunes the regularization inside each training fold, adds the annotation and topology
splits, and writes results/naive_allfolds.json; that is what reproduce.sh runs. Use this one only to
see the original one-patient-fold calculation.

It runs on the GPU host's data tree, like the other collection-side scripts: set P1_DATA, P1_RECON
and P1_OUT, or accept the ~/metagnn defaults. The reference network is read in its own file order and
checked against the aligned record's order before use, so a positional index cannot silently drift.
"""
import json, glob, numpy as np, pandas as pd, torch, h5py, os
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

D = os.environ.get("P1_DATA", os.path.expanduser("~/metagnn/work/crc_624"))
N = 10600
y = np.asarray(torch.load(D + "/activity_pseudolabels.pt", map_location="cpu",
                          weights_only=False)).astype(int)

# reproduce the pipeline's patient folds and its `has` mask exactly
meta = pd.read_csv(D + "/clinical_metadata.tsv", sep="\t")
ids = meta["tcga_barcode"].tolist(); proj = meta["project"].fillna("U").tolist()
pk = StratifiedKFold(5, shuffle=True, random_state=2024)
tr, te = next(iter(pk.split(ids, proj)))
tr_ids = [ids[i] for i in tr]
trn, _ = train_test_split(tr_ids, test_size=0.15, random_state=2024)
probe = trn[:20]
Xp = np.stack([h5py.File(f"{D}/reaction_features/{p}.h5")["X_R"][:, 0] for p in probe])
has = Xp.max(0) > 0
rk = StratifiedKFold(3, shuffle=True, random_state=2024)
rfolds = list(rk.split(np.zeros(N), 2 * y + has.astype(int)))

# ---- purely structural features, no patient involved ----
RECON = os.environ.get("P1_RECON", os.path.expanduser("~/metagnn/metagnn-data-v2/shared/Recon3D.json"))
RX = json.load(open(RECON))["reactions"]
assert len(RX) == N, f"{RECON} holds {len(RX)} reactions, not {N}"
# the features below index RX positionally against the training arrays, so check that the order is
# the canonical one rather than assuming it: the aligned record carries the same ids in that order
_ALIGNED = os.environ.get("P1_ALIGNED", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                     "data", "recon3d_aligned.json.gz"))
if os.path.exists(_ALIGNED):
    import gzip as _gzip
    _al = json.load(_gzip.open(_ALIGNED))["reactions"]
    assert [r.get("id") for r in RX] == [r.get("id") for r in _al], (
        "the reference network is not in the canonical order of the training arrays; "
        "positional indexing here would be wrong")
deg = {}
for e in ["substrate_of", "produces", "shared_metabolite"]:
    t = torch.load(f"{D}/edge_indices/{e}.pt", map_location="cpu", weights_only=False).numpy()
    d = np.zeros(N)
    src = t[0] if e != "produces" else t[1]
    np.add.at(d, src[src < N], 1)
    deg[e] = d
sub = pd.Series([(r.get("subsystem") or "(none)") for r in RX])
top = sub.value_counts().index[:60]
F = [deg["substrate_of"], deg["produces"], deg["shared_metabolite"],
     np.log1p(deg["shared_metabolite"]),
     np.array([len(r["metabolites"]) for r in RX], float),
     np.array([1.0 if r.get("lower_bound", 0) < 0 else 0.0 for r in RX]),
     np.array([1.0 if (r.get("gene_reaction_rule") or "").strip() else 0.0 for r in RX]),
     np.array([len(set((r.get("gene_reaction_rule") or "").replace("(", " ")
               .replace(")", " ").split()) - {"and", "or"}) for r in RX], float),
     np.array([1.0 if str(r["id"]).startswith("EX_") else 0.0 for r in RX])]
F += [(sub == s).values.astype(float) for s in top]
X = np.column_stack(F)
names = ["deg_substrate", "deg_produces", "deg_shared", "log_deg_shared", "n_metabolites",
         "reversible", "has_gpr", "n_genes", "is_exchange"]
print(f"structural design matrix: {X.shape}  ({len(names)} numeric + {len(top)} subsystem indicators)")
print("no patient data is used anywhere in this script\n")

print("%-8s %10s %10s %10s %10s" % ("rfold", "n_held", "structure", "indicator", "GNN(same fold)"))
print("-" * 56)
gnn = {}
for f in glob.glob(os.path.join(os.environ.get("P1_OUT", os.path.expanduser("~/metagnn/out")),
                                "dh", "gnn_B_mb0.0_p0_r*.json")):
    r = json.load(open(f)); gnn[r["rfold"]] = r["heldout_AUROC_perpatient"]
aucs = []
for rf, (rtr, rte) in enumerate(rfolds):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
    m.fit(X[rtr], y[rtr])
    p = m.predict_proba(X[rte])[:, 1]
    a = roc_auc_score(y[rte], p); aucs.append(a)
    ind = roc_auc_score(y[rte], has[rte].astype(float))
    g = gnn.get(rf)
    print("%-8d %10d %10.4f %10.4f %10s" % (rf, len(rte), a, ind, f"{g:.4f}" if g else "pending"))
print("-" * 56)
print(f"structure-only mean over folds: {np.mean(aucs):.4f}")

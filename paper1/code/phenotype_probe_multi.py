#!/usr/bin/env python3
"""The phenotype probe on further patient attributes, so the conclusion is not MSI-specific.

Three attributes annotated in the cohort, none of which the model ever saw:
  cin_vs_gs   chromosomal instability against genomically stable, a DNA-defined molecular
              subtype from copy-number profiling
  sex         male against female
  site        colon against rectum, the anatomical site of the primary

For each, the unsubstituted arm's out-of-fold score vector, the model's input, and the
cohort-mean arm are classified exactly as for microsatellite instability. If the output
tracks the input across attributes with very different transcriptional legibility, the
output is a faithful carrier of its input's patient signal and the MSI result is not a
special case.

Usage: python3 phenotype_probe_multi.py <attribute>   (one process per attribute)
"""
import os, re, sys, glob, json
import numpy as np, pandas as pd, h5py
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score

D = os.path.expanduser("~/metagnn/work/crc_624")
OUTD = os.path.expanduser("~/metagnn/out")
SEED = 2024
which = sys.argv[1]
OUT = os.path.join(OUTD, f"phenotype_{which}.json")

meta = pd.read_csv(os.path.join(D, "clinical_metadata_msi.tsv"), sep="\t").set_index("tcga_barcode")
if which == "cin_vs_gs":
    st = meta["SUBTYPE"].fillna("NA")
    y_all = st.map({"COAD_CIN": 1, "READ_CIN": 1, "COAD_GS": 0, "READ_GS": 0}).dropna().astype(int)
    label = "chromosomal instability vs genomically stable"
elif which == "sex":
    y_all = meta["gender"].map({"male": 1, "female": 0}).dropna().astype(int)
    label = "male vs female"
elif which == "site":
    y_all = meta["project"].map({"TCGA-COAD": 1, "TCGA-READ": 0}).dropna().astype(int)
    label = "colon vs rectum"
else:
    raise SystemExit("unknown attribute")
print("%s: pos=%d neg=%d" % (label, (y_all == 1).sum(), (y_all == 0).sum()), flush=True)

PAT = re.compile(r"^gnn_B_mb0\.0_p(?P<pf>\d)_r(?P<rf>\d)(?:_(?P<mode>mean|zero|rewire\d+))?_preds\.npz$")

def assemble(mode, rf):
    rows, ids = [], []
    for d in (OUTD + "/dh", OUTD + "/ctrl"):
        for f in sorted(glob.glob(os.path.join(d, "*_preds.npz"))):
            m = PAT.match(os.path.basename(f))
            if not m or int(m.group("rf")) != rf or (m.group("mode") or "real") != mode:
                continue
            z = np.load(f, allow_pickle=True)
            rows.append(z["scores"].astype(np.float32)); ids += [str(x) for x in z["patient_ids"]]
    return (np.vstack(rows), ids) if rows else (None, None)

def expression_input(ids):
    return np.stack([h5py.File(f"{D}/reaction_features/{p}.h5")["X_R"][:, 0] for p in ids]).astype(np.float32)

def nested_auc(X, y, seed=SEED, ids=None):
    """Identical to phenotype_probe.py; with ids given, the out-of-fold predictions are stored
    beside the summary so that build_results_p1.py can form an interval for the mean over
    reaction folds by resampling patients across folds."""
    outer = StratifiedKFold(5, shuffle=True, random_state=seed); oof = np.zeros(len(y))
    for tr, te in outer.split(X, y):
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced"))
        gs = GridSearchCV(pipe, {"logisticregression__C": [1e-3, 1e-2, 1e-1, 1.0]},
                          cv=StratifiedKFold(3, shuffle=True, random_state=seed), scoring="roc_auc", n_jobs=4)
        gs.fit(X[tr], y[tr]); oof[te] = gs.predict_proba(X[te])[:, 1]
    auc = float(roc_auc_score(y, oof))
    rng = np.random.default_rng(seed); boots = []
    for _ in range(1000):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) > 1: boots.append(roc_auc_score(y[i], oof[i]))
    res = dict(auroc=auc, ci_lo=float(np.percentile(boots, 2.5)), ci_hi=float(np.percentile(boots, 97.5)),
               n=int(len(y)), n_pos=int(y.sum()))
    if ids is not None:
        res["oof"] = dict(patient_ids=[str(p) for p in ids], y=[int(v) for v in y],
                          score=[round(float(v), 6) for v in oof])
    return res

R = {"attribute": which, "label": label, "arms": {}}
for mode in ("real", "mean"):
    for rf in (0, 1, 2) if mode == "real" else (0,):
        S, ids = assemble(mode, rf)
        if S is None: continue
        keep = [i for i, p in enumerate(ids) if p in y_all.index]
        y = y_all.loc[[ids[i] for i in keep]].values; X = S[keep]
        if len(set(y)) < 2 or y.sum() < 20 or (len(y) - y.sum()) < 20: continue
        r = nested_auc(X, y, ids=[ids[i] for i in keep]); R["arms"].setdefault(mode, {})[f"r{rf}"] = r
        print("%-6s r%d n=%3d pos=%3d  AUROC %.3f [%.3f, %.3f]" % (mode, rf, r["n"], r["n_pos"], r["auroc"], r["ci_lo"], r["ci_hi"]), flush=True)
        json.dump(R, open(OUT, "w"), indent=1)
S, ids = assemble("real", 0); keep = [i for i, p in enumerate(ids) if p in y_all.index]
sel = [ids[i] for i in keep]; y = y_all.loc[sel].values
r = nested_auc(expression_input(sel), y, ids=sel); R["input_reaction_expression"] = r
print("%-6s    n=%3d pos=%3d  AUROC %.3f [%.3f, %.3f]   <- input" % ("input", r["n"], r["n_pos"], r["auroc"], r["ci_lo"], r["ci_hi"]), flush=True)
json.dump(R, open(OUT, "w"), indent=1); print("wrote", OUT)

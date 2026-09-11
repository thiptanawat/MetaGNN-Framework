#!/usr/bin/env python3
"""Permutation nulls for the phenotype probe, through the identical nested pipeline.

The first pass at a null (phenotype_probe.py) refit the classifier at a fixed
regularization strength, which is not the procedure that produced the observed value,
and it covered microsatellite instability only. This script permutes the phenotype
labels and reruns the exact nested procedure (an outer 5-fold split, the regularization
strength chosen by an inner 3-fold grid inside every outer fold, balanced class weights),
so the null calibrates the estimator that was actually used, including its model
selection. It does so for every attribute in the paper, on the unsubstituted arm at the
first reaction fold, and for the model's input on the same patients.

Usage: python3 phenotype_null.py <attribute> [n_perm]     attribute in msi cin_vs_gs sex site
Writes out/phenotype_null_<attribute>.json.
"""
import os, re, sys, glob, json, time
import numpy as np, pandas as pd, h5py
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score

D = os.path.expanduser("~/metagnn/work/crc_624")
OUTD = os.path.expanduser("~/metagnn/out")
SEED = 2024
which = sys.argv[1]; NPERM = int(sys.argv[2]) if len(sys.argv) > 2 else 200
NJOBS = int(os.environ.get("NJOBS", "16"))
OUT = os.path.join(OUTD, f"phenotype_null_{which}.json")

meta = pd.read_csv(os.path.join(D, "clinical_metadata_msi.tsv"), sep="\t").set_index("tcga_barcode")
if which == "msi":
    y_all = meta["msi_status"].fillna("NA").map({"MSI-H": 1, "MSS": 0}).dropna().astype(int)
    label = "MSI-H vs MSS"
elif which == "cin_vs_gs":
    y_all = meta["SUBTYPE"].fillna("NA").map({"COAD_CIN": 1, "READ_CIN": 1, "COAD_GS": 0, "READ_GS": 0}).dropna().astype(int)
    label = "chromosomal instability vs genomically stable"
elif which == "sex":
    y_all = meta["gender"].map({"male": 1, "female": 0}).dropna().astype(int)
    label = "male vs female"
elif which == "site":
    y_all = meta["project"].map({"TCGA-COAD": 1, "TCGA-READ": 0}).dropna().astype(int)
    label = "colon vs rectum"
else:
    raise SystemExit("unknown attribute")

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

def nested_oof(X, y, seed):
    """The identical procedure to phenotype_probe.py and phenotype_probe_multi.py."""
    outer = StratifiedKFold(5, shuffle=True, random_state=seed); oof = np.zeros(len(y))
    for tr, te in outer.split(X, y):
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced"))
        gs = GridSearchCV(pipe, {"logisticregression__C": [1e-3, 1e-2, 1e-1, 1.0]},
                          cv=StratifiedKFold(3, shuffle=True, random_state=seed), scoring="roc_auc", n_jobs=1)
        gs.fit(X[tr], y[tr]); oof[te] = gs.predict_proba(X[te])[:, 1]
    return oof

def one_perm(X, y, k):
    rng = np.random.default_rng(SEED + 1000 + k)
    yp = rng.permutation(y)
    return float(roc_auc_score(yp, nested_oof(X, yp, SEED + k)))

def run(X, y, name):
    t0 = time.time()
    obs = float(roc_auc_score(y, nested_oof(X, y, SEED)))
    null = Parallel(n_jobs=NJOBS)(delayed(one_perm)(X, y, k) for k in range(NPERM))
    null = np.asarray(null)
    res = dict(auroc=obs, n=int(len(y)), n_pos=int(y.sum()), n_perm=int(NPERM),
               perm_null_mean=float(null.mean()), perm_null_sd=float(null.std()),
               perm_null_max=float(null.max()),
               perm_p=float((np.sum(null >= obs) + 1) / (NPERM + 1)),
               perm_p_floor=float(1.0 / (NPERM + 1)),
               null_draws=[round(float(v), 4) for v in null])
    print("%-24s %-6s n=%3d pos=%3d  observed %.3f  null %.3f (sd %.3f, max %.3f)  p = %.4f  [%.0f s]"
          % (which, name, res["n"], res["n_pos"], obs, res["perm_null_mean"], res["perm_null_sd"],
             res["perm_null_max"], res["perm_p"], time.time() - t0), flush=True)
    return res

# the unsubstituted arm at every reaction fold, so the null sits beside the three-fold mean the table
# reports; the model's input once, on the patients of the first fold's assembly
R = dict(attribute=which, label=label, arm="real, every reaction fold",
         pipeline="nested, identical to the observed value", output_by_fold={})
for rf in (0, 1, 2):
    S, ids = assemble("real", rf)
    if S is None:
        continue
    keep = [i for i, p in enumerate(ids) if p in y_all.index]
    sel = [ids[i] for i in keep]; y = y_all.loc[sel].values
    R["output_by_fold"][f"r{rf}"] = run(S[keep], y, f"output r{rf}")
    json.dump(R, open(OUT, "w"), indent=1)
folds = R["output_by_fold"]
if folds:
    R["output"] = dict(auroc=float(np.mean([v["auroc"] for v in folds.values()])),
                       n=folds["r0"]["n"], n_pos=folds["r0"]["n_pos"], n_perm=NPERM, n_rfolds=len(folds),
                       perm_null_mean=float(np.mean([v["perm_null_mean"] for v in folds.values()])),
                       perm_null_sd=float(np.mean([v["perm_null_sd"] for v in folds.values()])),
                       perm_null_max=float(max(v["perm_null_max"] for v in folds.values())),
                       perm_p=float(max(v["perm_p"] for v in folds.values())),        # the least favorable fold
                       perm_p_by_fold={k: v["perm_p"] for k, v in folds.items()},
                       perm_p_floor=float(1.0 / (NPERM + 1)))
    json.dump(R, open(OUT, "w"), indent=1)
S, ids = assemble("real", 0)
keep = [i for i, p in enumerate(ids) if p in y_all.index]
sel = [ids[i] for i in keep]; y = y_all.loc[sel].values
R["input"] = run(expression_input(sel), y, "input")
json.dump(R, open(OUT, "w"), indent=1)
print("wrote", OUT)

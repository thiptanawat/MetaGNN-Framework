#!/usr/bin/env python3
"""Does the model's patient-specific output carry real patient signal?

Every claim in the paper so far is made against a label vector shared by all patients,
which by construction cannot reward patient specificity. This asks the complementary
question against a phenotype that does vary between patients: microsatellite
instability, annotated for nearly the whole cohort and carrying a strong transcriptional
signature.

For each feature-mode arm, the out-of-fold score matrix over patients is assembled from
the five patient-fold cells at one reaction fold, so every patient is scored by a model
that never trained on them. A logistic regression is then trained, with nested
cross-validation over patients, to predict MSI-H against MSS from that 10,600-dimensional
score vector. The same classifier on the model's own input, reaction-level expression,
says how much patient signal the model was given; the cohort-mean and zeroed arms, where
every patient receives an identical input, are negative controls that should sit at
chance; a gene-level classifier is an upper reference for how much MSI signal the
transcriptome carries before the GPR mapping.

Three outcomes, all informative. Scores match the input: the model preserves patient
signal but adds none. Scores fall below the input: the model discards patient signal,
which is what collapsing toward a shared ranking predicts. Scores exceed the input: the
model extracts patient-specific structure the benchmark cannot see.
"""
import os, re, glob, json, argparse, statistics as st
import numpy as np, pandas as pd, h5py
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--dirs", nargs="+", default=["dh", "ctrl", "rewire"], help="result directories under ~/metagnn/out to scan")
ap.add_argument("--suffix", default="", help="tag suffix of the cells to probe, e.g. _ivr for the matched-rule reruns")
ap.add_argument("--modes", nargs="+", default=["real", "mean", "zero", "rewire7"])
ap.add_argument("--out", default="phenotype_probe.json")
A = ap.parse_args()

D = os.path.expanduser("~/metagnn/work/crc_624")
OUTD = os.path.expanduser("~/metagnn/out")
OUT = os.path.join(OUTD, A.out)
SEED = 2024

# ---- phenotype -------------------------------------------------------------
meta = pd.read_csv(os.path.join(D, "clinical_metadata_msi.tsv"), sep="\t")
lab = meta.set_index("tcga_barcode")["msi_status"].fillna("NA")
y_all = lab.map({"MSI-H": 1, "MSS": 0}).dropna().astype(int)
print("phenotype: MSI-H=%d MSS=%d (MSI-L and not-evaluable excluded)"
      % ((y_all == 1).sum(), (y_all == 0).sum()), flush=True)

# ---- score matrices, out of fold -------------------------------------------
PAT = re.compile(r"^gnn_B_mb0\.0_p(?P<pf>\d)_r(?P<rf>\d)(?:_(?P<mode>mean|zero|indicator|permute|rewire\d+))?" + re.escape(A.suffix) + r"_preds\.npz$")

def assemble(mode, rf):
    """Stack the test patients of every patient-fold cell at reaction fold rf."""
    rows, ids = [], []
    for d in (OUTD + "/" + x for x in A.dirs):
        for f in sorted(glob.glob(os.path.join(d, "*_preds.npz"))):
            m = PAT.match(os.path.basename(f))
            if not m or int(m.group("rf")) != rf or (m.group("mode") or "real") != mode:
                continue
            z = np.load(f, allow_pickle=True)
            rows.append(z["scores"].astype(np.float32)); ids += [str(x) for x in z["patient_ids"]]
    if not rows:
        return None, None
    return np.vstack(rows), ids

def expression_input(ids):
    return np.stack([h5py.File(f"{D}/reaction_features/{p}.h5")["X_R"][:, 0] for p in ids]).astype(np.float32)

def gene_level(ids):
    """The transcriptome before GPR mapping, as an upper reference; None if unreadable."""
    try:
        with h5py.File(os.path.join(D, "tcga_crc_rnaseq.h5")) as h:
            keys = list(h.keys())
            # find an expression matrix and a barcode index
            X = None; bc = None
            for k in keys:
                a = h[k]
                if isinstance(a, h5py.Dataset) and a.ndim == 2 and max(a.shape) > 5000:
                    X = a[:]
                if isinstance(a, h5py.Dataset) and a.dtype.kind in ("S", "O", "U") and a.ndim == 1:
                    vals = [v.decode() if isinstance(v, bytes) else str(v) for v in a[:]]
                    if any(v.startswith("TCGA-") for v in vals[:5]):
                        bc = vals
            if X is None or bc is None:
                return None
            if X.shape[0] != len(bc):
                X = X.T
            idx = {b: i for i, b in enumerate(bc)}
            sel = [idx[p] for p in ids if p in idx]
            if len(sel) != len(ids):
                return None
            G = X[sel].astype(np.float32)
            return np.log2(G + 1) if G.max() > 50 else G
    except Exception as e:
        print("gene-level reference unavailable:", type(e).__name__, str(e)[:80], flush=True)
        return None

# ---- classifier -------------------------------------------------------------
def nested_auc(X, y, seed=SEED, n_perm=0, ids=None):
    """Out-of-fold AUROC with the regularization strength chosen inside each fold.

    When the patient barcodes are passed as ids, the out-of-fold predictions are stored
    beside the summary ("oof": barcodes, labels and scores), so that an interval for the
    mean over reaction folds can later be formed by resampling patients across folds
    (build_results_p1.py) instead of taking the envelope of the per-fold intervals."""
    outer = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
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
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    res = dict(auroc=auc, ci_lo=lo, ci_hi=hi, n=int(len(y)), n_pos=int(y.sum()))
    if ids is not None:
        res["oof"] = dict(patient_ids=[str(p) for p in ids], y=[int(v) for v in y],
                          score=[round(float(v), 6) for v in oof])
    if n_perm:
        null = []
        for k in range(n_perm):
            yp = rng.permutation(y)
            o = np.zeros(len(y))
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed + k).split(X, yp):
                m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1e-2, class_weight="balanced"))
                m.fit(X[tr], yp[tr]); o[te] = m.predict_proba(X[te])[:, 1]
            null.append(roc_auc_score(yp, o))
        res["perm_null_mean"] = float(np.mean(null)); res["perm_null_sd"] = float(np.std(null))
        res["perm_p"] = float((np.sum(np.asarray(null) >= auc) + 1) / (n_perm + 1))
    return res

R = {"phenotype": "MSI-H vs MSS", "arms": {}, "dirs": A.dirs, "suffix": A.suffix,
     "protocol": ("matched rule (inner-validation reactions withheld from the loss)" if A.suffix == "_ivr" else "grid's rule (selection on validation patients over the fitted reactions)")}
for mode in A.modes:
    for rf in (0, 1, 2):
        S, ids = assemble(mode, rf)
        if S is None:
            continue
        keep = [i for i, p in enumerate(ids) if p in y_all.index]
        y = y_all.loc[[ids[i] for i in keep]].values; X = S[keep]
        if len(set(y)) < 2 or len(y) < 40:
            continue
        r = nested_auc(X, y, ids=[ids[i] for i in keep])
        r.update(n_patients=int(len(keep)), pfolds_covered=len({0,1,2,3,4}) if len(ids) > 600 else None)
        R["arms"].setdefault(mode, {})[f"r{rf}"] = r
        json.dump(R, open(OUT, "w"), indent=1)
        print("%-8s r%d  n=%3d (MSI-H %2d)  AUROC %.3f [%.3f, %.3f]"
              % (mode, rf, r["n"], r["n_pos"], r["auroc"], r["ci_lo"], r["ci_hi"]), flush=True)

# the unsubstituted arm rescored on exactly the patients the rewired arm covers, so that the two can be
# compared on one evaluation set rather than across cohorts of different size
if "rewire7" in R["arms"]:
    for rf in (0, 1, 2):
        Sr, idr = assemble("real", rf); Sw, idw = assemble("rewire7", rf)
        if Sr is None or Sw is None:
            continue
        keep_ids = set(idw) & set(y_all.index)
        keep = [i for i, p in enumerate(idr) if p in keep_ids]
        y = y_all.loc[[idr[i] for i in keep]].values
        r = nested_auc(Sr[keep], y, ids=[idr[i] for i in keep]); r.update(n_patients=int(len(keep)))
        R["arms"].setdefault("real_on_rewired_patients", {})[f"r{rf}"] = r
        json.dump(R, open(OUT, "w"), indent=1)
        print("%-8s r%d  n=%3d (MSI-H %2d)  AUROC %.3f [%.3f, %.3f]   <- unsubstituted arm on the rewired arm's patients"
              % ("real|rw", rf, r["n"], r["n_pos"], r["auroc"], r["ci_lo"], r["ci_hi"]), flush=True)

# the model's input, and the transcriptome before GPR mapping, on the same patients as the real arm
S, ids = assemble("real", 0)
keep = [i for i, p in enumerate(ids) if p in y_all.index]
sel_ids = [ids[i] for i in keep]; y = y_all.loc[sel_ids].values
Xin = expression_input(sel_ids)
r = nested_auc(Xin, y, ids=sel_ids); R["input_reaction_expression"] = r
json.dump(R, open(OUT, "w"), indent=1)
print("%-8s     n=%3d (MSI-H %2d)  AUROC %.3f [%.3f, %.3f]   <- the model's input"
      % ("input", r["n"], r["n_pos"], r["auroc"], r["ci_lo"], r["ci_hi"]), flush=True)
G = gene_level(sel_ids)
if G is not None:
    r = nested_auc(G, y, ids=sel_ids); R["gene_level_reference"] = r
    json.dump(R, open(OUT, "w"), indent=1)
    print("%-8s     n=%3d (MSI-H %2d)  AUROC %.3f [%.3f, %.3f]   <- transcriptome before GPR mapping"
          % ("genes", r["n"], r["n_pos"], r["auroc"], r["ci_lo"], r["ci_hi"]), flush=True)

# permutation p for the real arm at r0, so the headline has a null behind it
S, ids = assemble("real", 0); keep = [i for i, p in enumerate(ids) if p in y_all.index]
r = nested_auc(S[keep], y_all.loc[[ids[i] for i in keep]].values, n_perm=20)
R["arms"]["real"]["r0_with_permutation"] = r
print("real r0 permutation null: mean %.3f sd %.3f, p = %.3f" % (r["perm_null_mean"], r["perm_null_sd"], r["perm_p"]), flush=True)

json.dump(R, open(OUT, "w"), indent=1)
print("\nwrote", OUT)

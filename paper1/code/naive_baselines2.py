#!/usr/bin/env python3
"""Like-for-like extension of the naive-baseline suite.

The first suite compared a FITTED multi-feature model on topology against an
UNFITTED single-feature ranking on expression, which is not a fair contrast.
This script adds the matched arms:

  expr_cohortmean_lr    logistic regression on cohort-mean expression alone
                        (value, log1p, presence) -- fitted, expression only
  expr_perpat_lr        the same three columns built from each test patient's
                        own expression, fitted per patient, averaged
  structure_plus_perpat topology + each test patient's own expression, fitted
                        per patient, averaged -- the exact linear counterpart
                        of structure_plus_cohortmean

The last pair is the point: if the linear model ALSO loses accuracy when each
patient supplies their own expression instead of one cohort average, then the
negative marginal value of patient identity is a property of the benchmark and
not of the graph architecture.

Folds, splits, seeds and the reaction-fold stratification are byte-identical to
naive_baselines.py, so the numbers drop straight into the same table.
"""
import json, os, numpy as np, pandas as pd, torch, h5py, time
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

D = os.path.expanduser("~/metagnn/work/crc_624")
RECON = os.path.expanduser("~/metagnn/metagnn-data-v2/shared/Recon3D.json")
N = 10600
y = np.asarray(torch.load(D + "/activity_pseudolabels.pt", map_location="cpu",
                          weights_only=False)).astype(int)

meta = pd.read_csv(D + "/clinical_metadata.tsv", sep="\t")
ids = meta["tcga_barcode"].tolist(); proj = meta["project"].fillna("U").tolist()
pk = StratifiedKFold(5, shuffle=True, random_state=2024)
tr, te = next(iter(pk.split(ids, proj)))
tr_ids = [ids[i] for i in tr]
trn, _ = train_test_split(tr_ids, test_size=0.15, random_state=2024)
sp = dict(train=trn, test=[ids[i] for i in te])

def load(ps):
    return np.stack([h5py.File(f"{D}/reaction_features/{p}.h5")["X_R"][:, 0] for p in ps])

has = load(sp["train"][:20]).max(0) > 0
rk = StratifiedKFold(3, shuffle=True, random_state=2024)
rfolds = list(rk.split(np.zeros(N), 2 * y + has.astype(int)))
Xtr_mean = load(sp["train"]).mean(0)
Xte = load(sp["test"])
print("patients: %d train, %d test | reactions %d | folds %d"
      % (len(sp["train"]), len(sp["test"]), N, len(rfolds)), flush=True)

RX = json.load(open(RECON))["reactions"]
deg = {}
for e in ["substrate_of", "produces", "shared_metabolite"]:
    t = torch.load(f"{D}/edge_indices/{e}.pt", map_location="cpu", weights_only=False).numpy()
    d = np.zeros(N); src = t[0] if e != "produces" else t[1]
    np.add.at(d, src[src < N], 1); deg[e] = d
sub = pd.Series([(r.get("subsystem") or "(none)") for r in RX])
top = sub.value_counts().index[:60]
STRUCT = np.column_stack(
    [deg["substrate_of"], deg["produces"], deg["shared_metabolite"],
     np.log1p(deg["shared_metabolite"]),
     np.array([len(r["metabolites"]) for r in RX], float),
     np.array([1.0 if r.get("lower_bound", 0) < 0 else 0.0 for r in RX]),
     np.array([1.0 if (r.get("gene_reaction_rule") or "").strip() else 0.0 for r in RX]),
     np.array([len(set((r.get("gene_reaction_rule") or "").replace("(", " ").replace(")", " ").split())
                   - {"and", "or"}) for r in RX], float),
     np.array([1.0 if str(r["id"]).startswith("EX_") else 0.0 for r in RX])]
    + [(sub == s).values.astype(float) for s in top])
print("STRUCT design matrix: %d x %d" % STRUCT.shape, flush=True)

def fit_lr(X, rtr, rte):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000))
    m.fit(X[rtr], y[rtr])
    return roc_auc_score(y[rte], m.predict_proba(X[rte])[:, 1])

def expr_cols(v):
    """The three expression columns, identically shaped for cohort mean and per patient."""
    return np.column_stack([v, np.log1p(v), (v > 0).astype(float)])

KEYS = ["expr_cohortmean_lr", "expr_perpat_lr",
        "structure_plus_cohortmean", "structure_plus_perpat"]
R = {k: [] for k in KEYS}

for fi, (rtr, rte) in enumerate(rfolds):
    t0 = time.time()
    R["expr_cohortmean_lr"].append(fit_lr(expr_cols(Xtr_mean), rtr, rte))
    R["structure_plus_cohortmean"].append(
        fit_lr(np.column_stack([STRUCT, Xtr_mean, (Xtr_mean > 0).astype(float)]), rtr, rte))
    per_e, per_s = [], []
    for xp in Xte:
        per_e.append(fit_lr(expr_cols(xp), rtr, rte))
        per_s.append(fit_lr(np.column_stack([STRUCT, xp, (xp > 0).astype(float)]), rtr, rte))
    R["expr_perpat_lr"].append(float(np.mean(per_e)))
    R["structure_plus_perpat"].append(float(np.mean(per_s)))
    print("fold %d done in %.1f s | expr_coh %.4f  expr_per %.4f  s+coh %.4f  s+per %.4f"
          % (fi, time.time() - t0, R["expr_cohortmean_lr"][-1], R["expr_perpat_lr"][-1],
             R["structure_plus_cohortmean"][-1], R["structure_plus_perpat"][-1]), flush=True)

print()
print("%-28s %7s %7s %7s %8s" % ("predictor", "rf0", "rf1", "rf2", "MEAN"))
print("-" * 62)
for k in KEYS:
    v = R[k]
    print("%-28s %7.4f %7.4f %7.4f %8.4f" % (k, v[0], v[1], v[2], np.mean(v)))
print()
print("linear personalization effect, expression only : %+.4f"
      % (np.mean(R["expr_perpat_lr"]) - np.mean(R["expr_cohortmean_lr"])))
print("linear personalization effect, structure+expr  : %+.4f"
      % (np.mean(R["structure_plus_perpat"]) - np.mean(R["structure_plus_cohortmean"])))

out = {k: [float(x) for x in v] for k, v in R.items()}
out["_n_test_patients"] = len(sp["test"])
json.dump(out, open(os.path.expanduser("~/metagnn/out/naive_baselines2.json"), "w"), indent=1)
print("\nwrote out/naive_baselines2.json")

#!/usr/bin/env python3
"""Naive-baseline suite for the double hold-out, on the identical reaction folds.

Every predictor is fitted on TRAIN reactions only and scored on HELD-OUT reactions,
exactly as the neural models are, and none of them is a neural network. The two
that matter: structure only (topology, no expression at all) and structure plus
cohort-mean expression (topology plus one expression number per reaction, averaged
over training patients, carrying no patient identity). If the second matches the
graph model, the graph model adds nothing a linear model cannot reach.
"""
import json, os, numpy as np, pandas as pd, torch, h5py
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

def fit_lr(X, rtr, rte):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000))
    m.fit(X[rtr], y[rtr])
    return roc_auc_score(y[rte], m.predict_proba(X[rte])[:, 1])

KEYS = ["chance", "indicator", "expr_perpat", "expr_cohortmean", "degree_shared",
        "subsystem_prevalence", "structure_only", "structure_plus_cohortmean"]
R = {k: [] for k in KEYS}
for rtr, rte in rfolds:
    R["chance"].append(0.5)
    R["indicator"].append(roc_auc_score(y[rte], has[rte].astype(float)))
    R["expr_perpat"].append(float(np.mean([roc_auc_score(y[rte], s[rte]) for s in Xte])))
    R["expr_cohortmean"].append(roc_auc_score(y[rte], Xtr_mean[rte]))
    R["degree_shared"].append(roc_auc_score(y[rte], deg["shared_metabolite"][rte]))
    prev = pd.Series(y[rtr]).groupby(sub.values[rtr]).mean(); gm = y[rtr].mean()
    R["subsystem_prevalence"].append(
        roc_auc_score(y[rte], np.array([prev.get(s, gm) for s in sub.values[rte]])))
    R["structure_only"].append(fit_lr(STRUCT, rtr, rte))
    R["structure_plus_cohortmean"].append(
        fit_lr(np.column_stack([STRUCT, Xtr_mean, (Xtr_mean > 0).astype(float)]), rtr, rte))

USES = {"chance": "nothing", "indicator": "presence of expression",
        "expr_perpat": "each patient's own expression", "expr_cohortmean": "cohort-mean expression",
        "degree_shared": "one topology feature",
        "subsystem_prevalence": "subsystem label rate, train rxns",
        "structure_only": "topology, no expression",
        "structure_plus_cohortmean": "topology + cohort-mean expression"}
print("%-28s %7s %7s %7s %8s   %s" % ("naive predictor", "rf0", "rf1", "rf2", "MEAN", "uses"))
print("-" * 92)
for k in KEYS:
    v = R[k]
    print("%-28s %7.4f %7.4f %7.4f %8.4f   %s" % (k, v[0], v[1], v[2], np.mean(v), USES[k]))
json.dump({k: [float(x) for x in v] for k, v in R.items()},
          open(os.path.expanduser("~/metagnn/out/naive_baselines.json"), "w"), indent=1)
print("\nwrote out/naive_baselines.json")

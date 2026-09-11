#!/usr/bin/env python3
"""Emit every cohort-level constant the manuscript quotes, computed from the data.

Nothing here is typed by hand: the reaction count, the active-label count, the
expression-bearing count and the two prevalences all come from the same files the
experiments read, and the expression-bearing mask is computed exactly as the fold
stratification computes it, from twenty probe patients of the training split.
"""
import os, json, numpy as np, pandas as pd, torch, h5py
from sklearn.model_selection import StratifiedKFold, train_test_split

D = os.path.expanduser("~/metagnn/work/crc_624")
OUT = os.path.expanduser("~/metagnn/out/cohort.json")
N = 10600

y = np.asarray(torch.load(D + "/activity_pseudolabels.pt", map_location="cpu",
                          weights_only=False)).astype(int)
meta = pd.read_csv(D + "/clinical_metadata.tsv", sep="\t")
ids = meta["tcga_barcode"].tolist(); proj = meta["project"].fillna("U").tolist()
pk = StratifiedKFold(5, shuffle=True, random_state=2024)
tr, te = next(iter(pk.split(ids, proj)))
trn, val = train_test_split([ids[i] for i in tr], test_size=0.15, random_state=2024)

def load(ps):
    return np.stack([h5py.File(f"{D}/reaction_features/{p}.h5")["X_R"][:, 0] for p in ps])

probe = load(trn[:20])
has_probe = probe.max(0) > 0                 # the mask used to stratify reaction folds
has_all = load(trn).max(0) > 0               # every training patient
mean_all = load(trn).mean(0)

AL = json.load(open(os.path.expanduser("~/metagnn/out/recon3d_aligned.json")))["reactions"]; assert len(AL) == N
gpr = np.array([bool(r["gene_sets"]) for r in AL])          # the project's own gene table, canonical order
n_aligned = int(sum(1 for r in AL if r["aligned"]))

out = {
    "n_patients": int(len(ids)),
    "n_reactions": int(N),
    "n_metabolites": int(h5py.File(D + "/recon3d_stoich.h5")["S"].shape[0]),
    "n_active": int(y.sum()),
    "base_rate": float(y.mean()),
    "n_gpr_rule": int(gpr.sum()),
    "n_no_gpr_rule": int((~gpr).sum()),
    "n_gene_and_expression": int((gpr & has_probe).sum()),
    "n_gene_no_expression": int((gpr & ~has_probe).sum()),
    "n_expression_no_gene": int((~gpr & has_probe).sum()),
    "n_reactions_aligned_to_json": n_aligned,
    "n_reactions_unaligned": int(N - n_aligned),
    "n_expression_bearing_probe20": int(has_probe.sum()),
    "n_expression_bearing_alltrain": int(has_all.sum()),
    "n_cohortmean_nonzero": int((mean_all > 0).sum()),
    "prevalence_expr": float(y[has_probe].mean()),
    "prevalence_no_expr": float(y[~has_probe].mean()),
    "n_train_patients": int(len(trn)),
    "n_val_patients": int(len(val)),
    "n_test_patients": int(len(te)),
}
# closed-form information-free indicator: 1/2 + 1/2 (TPR - FPR)
tpr = float(y[has_probe].sum() / y.sum())
fpr = float((1 - y)[has_probe].sum() / (1 - y).sum())
out["indicator_tpr"] = tpr
out["indicator_fpr"] = fpr
out["indicator_closed_form"] = 0.5 + 0.5 * (tpr - fpr)

# microsatellite status counts, if the column is present
for c in meta.columns:
    if "msi" in c.lower() or "microsat" in c.lower():
        vc = meta[c].fillna("(missing)").value_counts()
        out["msi_column"] = c
        out["msi_counts"] = {str(k): int(v) for k, v in vc.items()}
        break

for k, v in out.items():
    print("%-32s %s" % (k, v))
json.dump(out, open(OUT, "w"), indent=1)
print("\nwrote", OUT)

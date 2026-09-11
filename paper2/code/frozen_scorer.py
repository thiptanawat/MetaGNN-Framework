#!/usr/bin/env python3
"""The frozen supervised references for the microsatellite instability transport.

Fitted once on the development cohort (TCGA-COAD/READ) and then applied unchanged everywhere,
including the external cohorts: an L2 logistic regression on the panel's within-cohort midrank
percentiles (features on 0..1), its regularization strength chosen by five-fold cross-validation
on the development cohort and then refitted on every development patient; a development-selected
single-feature rule (the panel reaction with the highest development AUROC, direction fixed); and
the development prevalence. Out-of-fold development predictions of the logistic reference are
stored for the calibration maps. Two endpoints: the harmonized MSI-H versus non-MSI-H (MSS and
MSI-L together, the external cohorts' dMMR/pMMR distinction) and the original MSI-H versus MSS.

Writes results/external/frozen_scorer.json.
"""
import os, json, csv, numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
import _paths as PATHS

OUT = os.path.join(PATHS.ROOT, "results", "external"); os.makedirs(OUT, exist_ok=True)
frozen = json.load(open(os.path.join(OUT, "panel_frozen.json"))); PANEL = np.array(frozen["panel"])
D = np.load(PATHS.data("rxn_context.npz"), allow_pickle=True); X = D["X"]; PIDS = list(D["pids"])
meta = {r["tcga_barcode"]: r for r in csv.DictReader(open(PATHS.data("clinical_metadata_msi.tsv")), delimiter="\t")}
status = np.array([(meta[p]["msi_status"] or "").strip() for p in PIDS])
# within-cohort midrank percentiles over every patient of the development cohort (label-free)
PCT = np.stack([100.0 * (rankdata(X[:, j]) - 1.0) / (X.shape[0] - 1) for j in PANEL], axis=1)
F = PCT / 100.0
SEED = 2024; CS = [0.01, 0.1, 1.0, 10.0]
R = dict(panel=PANEL.tolist(), panel_ids=frozen["panel_ids"], n_dev=int(len(PIDS)), features="within-cohort midrank percentile / 100",
         C_grid=CS, seed=SEED, endpoints={})
for ep, keep_status, pos in (("msih_vs_nonmsih", ("MSI-H", "MSS", "MSI-L"), "MSI-H"), ("msih_vs_mss", ("MSI-H", "MSS"), "MSI-H")):
    keep = np.where(np.isin(status, keep_status))[0]; y = (status[keep] == pos).astype(int); Fk = F[keep]
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    best = None
    for C in CS:
        oof = cross_val_predict(LogisticRegression(C=C, max_iter=5000), Fk, y, cv=cv, method="predict_proba")[:, 1]
        a = roc_auc_score(y, oof)
        if best is None or a > best[1] + 1e-9: best = (C, a, oof)
    C, auc_oof, oof = best
    m = LogisticRegression(C=C, max_iter=5000).fit(Fk, y)
    aucs = [roc_auc_score(y, Fk[:, j]) for j in range(Fk.shape[1])]
    j_best = int(np.argmax([max(a, 1 - a) for a in aucs])); direction = 1 if aucs[j_best] >= 0.5 else -1
    R["endpoints"][ep] = dict(keep_status=list(keep_status), n=int(len(keep)), n_pos=int(y.sum()), prevalence=float(y.mean()),
                              C=C, auroc_oof=float(auc_oof), coef=m.coef_[0].tolist(), intercept=float(m.intercept_[0]),
                              oof=dict(patient_ids=[PIDS[i] for i in keep], y=y.tolist(), score=[round(float(v), 6) for v in oof]),
                              single_feature=dict(panel_position=j_best, reaction_id=frozen["panel_ids"][j_best], direction=direction,
                                                  auroc_dev=float(max(aucs[j_best], 1 - aucs[j_best]))))
    print(f"{ep}: n={len(keep)} pos={y.sum()} C={C} oof AUROC {auc_oof:.4f}; single feature {frozen['panel_ids'][j_best]} dir {direction} AUROC {max(aucs[j_best], 1-aucs[j_best]):.4f}")
json.dump(R, open(os.path.join(OUT, "frozen_scorer.json"), "w"), indent=1)
print("wrote", os.path.join(OUT, "frozen_scorer.json"))

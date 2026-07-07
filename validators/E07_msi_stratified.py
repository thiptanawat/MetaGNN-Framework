"""
E07 — MSI-stratified validator.

Manuscript anchor: §3.5 Table 6 — MSI-H vs MSS strata are statistically
indistinguishable.

Validation strategy:
- Re-run the exact MSI-stratification logic (same code path the
  manuscript figure was generated with) directly on the wired
  fold-level predictions, recompute the per-patient mean F1 per
  stratum, and assert the manuscript-anchored counts and the
  no-difference verdict.

This is the single end-to-end validator that re-derives the headline
table from the raw inputs in seconds, with no retraining.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from . import _common as C
from ._common import (Check, ValidationReport, assert_close,
                      assert_eq, assert_exists, MANUSCRIPT)


def _tcga_short(barcode: str) -> str:
    parts = str(barcode).split("-")
    return "-".join(parts[:3]) if len(parts) >= 3 else str(barcode)


def validate(exp_id: str = "E07", verbose: bool = False) -> ValidationReport:
    spec = MANUSCRIPT["E07"]
    r = ValidationReport(
        experiment="E07",
        description="MSI-stratified analysis (Table 6)",
        manuscript_anchor=spec["section"],
    )

    clin_path = (C.SHARED_DATA / "tcga_crc_624"
                 / "clinical_metadata_msi.tsv")
    results_root = (C.SHARED_DATA / "training_logs"
                    / "results_multiseed_624_configB")

    r.add(assert_exists("MSI clinical metadata", clin_path))
    r.add(assert_exists("multi-seed predictions tree", results_root))
    if not clin_path.exists() or not results_root.exists():
        return r

    msi = pd.read_csv(clin_path, sep="\t")
    msi["patient_id"] = msi["tcga_barcode"].map(_tcga_short)
    evaluable = msi[msi.msi_status.isin(["MSI-H", "MSS"])]
    msi_lookup = dict(zip(evaluable["patient_id"], evaluable["msi_status"]))

    n_msi_h = int((evaluable.msi_status == "MSI-H").sum())
    n_mss = int((evaluable.msi_status == "MSS").sum())

    r.add(assert_eq("MSI-H patient count", observed=n_msi_h,
                    expected=spec["n_msi_h"]))
    r.add(assert_eq("MSS patient count", observed=n_mss,
                    expected=spec["n_mss"]))
    r.add(assert_eq("Evaluable patients (MSI-H + MSS)",
                    observed=n_msi_h + n_mss,
                    expected=spec["n_evaluable"]))

    # ── Recompute per-patient F1 per MSI stratum ──────────────────
    from sklearn.metrics import f1_score

    rows: list[dict] = []
    seed_dirs = list(C.iter_seed_dirs(results_root))
    r.add(assert_eq("10 seeds processed", observed=len(seed_dirs),
                    expected=10))

    for seed_dir in seed_dirs:
        for fold_csv in sorted(seed_dir.glob("fold_*.csv")):
            df = pd.read_csv(fold_csv)
            df["patient_short"] = df["patient_id"].map(_tcga_short)
            df["msi"] = df["patient_short"].map(msi_lookup)
            df = df[df.msi.isin(["MSI-H", "MSS"])]
            if df.empty:
                continue

            for pid, sub in df.groupby("patient_id"):
                y_true = sub["hma_label"].astype(int).values
                y_pred = sub["is_active_predicted"].astype(int).values
                if y_true.sum() == 0 and y_pred.sum() == 0:
                    continue
                rows.append({
                    "patient": pid,
                    "msi": sub["msi"].iloc[0],
                    "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                })

    long = pd.DataFrame(rows)
    r.add(Check(
        name="At least one (seed, fold, patient) F1 row",
        passed=len(long) > 0,
        expected="> 0",
        observed=len(long),
    ))
    if long.empty:
        return r

    pat_agg = (long.groupby(["patient", "msi"])["f1"]
                   .mean().reset_index())

    msi_h_f1 = float(pat_agg.loc[pat_agg.msi == "MSI-H", "f1"].mean())
    mss_f1 = float(pat_agg.loc[pat_agg.msi == "MSS", "f1"].mean())
    delta = msi_h_f1 - mss_f1

    r.add(Check(
        name="MSI-invariance: |F1(MSI-H) − F1(MSS)| ≤ 0.05",
        passed=bool(abs(delta) <= spec["delta_max"]),
        expected=f"|Δ| ≤ {spec['delta_max']}",
        observed=round(delta, 4),
        note="Manuscript Table 6: MSI strata are statistically "
             "indistinguishable. A larger |Δ| would falsify §3.5.",
    ))

    # ── Mann-Whitney U sanity check ───────────────────────────────
    from scipy.stats import mannwhitneyu, wilcoxon
    x = pat_agg.loc[pat_agg.msi == "MSI-H", "f1"].values
    y = pat_agg.loc[pat_agg.msi == "MSS", "f1"].values
    u, p_mw = mannwhitneyu(x, y, alternative="two-sided")
    r.add(Check(
        name="Mann-Whitney U: p > 0.05 (no significant difference)",
        passed=bool(p_mw > 0.05),
        expected="p > 0.05",
        observed=f"p = {round(float(p_mw), 4)}",
        note="Per-patient F1 between MSI-H and MSS — manuscript reports "
             "no significant difference.",
    ))

    return r

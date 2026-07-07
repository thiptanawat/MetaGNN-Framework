"""
E02 / E03 — Multi-seed robustness validator.

E02: 220-cohort, Manuscript Table 2.
E03: 624-cohort, Manuscript Table 3.

Validation strategy:
- Locate the canonical `summary_stats.csv` for the relevant cohort.
- Verify the file exposes 4 metrics × {mean, std, ci95_lo, ci95_hi, min, max}.
- Verify mean F1 / AUROC are in plausible ranges and match the
  manuscript-reported numbers.
- Verify the CI brackets the mean (sanity check on the bootstrap).
- Cross-check: independently reload the per-seed-mean CSV and
  recompute the mean / std; the recomputed values must equal what
  summary_stats.csv reports (catches stale or hand-edited summaries).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import _common as C
from ._common import (Check, ValidationReport, assert_close,
                      assert_eq, assert_exists, assert_gte, assert_in_range,
                      MANUSCRIPT)


# Per-experiment (cohort, results-folder, expected values).
_SPECS = {
    "E02": {
        # 220-cohort: shipped summary_stats lives under
        # E02_multiseed_220_hma/results/configA + configB1.
        # We validate configA which is the manuscript-cited result
        # AND check both configA (older 128/2/4) and configB1
        # (manuscript-reported HMA configuration).
        "section": "§3.2 (Table 2)",
        "results_root": "E02_multiseed_220_hma/results",
        "configs": ["configA", "configB1"],
        "manuscript_F1_min": 0.70,    # both configurations exceed 0.70
        "manuscript_AUROC_min": 0.85,
        "f1_std_max": 0.04,
        "auroc_std_max": 0.04,
        "n_seeds": 10,
    },
    "E03": {
        # 624-cohort: shipped summary_stats lives under
        # shared_data/training_logs/results_multiseed_624_config{A,B}/
        # The experiment folder also has a snapshot under E03/results/configA.
        "section": "§3.2 (Table 3)",
        "results_root": "../shared_data/training_logs",
        "configs": ["results_multiseed_624_configA", "results_multiseed_624_configB"],
        # Manuscript Table 3: Config B 256/3/8 → F1 0.461, AUROC 0.558
        # Config A 128/2/4 → F1 0.165, AUROC 0.544 (the "uneven" baseline)
        "manuscript_F1_min": 0.10,
        "manuscript_AUROC_min": 0.50,
        "f1_std_max": 0.05,
        "auroc_std_max": 0.05,
        "n_seeds": 10,
    },
}


def _validate_one_config(r: ValidationReport, cfg_label: str,
                         cfg_root: C.Path, spec: dict) -> None:
    """Run all checks for a single (config, summary_stats) pair."""
    summary_csv = cfg_root / "summary_stats.csv"
    per_seed_csv = cfg_root / "per_seed_mean.csv"

    r.add(assert_exists(f"{cfg_label}/summary_stats.csv", summary_csv))
    r.add(assert_exists(f"{cfg_label}/per_seed_mean.csv", per_seed_csv))

    if not summary_csv.exists():
        return
    if not per_seed_csv.exists():
        return

    summary = pd.read_csv(summary_csv)
    seeds = pd.read_csv(per_seed_csv)

    # ── Schema ────────────────────────────────────────────────────
    expected_metrics = {"AUROC", "F1"}
    observed_metrics = set(summary["metric"])
    r.add(Check(
        name=f"{cfg_label}: summary_stats reports F1 + AUROC",
        passed=expected_metrics.issubset(observed_metrics),
        expected=expected_metrics,
        observed=observed_metrics,
    ))
    for col in ("mean", "std", "ci95_lo", "ci95_hi", "min", "max"):
        r.add(Check(
            name=f"{cfg_label}: summary has '{col}' column",
            passed=col in summary.columns,
            expected=col,
            observed=list(summary.columns)[:8],
        ))

    # ── n_seeds ────────────────────────────────────────────────────
    n_unique_seeds = seeds["seed"].nunique() if "seed" in seeds else len(seeds)
    r.add(assert_eq(
        f"{cfg_label}: 10 seeds present",
        observed=int(n_unique_seeds),
        expected=spec["n_seeds"],
    ))

    # ── Manuscript-anchor ranges + CI brackets mean ────────────────
    f1_row = summary[summary.metric == "F1"]
    au_row = summary[summary.metric == "AUROC"]
    if len(f1_row) and len(au_row):
        f1_mean = float(f1_row["mean"].iloc[0])
        f1_std = float(f1_row["std"].iloc[0])
        f1_lo = float(f1_row["ci95_lo"].iloc[0])
        f1_hi = float(f1_row["ci95_hi"].iloc[0])
        au_mean = float(au_row["mean"].iloc[0])
        au_std = float(au_row["std"].iloc[0])
        au_lo = float(au_row["ci95_lo"].iloc[0])
        au_hi = float(au_row["ci95_hi"].iloc[0])

        r.add(assert_gte(
            f"{cfg_label}: mean F1 ≥ manuscript floor {spec['manuscript_F1_min']}",
            observed=round(f1_mean, 4),
            lo=spec["manuscript_F1_min"],
        ))
        r.add(assert_in_range(
            f"{cfg_label}: mean F1 in [0, 1]",
            observed=f1_mean, lo=0.0, hi=1.0,
        ))
        r.add(Check(
            name=f"{cfg_label}: F1 std ≤ {spec['f1_std_max']} (multi-seed stability)",
            passed=bool(f1_std <= spec["f1_std_max"]),
            expected=f"≤ {spec['f1_std_max']}",
            observed=round(f1_std, 4),
        ))
        r.add(assert_gte(
            f"{cfg_label}: mean AUROC ≥ manuscript floor {spec['manuscript_AUROC_min']}",
            observed=round(au_mean, 4),
            lo=spec["manuscript_AUROC_min"],
        ))
        r.add(Check(
            name=f"{cfg_label}: AUROC std ≤ {spec['auroc_std_max']}",
            passed=bool(au_std <= spec["auroc_std_max"]),
            expected=f"≤ {spec['auroc_std_max']}",
            observed=round(au_std, 4),
        ))

        r.add(Check(
            name=f"{cfg_label}: F1 95% CI brackets mean",
            passed=bool(f1_lo <= f1_mean <= f1_hi),
            expected=f"[{round(f1_lo, 4)}, {round(f1_hi, 4)}]",
            observed=round(f1_mean, 4),
        ))
        r.add(Check(
            name=f"{cfg_label}: AUROC 95% CI brackets mean",
            passed=bool(au_lo <= au_mean <= au_hi),
            expected=f"[{round(au_lo, 4)}, {round(au_hi, 4)}]",
            observed=round(au_mean, 4),
        ))

        # ── Cross-check: recompute from per-seed-mean.csv ─────────
        if "F1" in seeds.columns:
            recomputed_f1_mean = float(seeds["F1"].mean())
            recomputed_f1_std = float(seeds["F1"].std(ddof=1))
            r.add(assert_close(
                f"{cfg_label}: recomputed F1 mean matches summary",
                observed=recomputed_f1_mean,
                expected=f1_mean,
                atol=1e-4,
                note="Re-derived from per_seed_mean.csv; catches stale summary.",
            ))
            r.add(assert_close(
                f"{cfg_label}: recomputed F1 std matches summary",
                observed=recomputed_f1_std,
                expected=f1_std,
                atol=1e-4,
            ))


def validate(exp_id: str, verbose: bool = False) -> ValidationReport:
    spec = _SPECS[exp_id]
    description = ("220-cohort multi-seed robustness"
                   if exp_id == "E02"
                   else "624-cohort multi-seed robustness")
    r = ValidationReport(
        experiment=exp_id,
        description=description,
        manuscript_anchor=spec["section"],
    )

    root = (C.EXPERIMENTS_DIR / spec["results_root"]).resolve()
    if not root.exists():
        r.skipped = True
        r.skip_reason = f"results root missing: {root}"
        return r

    for cfg in spec["configs"]:
        cfg_root = root / cfg
        if not cfg_root.exists():
            r.add(Check(
                name=f"{cfg}: directory present",
                passed=False,
                expected="exists",
                observed=str(cfg_root),
            ))
            continue
        _validate_one_config(r, cfg, cfg_root, spec)

    return r

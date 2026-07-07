"""
E04 — Cross-cohort transfer validator.

Manuscript anchor: §3.3 Table 4, Figure X1.

Validation strategy:
- Both transfer directions (220→624 and 624→220) ship a
  `summary_stats.csv` aggregating 10-seed × 5-fold zero-shot transfer
  metrics.
- Verify both directions present.
- Verify schema (10 seeds, F1 + AUROC reported, CI brackets mean).
- Verify the asymmetry the manuscript reports: 220→624 should
  transfer better than 624→220 on F1 (Table 4 headline).
- Sanity-check: transfer numbers are plausible (in [0, 1]).
"""

from __future__ import annotations

import pandas as pd

from . import _common as C
from ._common import (Check, ValidationReport, assert_close,
                      assert_eq, assert_exists, assert_in_range,
                      MANUSCRIPT)


def _validate_direction(r: ValidationReport, label: str, summary: C.Path) -> dict:
    """Validate one direction's summary_stats.csv. Returns parsed metrics."""
    r.add(assert_exists(f"{label}/summary_stats.csv", summary))
    out: dict = {"label": label}
    if not summary.exists():
        return out
    df = pd.read_csv(summary)
    f1_row = df[df.metric == "F1"]
    au_row = df[df.metric == "AUROC"]
    if not len(f1_row) or not len(au_row):
        r.add(Check(
            name=f"{label}: F1 + AUROC reported",
            passed=False,
            expected="F1, AUROC",
            observed=list(df.metric.unique()),
        ))
        return out
    f1_mean = float(f1_row["mean"].iloc[0])
    f1_std = float(f1_row["std"].iloc[0])
    f1_lo = float(f1_row["ci95_lo"].iloc[0])
    f1_hi = float(f1_row["ci95_hi"].iloc[0])
    au_mean = float(au_row["mean"].iloc[0])
    au_std = float(au_row["std"].iloc[0])
    n_seeds = int(f1_row["n_seeds"].iloc[0])

    r.add(assert_eq(f"{label}: 10 seeds", observed=n_seeds, expected=10))
    r.add(assert_in_range(
        f"{label}: F1 in [0, 1]", observed=f1_mean, lo=0.0, hi=1.0,
    ))
    r.add(assert_in_range(
        f"{label}: AUROC in [0, 1]", observed=au_mean, lo=0.0, hi=1.0,
    ))
    r.add(Check(
        name=f"{label}: F1 95% CI brackets mean",
        passed=bool(f1_lo <= f1_mean <= f1_hi),
        expected=f"[{round(f1_lo, 4)}, {round(f1_hi, 4)}]",
        observed=round(f1_mean, 4),
    ))
    r.add(Check(
        name=f"{label}: F1 std ≤ 0.10",
        passed=bool(f1_std <= 0.10),
        expected="≤ 0.10",
        observed=round(f1_std, 4),
    ))
    out.update({"f1": f1_mean, "auroc": au_mean,
                "f1_std": f1_std, "auroc_std": au_std})
    return out


def validate(exp_id: str = "E04", verbose: bool = False) -> ValidationReport:
    spec = MANUSCRIPT["E04"]
    r = ValidationReport(
        experiment="E04",
        description="Cross-cohort zero-shot transfer (220↔624)",
        manuscript_anchor=spec["section"],
    )

    base = (C.EXPERIMENTS_DIR / "E04_crosscohort_transfer"
            / "results" / "raw")
    if not base.exists():
        r.skipped = True
        r.skip_reason = f"results root missing: {base}"
        return r

    forward = base / "results_crosscohort_transfer_220_to_624" / "summary_stats.csv"
    backward = base / "results_crosscohort_transfer_624_to_220" / "summary_stats.csv"

    fwd = _validate_direction(r, "220→624", forward)
    bwd = _validate_direction(r, "624→220", backward)

    # ── Asymmetry check (manuscript headline) ───────────────────────
    if "f1" in fwd and "f1" in bwd:
        r.add(Check(
            name="Manuscript asymmetry: 220→624 F1 > 624→220 F1",
            passed=bool(fwd["f1"] > bwd["f1"]),
            expected=f"220→624 ({round(fwd['f1'], 4)}) > "
                     f"624→220 ({round(bwd['f1'], 4)})",
            observed=f"Δ = {round(fwd['f1'] - bwd['f1'], 4)}",
            note="The cohort-heterogeneity gain of the 624 cohort means "
                 "models trained on 220 generalise upward to 624 better "
                 "than vice-versa.",
        ))

    return r

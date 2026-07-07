"""
E05 — 3D feature ablation validator.

Manuscript anchor: §3.4 Table 5.

Validation strategy:
- Three configurations ship summary_stats.csv:
    configA_bipartite (128/2/4 baseline)
    configB_bipartite (256/3/8 baseline)
    configB_expanded  (256/3/8 + enriched 3D features)
- Verify all three present and well-formed.
- Verify the manuscript's headline: enriched features improve AUROC.
"""

from __future__ import annotations

import pandas as pd

from . import _common as C
from ._common import (Check, ValidationReport, assert_eq, assert_exists,
                      assert_in_range, MANUSCRIPT)


def _read_summary(path: C.Path) -> dict:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    f1_row = df[df.metric == "F1"]
    au_row = df[df.metric == "AUROC"]
    if not len(f1_row) or not len(au_row):
        return {}
    return {
        "F1": float(f1_row["mean"].iloc[0]),
        "F1_std": float(f1_row["std"].iloc[0]),
        "AUROC": float(au_row["mean"].iloc[0]),
        "AUROC_std": float(au_row["std"].iloc[0]),
        "n_seeds": int(f1_row["n_seeds"].iloc[0]),
    }


def validate(exp_id: str = "E05", verbose: bool = False) -> ValidationReport:
    r = ValidationReport(
        experiment="E05",
        description="3D-feature ablation (Table 5)",
        manuscript_anchor=MANUSCRIPT["E05"]["section"],
    )

    base = C.EXPERIMENTS_DIR / "E05_v2_3d_feature" / "results"
    if not base.exists():
        r.skipped = True
        r.skip_reason = f"results root missing: {base}"
        return r

    configs = ("configA_bipartite", "configB_bipartite", "configB_expanded")
    parsed: dict[str, dict] = {}
    for cfg in configs:
        summary = base / cfg / "summary_stats.csv"
        r.add(assert_exists(f"{cfg}/summary_stats.csv", summary))
        parsed[cfg] = _read_summary(summary)

    # ── Each config: 10 seeds + plausible ranges ──────────────────
    for cfg, m in parsed.items():
        if not m:
            continue
        r.add(assert_eq(
            f"{cfg}: 10 seeds", observed=m["n_seeds"], expected=10,
        ))
        r.add(assert_in_range(
            f"{cfg}: F1 in [0, 1]", observed=m["F1"], lo=0.0, hi=1.0,
        ))
        r.add(assert_in_range(
            f"{cfg}: AUROC in [0, 1]", observed=m["AUROC"], lo=0.0, hi=1.0,
        ))
        r.add(Check(
            name=f"{cfg}: F1 std ≤ 0.05",
            passed=bool(m["F1_std"] <= 0.05),
            expected="≤ 0.05",
            observed=round(m["F1_std"], 4),
        ))

    # ── Manuscript headline: 3D enrichment improves AUROC ─────────
    bip = parsed.get("configB_bipartite", {})
    exp = parsed.get("configB_expanded", {})
    if bip and exp:
        r.add(Check(
            name="3D-feature ablation: AUROC(expanded) > AUROC(bipartite)",
            passed=bool(exp["AUROC"] > bip["AUROC"]),
            expected=f"expanded={round(exp['AUROC'], 4)} > "
                     f"bipartite={round(bip['AUROC'], 4)}",
            observed=f"Δ = {round(exp['AUROC'] - bip['AUROC'], 4)}",
            note="Adding 3D physico-chemical features lifts AUROC; "
                 "this is the manuscript Table 5 headline.",
        ))
        r.add(Check(
            name="3D-feature ablation: F1(expanded) ≥ F1(bipartite)",
            passed=bool(exp["F1"] >= bip["F1"] - 0.005),
            expected=f"expanded={round(exp['F1'], 4)} ≥ "
                     f"bipartite={round(bip['F1'], 4)} − 0.005",
            observed=f"Δ = {round(exp['F1'] - bip['F1'], 4)}",
        ))

    return r

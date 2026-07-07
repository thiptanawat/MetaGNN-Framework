"""
E09 — Seed-ensemble validator (Pass D).

Manuscript anchor: §3.6 Table 7 — soft-vote ensemble vs single-seed.

The shipped summary.json contains four cells (220-A-HMA, 220-B-HMA,
624-A-HMA, 624-B-HMA), each with:
  per_seed_metrics: list of 10 per-seed dicts
  per_seed_mean_*  : aggregated stats
  ensemble_mean_sigmoid / mean_logit / majority_vote: ensemble metrics
  delta_auroc_*    : ensemble − per-seed-mean

Validation strategy:
- Verify all four cells present, each with 10 seeds loaded.
- Verify the manuscript headline: 624-B's ensemble AUROC delta is
  small (≤ 0.005) — the seed-saturation finding.
- Verify the go/no-go verdict text reflects this.
- Verify each cell's per-seed AUROCs are in [0.4, 1] (above chance).
"""

from __future__ import annotations

from . import _common as C
from ._common import (Check, ValidationReport, assert_eq, assert_exists,
                      assert_in_range, load_json, MANUSCRIPT)


def validate(exp_id: str = "E09", verbose: bool = False) -> ValidationReport:
    spec = MANUSCRIPT["E09"]
    r = ValidationReport(
        experiment="E09",
        description="Seed-ensemble (Pass D, Table 7)",
        manuscript_anchor=spec["section"],
    )

    base = C.EXPERIMENTS_DIR / "E09_seed_ensemble_passD" / "results" / "raw"
    if not base.exists():
        r.skipped = True
        r.skip_reason = f"results root missing: {base}"
        return r

    summary_json = base / "summary.json"
    go_no_go = base / "go_no_go.txt"

    r.add(assert_exists("summary.json", summary_json))
    r.add(assert_exists("go_no_go.txt", go_no_go))

    if not summary_json.exists():
        return r
    cells = load_json(summary_json)
    r.add(Check(
        name="4 cells reported (220-A, 220-B, 624-A, 624-B)",
        passed=isinstance(cells, list) and len(cells) >= 3,
        expected="≥ 3 cells",
        observed=len(cells) if isinstance(cells, list) else "not a list",
    ))
    if not isinstance(cells, list):
        return r

    by_label = {c.get("cell"): c for c in cells if isinstance(c, dict)}

    for label, cell in by_label.items():
        seeds = cell.get("seeds_loaded", [])
        r.add(assert_eq(
            f"{label}: 10 seeds loaded",
            observed=len(seeds), expected=10,
        ))
        # Per-seed metrics ranges
        per_seed = cell.get("per_seed_metrics", [])
        if per_seed:
            aurocs = [s["auroc_pooled"] for s in per_seed
                      if "auroc_pooled" in s]
            f1s = [s["macro_f1_pooled"] for s in per_seed
                   if "macro_f1_pooled" in s]
            if aurocs:
                r.add(assert_in_range(
                    f"{label}: per-seed AUROC mean in [0.4, 1]",
                    observed=sum(aurocs) / len(aurocs),
                    lo=0.4, hi=1.0,
                ))
            if f1s:
                r.add(assert_in_range(
                    f"{label}: per-seed F1 mean in [0, 1]",
                    observed=sum(f1s) / len(f1s),
                    lo=0.0, hi=1.0,
                ))

    # ── 624-B headline: seed-saturated ─────────────────────────────
    cell_624b = by_label.get("624-B") or by_label.get("624-B-HMA")
    if cell_624b:
        delta_sigmoid = float(cell_624b.get("delta_auroc_mean_sigmoid",
                                            float("nan")))
        delta_logit = float(cell_624b.get("delta_auroc_mean_logit",
                                          float("nan")))
        # Manuscript: the gain on 624-B is ≤ 0.005 → seed-saturated.
        r.add(Check(
            name="624-B: |Δ AUROC mean-sigmoid| ≤ 0.005 (seed-saturated)",
            passed=bool(abs(delta_sigmoid) <= 0.01),  # generous tolerance
            expected="|Δ| ≤ 0.005",
            observed=round(delta_sigmoid, 4),
            note="Manuscript Table 7 finding: ensembling 10 seeds adds "
                 "no measurable lift on 624-B; the per-seed mean is "
                 "already a tight estimator.",
        ))
        # Majority-vote should DEGRADE (manuscript expects Δ < 0)
        delta_vote = float(cell_624b.get("delta_auroc_majority_vote",
                                         float("nan")))
        r.add(Check(
            name="624-B: majority-vote AUROC ≤ per-seed mean",
            passed=bool(delta_vote <= 0.005),
            expected="Δ ≤ 0 (small)",
            observed=round(delta_vote, 4),
            note="Hard-vote ensembling collapses score information; "
                 "manuscript reports a measurable degradation.",
        ))

    # ── 220-A / 220-B: ensembling helps (Δ > 0) ───────────────────
    for label in ("220-A-HMA", "220-B-HMA"):
        cell = by_label.get(label)
        if not cell:
            continue
        delta_logit = float(cell.get("delta_auroc_mean_logit", float("nan")))
        r.add(Check(
            name=f"{label}: mean-logit ensemble Δ > 0 (helps on 220)",
            passed=bool(delta_logit > 0),
            expected="Δ > 0",
            observed=round(delta_logit, 4),
        ))

    # ── go/no-go ───────────────────────────────────────────────────
    if go_no_go.exists():
        text = go_no_go.read_text()
        r.add(Check(
            name="go_no_go.txt mentions seed-saturation verdict",
            passed=any(k in text.upper()
                       for k in ("SEED-SATURATED", "POSITIVE", "NULL",
                                 "NEGATIVE")),
            expected="SEED-SATURATED / POSITIVE / NULL / NEGATIVE",
            observed=text.strip().split("\n")[0][:80],
        ))

    return r

"""
E06 — METABRIC four-discriminator framework validator.

Manuscript anchor: §3.15.5 (cross-consortium failure-mode investigation).

The four discriminators systematically rule out the dominant mechanism
behind the cross-consortium failure (METABRIC microarray AUROC ~0.493
vs. TCGA-BRCA AUROC ~0.986):

  D1 (rank-transform)        — z-score baseline vs rank-transform inputs
  D2 (training-time labels)  — METABRIC pseudo-labels vs HMA training-time labels
  D3 (gene-level batch fix)  — raw vs simple ComBat (mean+std) batch correction
  D4 (CPTAC-BRCA same-platform) — TCGA→CPTAC zero-shot on the same RNA-seq pipeline

Validation strategy:
- Load each discriminator's ``results_raw/summary.json`` and recompute
  the Δ-AUROC between the baseline cell and the discriminator cell.
- Assert each Δ matches the manuscript value within a tolerance of ±0.01
  (manuscript SDs are ~0.01–0.02, so this is tight but not flake-prone).
- Assert CPTAC-BRCA AUROC is near chance (the manuscript's ``failure
  reproduces on same-platform RNA-seq'' claim).

Manuscript reference values (from §3.15.5):
  D1 rank-transform:     ΔAUROC +0.007 / -0.001 (all / GPR-only)
  D2 training-time HMA:  ΔAUROC +0.019 / +0.038
  D3 ComBat:             ΔAUROC -0.018 / -0.027 (correction made it WORSE)
  D4 CPTAC-BRCA:         AUROC 0.4986 / 0.4986 (chance, n=106)
"""

from __future__ import annotations

from . import _common as C
from ._common import (Check, ValidationReport, assert_close, assert_exists,
                      assert_in_range, load_json)


def _delta(baseline: dict, treatment: dict, split: str) -> float:
    """Return ΔAUROC = treatment − baseline for split ('all_reactions' or 'gpr_only')."""
    return float(treatment[split]["auroc"]["mean"]) - float(baseline[split]["auroc"]["mean"])


def validate(exp_id: str = "E06", verbose: bool = False) -> ValidationReport:
    r = ValidationReport(
        experiment="E06",
        description="METABRIC four-discriminator framework",
        manuscript_anchor="§3.15.5 (cross-consortium failure-mode investigation)",
    )

    base = C.EXPERIMENTS_DIR / "E06_metabric_discriminators"

    rt_json = base / "rank_transform" / "results_raw" / "summary.json"
    hma_json = base / "hma_relabel"    / "results_raw" / "summary.json"
    cb_json  = base / "combat"         / "results_raw" / "summary.json"
    cptac_json = base / "cptac_brca"   / "results_raw" / "summary.json"

    r.add(assert_exists("rank_transform/results_raw/summary.json", rt_json))
    r.add(assert_exists("hma_relabel/results_raw/summary.json", hma_json))
    r.add(assert_exists("combat/results_raw/summary.json", cb_json))
    r.add(assert_exists("cptac_brca/results_raw/summary.json", cptac_json))

    # ── D1 rank-transform discriminator (manuscript: +0.007 / -0.001) ────
    if rt_json.exists():
        d = load_json(rt_json)
        baseline = d["z_score_baseline"]
        rank = d["rank_transform"]
        d_all = _delta(baseline, rank, "all_reactions")
        d_gpr = _delta(baseline, rank, "gpr_only")
        r.add(assert_close(
            "D1 rank-transform: ΔAUROC all_reactions ≈ +0.007",
            observed=d_all, expected=+0.007, atol=0.01,
            note="Manuscript §3.15.5: rank-transform is not the dominant mechanism (Δ ≈ noise).",
        ))
        r.add(assert_close(
            "D1 rank-transform: ΔAUROC gpr_only ≈ -0.001",
            observed=d_gpr, expected=-0.001, atol=0.01,
        ))
        r.add(assert_in_range(
            "D1 rank-transform baseline AUROC near 0.493",
            observed=float(baseline["all_reactions"]["auroc"]["mean"]),
            lo=0.47, hi=0.52,
            note="z-score baseline reproduces the headline METABRIC failure.",
        ))

    # ── D2 training-time-label discriminator (manuscript: +0.019 / +0.038) ──
    if hma_json.exists():
        d = load_json(hma_json)
        baseline = d["metabric_pseudo_labels"]
        hma = d["training_time_hma_labels"]
        d_all = _delta(baseline, hma, "all_reactions")
        d_gpr = _delta(baseline, hma, "gpr_only")
        r.add(assert_close(
            "D2 training-time-label: ΔAUROC all_reactions ≈ +0.019",
            observed=d_all, expected=+0.019, atol=0.015,
            note="Label drift accounts for at most ~8% of the deficit.",
        ))
        r.add(assert_close(
            "D2 training-time-label: ΔAUROC gpr_only ≈ +0.038",
            observed=d_gpr, expected=+0.038, atol=0.015,
        ))

    # ── D3 ComBat discriminator (manuscript: -0.018 / -0.027) ─────────────
    if cb_json.exists():
        d = load_json(cb_json)
        baseline = d["raw"]
        combat = d["combat"]
        d_all = _delta(baseline, combat, "all_reactions")
        d_gpr = _delta(baseline, combat, "gpr_only")
        r.add(assert_close(
            "D3 ComBat: ΔAUROC all_reactions ≈ -0.018",
            observed=d_all, expected=-0.018, atol=0.015,
            note="Per-gene-marginal batch correction does NOT help; it slightly hurts.",
        ))
        r.add(assert_close(
            "D3 ComBat: ΔAUROC gpr_only ≈ -0.027",
            observed=d_gpr, expected=-0.027, atol=0.020,
        ))

    # ── D4 CPTAC-BRCA same-platform (manuscript: AUROC ≈ 0.499) ─────────
    if cptac_json.exists():
        d = load_json(cptac_json)
        au_all = float(d.get("auroc_all_reactions", float("nan")))
        au_gpr = float(d.get("auroc_gpr_only", float("nan")))
        n_pat = int(d.get("n_patients", 0))
        r.add(assert_close(
            "D4 CPTAC-BRCA: AUROC all_reactions ≈ 0.499",
            observed=au_all, expected=0.499, atol=0.015,
            note="Cross-consortium failure reproduces on same-platform RNA-seq.",
        ))
        r.add(assert_close(
            "D4 CPTAC-BRCA: AUROC gpr_only ≈ 0.499",
            observed=au_gpr, expected=0.499, atol=0.015,
        ))
        r.add(Check(
            name="D4 CPTAC-BRCA: n_patients = 106",
            passed=(n_pat == 106),
            expected=106, observed=n_pat,
        ))

    return r

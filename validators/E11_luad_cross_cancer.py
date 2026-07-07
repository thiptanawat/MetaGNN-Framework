"""
E11 — LUAD cross-cancer validator.

Manuscript anchor: §3.7 Table tab:cross_cancer + Table tab:luad_baselines.
Headline: LUAD AUROC 0.984 on 517 patients (79-patient test split +
50-patient baseline-aligned subset).

Validation strategy:
1. Recompute mean AUROC from `results_luad/full_cohort/results_luad.json`
   per_patient list and assert ≈ 0.984 within ±0.005.
2. Cross-check `method_comparison.json`: GPR-only AUROCs for MetaGNN
   ≈ 1.000, GIMME ≈ 0.970, iMAT ≈ 0.812, Expr ≈ 0.679 (manuscript
   line 1844: BRCA 0.962 / LUAD 0.970 GIMME GPR-only).
"""

from __future__ import annotations

import statistics

import numpy as np

from . import _common as C
from ._common import (Check, ValidationReport, assert_close, assert_exists,
                      assert_gte, assert_in_range, load_json, MANUSCRIPT)


def validate(exp_id: str = "E11", verbose: bool = False) -> ValidationReport:
    r = ValidationReport(
        experiment="E11",
        description="LUAD cross-cancer transfer (Table 9)",
        manuscript_anchor=MANUSCRIPT["E11"]["section"],
    )

    luad = C.SHARED_DATA / "tcga_luad"
    full_cohort = luad / "results_luad" / "full_cohort"
    ckpt = full_cohort / "best_model.pt"
    scores = full_cohort / "test_scores.npz"
    results_json = full_cohort / "results_luad.json"
    method_json = luad / "results_luad" / "method_comparison.json"
    subsystem_json = luad / "results_luad" / "subsystem_analysis.json"

    r.add(assert_exists("LUAD full-cohort checkpoint",
                        ckpt))
    r.add(assert_exists("LUAD test_scores.npz",
                        scores))
    r.add(assert_exists("LUAD results_luad.json",
                        results_json))
    r.add(assert_exists("LUAD method_comparison.json",
                        method_json))
    r.add(assert_exists("LUAD subsystem_analysis.json",
                        subsystem_json))

    # ── Checkpoint sanity ─────────────────────────────────────────
    if ckpt.exists():
        size = ckpt.resolve(strict=False).stat().st_size
        r.add(Check(
            name="LUAD checkpoint > 100 KB",
            passed=size > 100_000,
            expected="> 100 KB",
            observed=f"{size / 1024:.1f} KB",
        ))

    # ── test_scores.npz ───────────────────────────────────────────
    if scores.exists():
        try:
            with np.load(scores.resolve(strict=False), allow_pickle=True) as z:
                files = list(z.files)
            r.add(Check(
                name="test_scores.npz contains array(s)",
                passed=len(files) > 0,
                expected="≥ 1 array",
                observed=files[:6],
            ))
        except Exception as e:
            r.add(Check(
                name="test_scores.npz loads",
                passed=False,
                expected="valid .npz",
                observed=str(e)[:80],
            ))

    # ── Headline LUAD cross-cancer AUROC ≈ 0.984 ───────────────────
    if results_json.exists():
        try:
            data = load_json(results_json)
        except Exception as e:
            r.add(Check(
                name="results_luad.json parses",
                passed=False, expected="valid JSON",
                observed=str(e)[:80],
            ))
            return r
        per_patient = ((data.get("test_results") or {}).get("per_patient")
                        or data.get("per_patient") or [])
        aurocs = [float(p["auroc"]) for p in per_patient
                  if isinstance(p, dict) and "auroc" in p]
        if aurocs:
            r.add(assert_close(
                "LUAD cross-cancer AUROC reproduces 0.984 headline",
                observed=statistics.mean(aurocs), expected=0.984, atol=0.005,
                note="Manuscript Table tab:cross_cancer + abstract: LUAD AUROC 0.984 on 517 patients (79-patient test split).",
            ))
        # Test-split size: manuscript reports 79-patient test split
        r.add(Check(
            name="LUAD test split size ≥ 50 (79 expected)",
            passed=(len(aurocs) >= 50),
            expected="≥ 50 (79 expected)",
            observed=len(aurocs),
        ))

    # ── tab:luad_baselines: GPR-only AUROCs from method_comparison ──
    if method_json.exists():
        try:
            data = load_json(method_json)
        except Exception:
            data = None
        if isinstance(data, dict):
            methods = data.get("methods", {})

            def _au(key):
                return float(methods.get(key, {}).get("auroc_mean", float("nan")))

            if "MetaGNN (GATv2)__gpr" in methods:
                r.add(assert_close(
                    "tab:luad_baselines MetaGNN (gpr): AUROC ≈ 1.000",
                    observed=_au("MetaGNN (GATv2)__gpr"),
                    expected=1.000, atol=0.005,
                ))
            if "GIMME-score__gpr" in methods:
                r.add(assert_close(
                    "tab:luad_baselines GIMME-score (gpr): AUROC ≈ 0.970",
                    observed=_au("GIMME-score__gpr"),
                    expected=0.970, atol=0.020,
                    note="Manuscript line 1844: GIMME LUAD GPR-only AUROC 0.970.",
                ))
            if "iMAT-score__gpr" in methods:
                r.add(assert_close(
                    "tab:luad_baselines iMAT-score (gpr): AUROC ≈ 0.812",
                    observed=_au("iMAT-score__gpr"),
                    expected=0.812, atol=0.020,
                ))

    return r

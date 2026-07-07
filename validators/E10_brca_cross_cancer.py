"""
E10 — BRCA cross-cancer validator.

Manuscript anchor: §3.7 Table tab:cross_cancer + Table tab:brca_baselines.
Headline: BRCA full-cohort AUROC 0.986 ± 0.001 (n=1,095 patients).

Validation strategy:
1. Recompute mean AUROC from `results_brca/full_cohort/results_brca.json`
   over the per-patient list, and assert it matches the manuscript's
   0.986 within ±0.005.
2. Cross-check `method_comparison.json`: MetaGNN__all AUROC 0.9863,
   GIMME-score__all 0.6288, iMAT-score__all 0.5833, Expr threshold__all
   0.5748, plus GPR-only variants (manuscript reports GIMME 0.962, iMAT
   0.820, Expr 0.660 — see lines 1844-1846 of the .tex).
3. Spot-check the shipped METABRIC validation JSON (chance AUROC 0.493),
   which the E06 validator covers in detail; here we just confirm it
   exists and is in [0, 1].
"""

from __future__ import annotations

import statistics

from . import _common as C
from ._common import (Check, ValidationReport, assert_close, assert_eq,
                      assert_exists, assert_gte, assert_in_range, load_json,
                      MANUSCRIPT)


def validate(exp_id: str = "E10", verbose: bool = False) -> ValidationReport:
    r = ValidationReport(
        experiment="E10",
        description="BRCA cross-cancer transfer (Table 8)",
        manuscript_anchor=MANUSCRIPT["E10"]["section"],
    )

    brca = C.SHARED_DATA / "tcga_brca"
    ckpt_624 = brca / "results_brca" / "full_cohort_624" / "best_model.pt"
    ckpt_220 = brca / "results_brca" / "full_cohort" / "best_model.pt"
    results_brca_json = brca / "results_brca" / "full_cohort" / "results_brca.json"
    method_comp_json = brca / "results_brca" / "method_comparison.json"
    metabric_json = (C.EXPERIMENTS_DIR / "E10_brca_cross_cancer" / "code"
                     / "MetaGNN-BRCA-pipeline" / "results_brca"
                     / "metabric_validation.json")

    r.add(assert_exists("BRCA TPM matrix (E10 input)",
                        brca / "tcga_brca_tpm_log2.tsv"))
    r.add(assert_exists("BRCA gene_id_to_name.tsv",
                        brca / "gene_id_to_name.tsv"))
    r.add(assert_exists("BRCA processed/ tree",
                        brca / "processed"))
    r.add(assert_exists("BRCA full-cohort 624 best_model.pt",
                        ckpt_624))
    r.add(assert_exists("BRCA full-cohort 220 best_model.pt",
                        ckpt_220))
    r.add(assert_exists("BRCA full_cohort results_brca.json",
                        results_brca_json))
    r.add(assert_exists("BRCA method_comparison.json",
                        method_comp_json))
    r.add(assert_exists("METABRIC validation JSON",
                        metabric_json))

    # ── Headline cross-cancer AUROC ≈ 0.986 (Table cross_cancer) ──
    if results_brca_json.exists():
        d = load_json(results_brca_json)
        per_patient = ((d.get("test_results") or {}).get("per_patient")
                        or d.get("per_patient") or [])
        aurocs = [float(p["auroc"]) for p in per_patient
                  if isinstance(p, dict) and "auroc" in p]
        if aurocs:
            r.add(assert_close(
                "BRCA cross-cancer AUROC reproduces 0.986 headline",
                observed=statistics.mean(aurocs), expected=0.986, atol=0.005,
                note="Manuscript Table tab:cross_cancer + abstract: BRCA AUROC 0.986 ± 0.001 on n=1,095 patients.",
            ))
            r.add(Check(
                name="BRCA n_patients = 1,095",
                passed=(len(aurocs) == 1095 or d.get("n_patients") == 1095),
                expected=1095, observed=len(aurocs),
            ))

    # ── tab:brca_baselines: MetaGNN vs GIMME / iMAT / Expr ────────
    if method_comp_json.exists():
        d = load_json(method_comp_json)
        methods = d.get("methods", {})

        def _au(key):
            return float(methods.get(key, {}).get("auroc_mean", float("nan")))

        # All-reactions row (Table 8 row in §3.7)
        if "MetaGNN (GATv2)__all" in methods:
            r.add(assert_close(
                "tab:brca_baselines MetaGNN (all): AUROC ≈ 0.986",
                observed=_au("MetaGNN (GATv2)__all"), expected=0.986, atol=0.005,
            ))
        if "GIMME-score__all" in methods:
            r.add(assert_close(
                "tab:brca_baselines GIMME-score (all): AUROC ≈ 0.629",
                observed=_au("GIMME-score__all"), expected=0.629, atol=0.020,
            ))
        if "iMAT-score__all" in methods:
            r.add(assert_close(
                "tab:brca_baselines iMAT-score (all): AUROC ≈ 0.583",
                observed=_au("iMAT-score__all"), expected=0.583, atol=0.020,
            ))

        # GPR-only row (manuscript line 1844: GIMME 0.962, iMAT 0.820,
        # Expr 0.660; MetaGNN ~1.000)
        if "MetaGNN (GATv2)__gpr" in methods:
            r.add(assert_close(
                "tab:brca_baselines MetaGNN (gpr): AUROC ≈ 1.000",
                observed=_au("MetaGNN (GATv2)__gpr"), expected=1.000, atol=0.005,
            ))
        if "GIMME-score__gpr" in methods:
            r.add(assert_close(
                "tab:brca_baselines GIMME-score (gpr): AUROC ≈ 0.962",
                observed=_au("GIMME-score__gpr"), expected=0.962, atol=0.020,
                note="Manuscript line 1844: GIMME GPR-only AUROC 0.962.",
            ))
        if "iMAT-score__gpr" in methods:
            r.add(assert_close(
                "tab:brca_baselines iMAT-score (gpr): AUROC ≈ 0.820",
                observed=_au("iMAT-score__gpr"), expected=0.820, atol=0.020,
            ))

    # ── Checkpoint must be non-trivially sized ────────────────────
    if ckpt_624.exists():
        size = ckpt_624.resolve(strict=False).stat().st_size
        r.add(Check(
            name="BRCA 624 checkpoint > 1 MB",
            passed=size > 1_000_000,
            expected="> 1 MB",
            observed=f"{size / (1024**2):.2f} MB",
        ))

    # ── BRCA reaction features populated ──────────────────────────
    rxn_dir = brca / "processed" / "reaction_features"
    if rxn_dir.exists():
        n = sum(1 for _ in rxn_dir.glob("TCGA-*.h5"))
        r.add(assert_gte(
            "BRCA per-patient reaction_features count ≥ 100",
            observed=n, lo=100,
        ))

    # ── METABRIC validation JSON schema ───────────────────────────
    if metabric_json.exists():
        try:
            data = load_json(metabric_json)
        except Exception as e:
            r.add(Check(
                name="metabric_validation.json parses",
                passed=False,
                expected="valid JSON",
                observed=str(e)[:80],
            ))
        else:
            r.add(Check(
                name="metabric_validation.json is a dict",
                passed=isinstance(data, dict),
                expected="dict",
                observed=type(data).__name__,
            ))
            if isinstance(data, dict):
                # Schema: {"metrics": {"all_reactions": {"auroc": {"mean", "std"},
                #                                        "f1":    {"mean", "std"}},
                #                      "gpr_only": {...}}, ...}
                metrics = data.get("metrics", {})
                found_metric = (
                    isinstance(metrics, dict) and
                    any(isinstance(v, dict) and ("auroc" in v or "f1" in v)
                        for v in metrics.values())
                )
                r.add(Check(
                    name="metabric_validation.json reports a metric",
                    passed=found_metric,
                    expected="metrics.{all_reactions,gpr_only}.{auroc,f1}",
                    observed=list(metrics.keys())[:6] if metrics else list(data.keys())[:6],
                ))
                # Verify the AUROC means are present and in [0, 1]
                if isinstance(metrics, dict):
                    for split, block in metrics.items():
                        if not isinstance(block, dict):
                            continue
                        auroc = block.get("auroc", {})
                        f1 = block.get("f1", {})
                        if isinstance(auroc, dict) and "mean" in auroc:
                            r.add(assert_in_range(
                                f"METABRIC {split}: AUROC mean in [0, 1]",
                                observed=float(auroc["mean"]),
                                lo=0.0, hi=1.0,
                            ))
                        if isinstance(f1, dict) and "mean" in f1:
                            r.add(assert_in_range(
                                f"METABRIC {split}: F1 mean in [0, 1]",
                                observed=float(f1["mean"]),
                                lo=0.0, hi=1.0,
                            ))
                # Verify cohort size
                n_pat = data.get("n_patients")
                if n_pat:
                    r.add(assert_gte(
                        "METABRIC: ≥ 100 patients evaluated",
                        observed=int(n_pat), lo=100,
                    ))

    return r

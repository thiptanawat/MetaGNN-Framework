"""
E01 — Headline 220-patient CRC benchmark validator.

The shipped artefacts under ``E01_headline_220/results/`` correspond to
the **methodology validation** tables in the manuscript, not the prose
"headline" 0.806 number. Specifically:

- ``baselines/baseline_comparison_results.json`` reproduces
  Table tab:extended_baselines (line 1034 of the .tex), which compares
  MetaGNN against three baselines on standardised benchmark features
  with 3 random seeds. Manuscript values:
    MetaGNN     F1 0.9286 ± 0.0008, AUROC 0.9082 ± 0.0004
    HomoGAT     F1 0.9241 ± 0.0019, AUROC 0.9092 ± 0.0013
    HomoGCN     F1 0.8821 ± 0.0020, AUROC 0.8457 ± 0.0036
    ReactionMLP F1 0.8379 ± 0.0001, AUROC 0.7663 ± 0.0000

- ``kfold/kfold_cv_results.json`` reproduces Table tab:kfold_cv
  (line 1054), the 5-fold stratified CV on the same standardised
  benchmark. Manuscript values:
    F1 mean 0.9276 ± 0.0011, AUROC mean 0.9069 ± 0.0005

The manuscript's *prose* headline of "F1 = 0.806, AUROC = 0.866 at
epoch 141" (training-curve caption) is a different evaluation —
single-split validation set during training — and is reproduced by
``regen_fig_training_curves.py`` against the canonical training log
``cohort_220_source/results/training_logs/stage2_finetune_log.csv``.

Validation strategy:
- Load each of the two shipped JSON files.
- Recompute the headline cell values with assert_close at ±0.005,
  tight enough to catch artefact drift.
- Confirm the comparison includes all four method names.
"""

from __future__ import annotations

from . import _common as C
from ._common import (Check, ValidationReport, assert_close,
                      assert_eq, assert_exists, assert_gte, assert_in_range,
                      load_json, MANUSCRIPT)


def validate(exp_id: str = "E01", verbose: bool = False) -> ValidationReport:
    spec = MANUSCRIPT["E01"]
    r = ValidationReport(
        experiment="E01",
        description="Methodology validation (extended baselines + kfold CV)",
        manuscript_anchor=spec["section"],
    )

    exp_dir = C.EXPERIMENTS_DIR / "E01_headline_220"
    baselines_json = exp_dir / "results" / "baselines" / "baseline_comparison_results.json"
    kfold_json = exp_dir / "results" / "kfold" / "kfold_cv_results.json"

    r.add(assert_exists("baselines/baseline_comparison_results.json", baselines_json))
    r.add(assert_exists("kfold/kfold_cv_results.json", kfold_json))

    if not baselines_json.exists():
        return r

    data = load_json(baselines_json)

    # ── Models compared ────────────────────────────────────────────
    expected_models = {"MetaGNN", "ReactionMLP", "HomoGAT", "HomoGCN"}
    r.add(assert_eq(
        "manuscript baselines reported (MetaGNN + 3 baselines)",
        observed=set(data.keys()),
        expected=expected_models,
    ))

    if "MetaGNN" not in data:
        return r

    meta = data["MetaGNN"]

    # ── Tight reproduction of Table tab:extended_baselines row 1 ─────
    f1 = float(meta.get("F1_mean", float("nan")))
    auroc = float(meta.get("AUROC_mean", float("nan")))

    # Manuscript Table tab:extended_baselines (line 1044):
    #   MetaGNN F1 0.9286, AUROC 0.9082, AUPRC 0.9494, n_params 873,217
    r.add(assert_close(
        "MetaGNN F1 reproduces tab:extended_baselines row 1 (0.9286)",
        observed=f1, expected=0.9286, atol=0.005,
        note="Manuscript row: F1 0.9286 ± 0.0008.",
    ))
    r.add(assert_close(
        "MetaGNN AUROC reproduces tab:extended_baselines row 1 (0.9082)",
        observed=auroc, expected=0.9082, atol=0.005,
        note="Manuscript row: AUROC 0.9082 ± 0.0004.",
    ))

    # ── Standard deviation must be small (3-seed convergence) ─────
    f1_std = float(meta.get("F1_std", 0.0))
    r.add(assert_in_range(
        "MetaGNN F1 std small (3-seed stability)",
        observed=f1_std, lo=0.0, hi=0.005,
    ))

    # ── MetaGNN > every baseline on F1 ─────────────────────────────
    for name in ("ReactionMLP", "HomoGAT", "HomoGCN"):
        if name not in data:
            continue
        bf1 = float(data[name].get("F1_mean", float("nan")))
        r.add(Check(
            name=f"MetaGNN F1 > {name} F1",
            passed=bool(f1 > bf1),
            expected=f"> {round(bf1, 4)}",
            observed=round(f1, 4),
        ))

    # ── k-fold CV: 5 folds present ─────────────────────────────────
    if kfold_json.exists():
        kf = load_json(kfold_json)
        # Schema: {"summary": {"n_folds": ..., "F1_mean": ...}, "folds": [...]}
        per_fold = (kf.get("folds") or kf.get("per_fold")
                    or kf.get("results") or [])
        n_folds = (len(per_fold) if isinstance(per_fold, list)
                   else int(kf.get("summary", {}).get("n_folds", 0)))
        r.add(Check(
            name="k-fold CV: 5 folds reported",
            passed=n_folds == 5,
            expected=5,
            observed=n_folds,
        ))
        # Tight reproduction of Table tab:kfold_cv (line 1064-1071):
        #   F1 mean 0.9276 ± 0.0011, AUROC mean 0.9069 ± 0.0005
        summary = kf.get("summary", {}) if isinstance(kf, dict) else {}
        if "F1_mean" in summary:
            kf_f1 = float(summary["F1_mean"])
            r.add(assert_close(
                "k-fold CV: F1_mean reproduces tab:kfold_cv (0.9276)",
                observed=kf_f1, expected=0.9276, atol=0.005,
            ))
        if "AUROC_mean" in summary:
            kf_au = float(summary["AUROC_mean"])
            r.add(assert_close(
                "k-fold CV: AUROC_mean reproduces tab:kfold_cv (0.9069)",
                observed=kf_au, expected=0.9069, atol=0.005,
            ))

    return r

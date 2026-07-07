"""
Unit tests for E02_multiseed_220_hma — aggregate_multiseed.py

Covers the pure-statistics helpers used to build Tables 2/3 of the
manuscript (multi-seed mean ± SD with bootstrap 95% CI).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from _paths import EXPERIMENTS_DIR

SCRIPT = (
    EXPERIMENTS_DIR
    / "E02_multiseed_220_hma"
    / "code"
    / "code"
    / "aggregate_multiseed.py"
)


@pytest.fixture(scope="module")
def aggregate_module(module_loader):
    if not SCRIPT.exists():
        pytest.skip(f"E02 aggregate_multiseed.py not found at {SCRIPT}")
    return module_loader("e02_aggregate_multiseed", SCRIPT)


@pytest.mark.unit
def test_module_exposes_expected_api(aggregate_module):
    for fn in [
        "load_seed_summaries",
        "bootstrap_ci",
        "build_per_seed_fold",
        "build_per_seed_mean",
        "build_summary_stats",
        "write_latex",
    ]:
        assert hasattr(aggregate_module, fn), f"missing function: {fn}"


@pytest.mark.unit
def test_bootstrap_ci_shape_and_order(aggregate_module):
    rng = np.random.default_rng(0)
    vals = rng.normal(loc=0.8, scale=0.02, size=10).tolist()
    lo, hi = aggregate_module.bootstrap_ci(vals, n_bootstrap=500, seed=1)
    assert lo < hi, "bootstrap CI must be ordered (low, high)"
    # CI must contain the sample mean for any reasonable sample
    assert lo <= float(np.mean(vals)) <= hi


@pytest.mark.unit
def test_bootstrap_ci_handles_empty_and_nan(aggregate_module):
    lo, hi = aggregate_module.bootstrap_ci([], n_bootstrap=10, seed=0)
    assert np.isnan(lo) and np.isnan(hi)
    lo2, hi2 = aggregate_module.bootstrap_ci(
        [float("nan"), float("nan")], n_bootstrap=10, seed=0,
    )
    assert np.isnan(lo2) and np.isnan(hi2)


@pytest.mark.unit
def test_bootstrap_ci_deterministic_for_same_seed(aggregate_module):
    vals = [0.81, 0.79, 0.82, 0.80, 0.81]
    a = aggregate_module.bootstrap_ci(vals, n_bootstrap=500, seed=2026)
    b = aggregate_module.bootstrap_ci(vals, n_bootstrap=500, seed=2026)
    assert a == b


@pytest.mark.unit
def test_load_seed_summaries_round_trip(aggregate_module, tmp_path: Path):
    """Drop a tiny seed_*/fold_summary.json tree and verify the loader."""
    for s in (0, 1, 7):
        d = tmp_path / f"seed_{s}"
        d.mkdir()
        (d / "fold_summary.json").write_text(json.dumps({
            "per_fold": [
                {"fold": 0, "F1": 0.80 + 0.001 * s, "AUROC": 0.86,
                 "mean_sigma": 0.10, "boundary_pct": 5.0,
                 "n_boundary": 100, "n_test_patients": 33,
                 "training_time_s": 60.0},
            ],
        }))
    # Add a malformed dir that should be skipped
    (tmp_path / "seed_bad").mkdir()
    out = aggregate_module.load_seed_summaries(tmp_path)
    seeds = sorted(s for s, _ in out)
    assert seeds == [0, 1, 7]


@pytest.mark.unit
def test_build_per_seed_fold_and_mean(aggregate_module):
    seeds = [
        (0, {"per_fold": [
            {"fold": 0, "F1": 0.80, "AUROC": 0.86, "mean_sigma": 0.10,
             "boundary_pct": 5.0, "training_time_s": 60.0},
            {"fold": 1, "F1": 0.82, "AUROC": 0.87, "mean_sigma": 0.09,
             "boundary_pct": 4.5, "training_time_s": 62.0},
        ]}),
        (1, {"per_fold": [
            {"fold": 0, "F1": 0.79, "AUROC": 0.85, "mean_sigma": 0.11,
             "boundary_pct": 5.5, "training_time_s": 58.0},
        ]}),
    ]
    df_fold = aggregate_module.build_per_seed_fold(seeds)
    assert len(df_fold) == 3
    df_mean = aggregate_module.build_per_seed_mean(df_fold)
    # Row count = unique seeds, training_time_s is summed
    assert len(df_mean) == 2
    seed0 = df_mean[df_mean.seed == 0].iloc[0]
    assert seed0["F1"] == pytest.approx((0.80 + 0.82) / 2)
    assert seed0["training_time_s"] == pytest.approx(60.0 + 62.0)


@pytest.mark.unit
def test_build_summary_stats_includes_all_metrics(aggregate_module):
    df_mean = pd.DataFrame({
        "seed": [0, 1, 2],
        "F1": [0.80, 0.81, 0.79],
        "AUROC": [0.86, 0.87, 0.85],
        "mean_sigma": [0.10, 0.11, 0.09],
        "boundary_pct": [5.0, 4.8, 5.2],
    })
    summary = aggregate_module.build_summary_stats(df_mean)
    metrics = set(summary["metric"].tolist())
    assert {"F1", "AUROC", "mean_sigma", "boundary_pct"} <= metrics
    f1_row = summary[summary.metric == "F1"].iloc[0]
    assert f1_row["mean"] == pytest.approx(0.80, abs=1e-6)
    assert f1_row["n_seeds"] == 3


@pytest.mark.unit
def test_write_latex_produces_booktabs(aggregate_module, tmp_path: Path):
    summary = pd.DataFrame([
        {"metric": "F1", "n_seeds": 10, "mean": 0.806, "std": 0.014,
         "ci95_lo": 0.79, "ci95_hi": 0.82, "min": 0.78, "max": 0.83},
        {"metric": "AUROC", "n_seeds": 10, "mean": 0.866, "std": 0.010,
         "ci95_lo": 0.85, "ci95_hi": 0.88, "min": 0.84, "max": 0.88},
    ])
    out = tmp_path / "summary.tex"
    aggregate_module.write_latex(summary, out)
    txt = out.read_text()
    assert "\\begin{table}" in txt
    assert "\\toprule" in txt
    assert "\\bottomrule" in txt or "\\midrule" in txt
    assert "AUROC" in txt

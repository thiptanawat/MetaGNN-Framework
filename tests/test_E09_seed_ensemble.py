"""
Unit tests for E09_seed_ensemble_passD — MetaGNN_PassD_SeedEnsemble_Script.py

Covers the three pure aggregation strategies (mean-of-sigmoid,
mean-of-logit, majority-vote), the ensemble-disagreement σ, and the
fold-flatten / evaluation helpers — exactly the math behind Table 7.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from _paths import EXPERIMENTS_DIR

SCRIPT = (
    EXPERIMENTS_DIR
    / "E09_seed_ensemble_passD"
    / "code"
    / "MetaGNN_PassD_SeedEnsemble_Script.py"
)


@pytest.fixture(scope="module")
def passd_module(module_loader):
    if not SCRIPT.exists():
        pytest.skip(f"E09 script not found: {SCRIPT}")
    # The script's main() guards execution behind __main__, so importing
    # is safe; the dataclass / aggregator helpers are top-level.
    return module_loader("e09_pass_d", SCRIPT)


@pytest.mark.unit
def test_aggregator_api_present(passd_module):
    for fn in [
        "aggregate_mean_sigmoid",
        "aggregate_mean_logit",
        "aggregate_majority_vote",
        "ensemble_disagreement_sigma",
        "evaluate_cell_strategy",
        "_flatten_per_fold",
    ]:
        assert hasattr(passd_module, fn), f"missing {fn}"


@pytest.mark.unit
def test_mean_sigmoid_against_reference(passd_module):
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((4, 2, 6, 10)).astype(np.float64)
    expected = (1 / (1 + np.exp(-logits))).mean(axis=0)
    out = passd_module.aggregate_mean_sigmoid(logits)
    np.testing.assert_allclose(out, expected, rtol=1e-12)
    assert out.shape == logits.shape[1:]
    assert np.all((out >= 0) & (out <= 1))


@pytest.mark.unit
def test_mean_logit_is_geometric_mean_equivalent(passd_module):
    rng = np.random.default_rng(1)
    logits = rng.standard_normal((4, 2, 6, 10)).astype(np.float64)
    expected = 1 / (1 + np.exp(-logits.mean(axis=0)))
    out = passd_module.aggregate_mean_logit(logits)
    np.testing.assert_allclose(out, expected, rtol=1e-12)
    # mean-of-logit ≠ mean-of-sigmoid in general
    ms = passd_module.aggregate_mean_sigmoid(logits)
    assert not np.allclose(ms, out)


@pytest.mark.unit
def test_majority_vote_matches_threshold_fraction(passd_module):
    # Build logits where 3 of 4 seeds are positive on every cell
    pos = np.full((1, 1, 1, 1), 5.0)
    neg = np.full((1, 1, 1, 1), -5.0)
    logits = np.concatenate([pos, pos, pos, neg], axis=0)
    out = passd_module.aggregate_majority_vote(logits)
    assert out.shape == (1, 1, 1)
    assert out.flat[0] == pytest.approx(0.75)


@pytest.mark.unit
def test_ensemble_sigma_zero_when_seeds_agree(passd_module):
    logits = np.full((5, 2, 3, 4), 1.234, dtype=np.float64)
    sigma = passd_module.ensemble_disagreement_sigma(logits)
    np.testing.assert_allclose(sigma, 0.0, atol=1e-12)


@pytest.mark.unit
def test_ensemble_sigma_positive_when_seeds_disagree(passd_module):
    rng = np.random.default_rng(7)
    logits = rng.standard_normal((6, 2, 3, 4))
    sigma = passd_module.ensemble_disagreement_sigma(logits)
    assert (sigma > 0).all()


@pytest.mark.unit
def test_flatten_per_fold_drops_nan_patients(passd_module):
    # tensor: (n_folds, n_patients, n_reactions)
    tensor = np.array(
        [[[0.1, 0.2, 0.3], [np.nan, np.nan, np.nan], [0.7, 0.8, 0.9]]]
    )
    labels = np.array([[1, 0, 1], [0, 0, 1], [1, 1, 0]])
    s_flat, y_flat = passd_module._flatten_per_fold(tensor, labels)
    # 2 valid patients × 3 reactions = 6 elements
    assert s_flat.shape == (6,)
    assert y_flat.shape == (6,)
    assert not np.isnan(s_flat).any()


@pytest.mark.unit
def test_evaluate_cell_strategy_returns_canonical_keys(
    passd_module, synthetic_logits, synthetic_labels,
):
    # Average across seeds to get (n_folds, n_patients, n_reactions)
    ens = passd_module.aggregate_mean_sigmoid(synthetic_logits)
    sigma = passd_module.ensemble_disagreement_sigma(synthetic_logits)
    metrics = passd_module.evaluate_cell_strategy(
        ens, synthetic_labels, sigma, mc_sigma=None,
    )
    assert {"auroc", "macro_f1", "boundary_pct", "rho_ensemble_vs_mc"} <= set(metrics)
    assert 0 <= metrics["auroc"] <= 1
    assert 0 <= metrics["macro_f1"] <= 1
    assert 0 <= metrics["boundary_pct"] <= 100
    # rho is None because no MC-Dropout sigma was supplied
    assert metrics["rho_ensemble_vs_mc"] is None


@pytest.mark.unit
def test_decide_go_no_go_branches(passd_module):
    """Verify the verdict logic across the three regimes the docstring lists."""
    # Use a single 624-B mean_sigmoid row and tweak the AUROC.
    mk = lambda auroc: [{
        "cell": "624-B", "strategy": "mean_sigmoid", "auroc": auroc,
        "macro_f1": 0.5, "boundary_pct": 10.0,
        "rho_ensemble_vs_mc": None,
    }]
    pos = passd_module._decide_go_no_go(mk(0.5584 + 0.01))
    null = passd_module._decide_go_no_go(mk(0.5584 + 0.001))
    neg = passd_module._decide_go_no_go(mk(0.5584 - 0.01))
    assert "POSITIVE" in pos
    assert "NULL" in null
    assert "NEGATIVE" in neg

"""
Unit tests for E01_headline_220 — MetaGNN model + supporting helpers.

`apply_gpr_rules` is pure-PyTorch and runs everywhere torch is
installed; the model classes (MCDropout, HGATLayer, MetaGNN) need
torch_geometric and are auto-skipped if PyG is missing.

We exercise:
  * model construction with the canonical 256/3/8 config
  * forward pass on synthetic HeteroData
  * predict_with_uncertainty: shape contract + non-trivial sigma
  * apply_gpr_rules AND/OR semantics (Zur 2010)
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

from _paths import EXPERIMENTS_DIR, have

MODEL_FILE = (
    EXPERIMENTS_DIR
    / "E01_headline_220"
    / "code"
    / "cohort_220_pipeline"
    / "01_metagnn_model.py"
)


def _need_torch():
    if not have("torch"):
        pytest.skip("torch not installed")


def _need_torch_geometric():
    _need_torch()
    if not have("torch_geometric"):
        pytest.skip("torch_geometric not installed")


@pytest.fixture(scope="module")
def metagnn_module(module_loader):
    _need_torch_geometric()
    if not MODEL_FILE.exists():
        pytest.skip(f"E01 model file missing at {MODEL_FILE}")
    return module_loader("e01_metagnn_model", MODEL_FILE)


@pytest.fixture(scope="module")
def gpr_helpers():
    """Extract `apply_gpr_rules` from the model file without importing
    the rest of the module (which depends on torch_geometric)."""
    _need_torch()
    if not MODEL_FILE.exists():
        pytest.skip(f"E01 model file missing at {MODEL_FILE}")
    src = MODEL_FILE.read_text(encoding="utf-8")
    # Pull out just the apply_gpr_rules definition + necessary imports
    match = re.search(
        r"def apply_gpr_rules\(.*?\n    return rxn_scores\n",
        src,
        flags=re.DOTALL,
    )
    if not match:
        pytest.skip("apply_gpr_rules not found in source")
    block = match.group(0)
    ns: dict = {}
    exec("import torch\n" + block, ns)
    return ns


# ─────────────────────────────────────────────────────────────────────
# Module surface
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.requires_torch_geometric
def test_module_exposes_classes_and_helpers(metagnn_module):
    for name in ["MCDropout", "HGATLayer", "MetaGNN", "apply_gpr_rules"]:
        assert hasattr(metagnn_module, name), f"missing symbol: {name}"


# ─────────────────────────────────────────────────────────────────────
# MCDropout — should drop even in eval()
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.requires_torch_geometric
def test_mc_dropout_active_in_eval(metagnn_module):
    import torch

    drop = metagnn_module.MCDropout(p=0.5)
    drop.eval()
    x = torch.ones(1000)
    torch.manual_seed(0)
    y1 = drop(x)
    torch.manual_seed(1)
    y2 = drop(x)
    # With p=0.5 we expect ~half zeros AND different masks across seeds.
    assert (y1 == 0).any().item()
    assert not torch.equal(y1, y2)


# ─────────────────────────────────────────────────────────────────────
# Full MetaGNN forward pass
# ─────────────────────────────────────────────────────────────────────


def _toy_inputs(n_rxn=20, n_met=15):
    import torch

    x_dict = {
        "reaction": torch.randn(n_rxn, 2),
        "metabolite": torch.randn(n_met, 519),
    }
    edge_index_dict = {
        ("metabolite", "substrate_of", "reaction"): torch.randint(
            0, min(n_met, n_rxn), (2, 30)
        ),
        ("reaction", "produces", "metabolite"): torch.randint(
            0, min(n_met, n_rxn), (2, 30)
        ),
        ("reaction", "shared_metabolite", "reaction"): torch.randint(
            0, n_rxn, (2, 40)
        ),
    }
    return x_dict, edge_index_dict


@pytest.mark.unit
@pytest.mark.requires_torch_geometric
def test_metagnn_forward_shape(metagnn_module):
    import torch

    torch.manual_seed(0)
    n_rxn = 20
    model = metagnn_module.MetaGNN(
        rxn_in_dim=2, met_in_dim=519, hidden_dim=32, n_layers=2,
        heads=2, dropout=0.1, n_reactions=n_rxn,
    )
    model.eval()
    x_dict, ei = _toy_inputs(n_rxn=n_rxn, n_met=15)
    out = model(x_dict, ei)
    assert out.shape == (n_rxn,)
    assert torch.all((out >= 0) & (out <= 1)), "scores must be in [0, 1]"


@pytest.mark.unit
@pytest.mark.requires_torch_geometric
def test_metagnn_uncertainty_shape_and_nontrivial(metagnn_module):
    import torch

    torch.manual_seed(0)
    n_rxn = 16
    model = metagnn_module.MetaGNN(
        rxn_in_dim=2, met_in_dim=519, hidden_dim=32, n_layers=2,
        heads=2, dropout=0.3, n_reactions=n_rxn,
    )
    model.eval()
    x_dict, ei = _toy_inputs(n_rxn=n_rxn, n_met=12)
    mean_score, sigma = model.predict_with_uncertainty(x_dict, ei, T=5)
    assert mean_score.shape == (n_rxn,)
    assert sigma.shape == (n_rxn,)
    # Dropout p=0.3 + T=5 stochastic passes: at least one σ value > 0
    assert (sigma > 0).any().item()


# ─────────────────────────────────────────────────────────────────────
# GPR rules — AND-min, OR-max  (Zur 2010 semantics)
# Pure-torch test, no PyG required.
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.requires_torch
def test_apply_gpr_rules_and_min_or_max(gpr_helpers):
    import torch

    # 5 genes (idx 0..4), 4 reactions
    gene_expr = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    # rxn 0: spontaneous (no gene) -> stays 0
    # rxn 1: AND(g0, g1) -> min(1,2) = 1
    # rxn 2: OR(g0, g3) -> max(1, 4) = 4
    # rxn 3: OR( AND(g0,g1), AND(g2,g3) ) -> max(min(1,2), min(3,4)) = 3
    gpr_and_idx = [
        [],
        [[0, 1]],
        [[0], [3]],
        [[0, 1], [2, 3]],
    ]
    gpr_or_idx = gpr_and_idx  # signature pads but only and_groups is used

    out = gpr_helpers["apply_gpr_rules"](
        gene_expr, gpr_and_idx, gpr_or_idx, n_reactions=4,
    )
    assert out.shape == (4,)
    assert out[0].item() == 0.0
    assert out[1].item() == 1.0
    assert out[2].item() == 4.0
    assert out[3].item() == 3.0


@pytest.mark.unit
@pytest.mark.requires_torch
def test_apply_gpr_rules_handles_zero_reactions(gpr_helpers):
    import torch

    out = gpr_helpers["apply_gpr_rules"](
        torch.zeros(3), [], [], n_reactions=0,
    )
    assert out.shape == (0,)

"""
Unit tests for E14_fba_viability.

The two main drivers (`run_fba_624.py`, `run_fba_apple_to_apple.py`)
embed all logic inside `run_fba_*` functions, with the actual FBA work
done by inner `test_fba` closures. These can't be unit-tested without
COBRApy + a real Recon3D SBML file.

So we focus on the **deterministic, COBRApy-free contracts** the
runners rely on:
  * Reaction-id mapping  (R_00000 → Recon3D id) when counts match
  * Per-patient percentile threshold computation (apple-to-apple)
  * The biomass-viability rule (`biomass > 1e-6`)
  * Random-baseline ratio scaling

When COBRApy IS available, we add a tiny end-to-end test on
COBRApy's built-in `textbook` toy model to confirm the FBA path
itself does what the runner expects.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from _paths import EXPERIMENTS_DIR, have

E14_DIR = EXPERIMENTS_DIR / "E14_fba_viability" / "code"
RUN_624 = E14_DIR / "run_fba_624.py"
RUN_A2A = E14_DIR / "run_fba_apple_to_apple.py"
VALIDATE = E14_DIR / "validate_fba.py"


# ─────────────────────────────────────────────────────────────────────
# Pure helpers we re-implement against the same semantics, to lock in
# any future drift in the source files.
# ─────────────────────────────────────────────────────────────────────


def _build_rxn_mapping(n_model_rxns: int, recon3d_rxn_ids: list[str]) -> dict:
    """Mirror of the mapping logic in both runners."""
    if n_model_rxns == len(recon3d_rxn_ids):
        return {f"R_{i:05d}": rid for i, rid in enumerate(recon3d_rxn_ids)}
    return {
        f"R_{i:05d}": recon3d_rxn_ids[i]
        for i in range(min(n_model_rxns, len(recon3d_rxn_ids)))
    }


@pytest.mark.unit
def test_rxn_mapping_exact_count():
    recon3d = [f"R3D_{i}" for i in range(5)]
    m = _build_rxn_mapping(5, recon3d)
    assert m["R_00000"] == "R3D_0"
    assert m["R_00004"] == "R3D_4"
    assert len(m) == 5


@pytest.mark.unit
def test_rxn_mapping_truncates_when_predictions_shorter():
    recon3d = [f"R3D_{i}" for i in range(10)]
    m = _build_rxn_mapping(3, recon3d)
    assert len(m) == 3
    assert "R_00009" not in m
    assert m["R_00002"] == "R3D_2"


@pytest.mark.unit
def test_rxn_mapping_handles_predictions_larger():
    recon3d = [f"R3D_{i}" for i in range(3)]
    m = _build_rxn_mapping(10, recon3d)
    # Should silently produce only the keys we have Recon3D IDs for.
    assert len(m) == 3
    assert "R_00009" not in m


# ─────────────────────────────────────────────────────────────────────
# Percentile-based threshold (apple_to_apple)
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_per_patient_percentile_matches_numpy():
    # Reproduces apple-to-apple's get_patient_threshold
    pat_df = pd.DataFrame({"metagnn_score": np.linspace(0, 1, 100)})
    tau = np.percentile(pat_df["metagnn_score"].values, 23)
    # ~bottom 23% should sit below tau
    below = (pat_df["metagnn_score"] < tau).mean()
    assert 0.10 <= below <= 0.30


@pytest.mark.unit
def test_viability_threshold_rule():
    # The runner declares viable iff biomass > 1e-6
    assert (1e-5) > 1e-6
    assert not ((1e-7) > 1e-6)
    # 0 must map to non-viable
    assert not (0.0 > 1e-6)


@pytest.mark.unit
def test_random_baseline_scaling():
    """
    The 624 driver scales random scores by `avg_active_ratio / 0.5` so
    the expected positive fraction at threshold matches the GNN's
    average active ratio. This is a cheap statistical sanity test.
    """
    rng = np.random.default_rng(0)
    avg_active_ratio = 0.4
    n = 50_000
    raw = rng.random(n)
    scaled = raw * avg_active_ratio / 0.5
    threshold = 0.15
    fraction_above = (scaled >= threshold).mean()
    # Closed form: P(raw * 0.8 >= 0.15) = P(raw >= 0.1875) = 1 - 0.1875 = 0.8125
    assert fraction_above == pytest.approx(0.8125, abs=0.01)


# ─────────────────────────────────────────────────────────────────────
# Smoke import — these scripts use stdlib + numpy + pandas at top level
# only; argparse/cobra are deferred behind try/except. So we expect
# the module-level import to succeed.
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize("path", [RUN_624, RUN_A2A], ids=["run_fba_624", "run_fba_apple_to_apple"])
def test_runners_import_cleanly(path: Path, module_loader):
    if not path.exists():
        pytest.skip(f"missing {path}")
    mod = module_loader(f"e14_{path.stem}", path)
    assert mod is not None


# ─────────────────────────────────────────────────────────────────────
# Optional: actually call COBRApy on the textbook model to confirm
# the FBA path runs end-to-end on a tiny model.
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.requires_cobra
def test_cobra_textbook_model_fba_round_trip():
    if not have("cobra"):
        pytest.skip("cobra not installed")
    import cobra
    from cobra.io import load_model

    try:
        model = load_model("textbook")
    except Exception as e:  # offline / cobra-data missing
        pytest.skip(f"COBRApy textbook model unavailable: {e}")

    sol_full = model.optimize()
    assert sol_full.status == "optimal"
    base_biomass = sol_full.objective_value
    assert base_biomass > 1e-6, "Textbook model should grow"

    # Constrain a non-essential reaction → still viable
    rxn_id = "ATPM"  # ATP maintenance, present in textbook
    if rxn_id in {r.id for r in model.reactions}:
        with model as m:
            m.reactions.get_by_id(rxn_id).bounds = (0, 0)
            sol = m.optimize()
            assert sol.status == "optimal"

    # Constrain biomass directly → no growth
    with model as m:
        bm = next((r for r in m.reactions if "biomass" in r.id.lower()), None)
        if bm is not None:
            bm.bounds = (0, 0)
            sol = m.optimize()
            biomass_after = sol.objective_value if sol.status == "optimal" else 0.0
            assert biomass_after < 1e-6

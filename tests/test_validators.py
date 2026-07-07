"""
Pytest wrapper for the reviewer-facing validators.

Runs each `validators/E*.py` module's `validate()` function against
the wired `shared_data/` tree and asserts every check passes.

This is what a CI pipeline runs to gate that the shipped artefacts
still match the manuscript's headline numbers — no retraining,
deterministic, ~5 seconds for the full grid.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from _paths import EXPERIMENTS_DIR

# Ensure validators/ is importable as a package.
sys.path.insert(0, str(EXPERIMENTS_DIR))


VALIDATORS = [
    ("E01", "E01_headline_220"),
    ("E02", "E02_E03_multiseed"),
    ("E03", "E02_E03_multiseed"),
    ("E04", "E04_crosscohort"),
    ("E05", "E05_3d_feature"),
    ("E06", "E06_metabric_discriminators"),
    ("E07", "E07_msi_stratified"),
    ("E08", "E08_clustering"),
    ("E09", "E09_seed_ensemble"),
    ("E10", "E10_brca_cross_cancer"),
    ("E11", "E11_luad_cross_cancer"),
    ("E12", "E12_llm_agent_ablation"),
    ("E13", "E13_5llm_cross_backbone"),
    ("E14", "E14_fba_viability"),
]


@pytest.mark.parametrize("exp_id,module_name", VALIDATORS,
                         ids=[eid for eid, _ in VALIDATORS])
def test_validator(exp_id: str, module_name: str):
    """Each validator's `validate()` must return a passing report."""
    mod = importlib.import_module(f"validators.{module_name}")
    report = mod.validate(exp_id, verbose=False)

    if report.skipped:
        pytest.skip(report.skip_reason)

    if not report.passed:
        # Build a concise failure message listing the failed checks
        lines = [f"\n{report.experiment} — {report.description}"]
        lines.append(f"  Manuscript: {report.manuscript_anchor}")
        for c in report.checks:
            if not c.passed:
                lines.append(
                    f"  ❌ {c.name}: "
                    f"expected={c.expected!r}, observed={c.observed!r}"
                )
                if c.note:
                    lines.append(f"     {c.note}")
        pytest.fail("\n".join(lines))


def test_validation_summary_count():
    """Sanity: the validate.py CLI registers exactly the expected
    experiments. If a validator gets dropped silently, this test
    flags the mismatch."""
    from validate import EXPERIMENTS as REGISTERED
    assert {eid for eid, _ in REGISTERED} == {eid for eid, _ in VALIDATORS}, (
        "validate.py and test_validators.py disagree on which "
        "experiments have a validator"
    )

"""
Structural / organisational tests.

Ensures each experiment ships the artefacts a reviewer or a CI step
expects: a README, a code/ directory (or the documented E06-style
sub-structure), and a results/ directory.

This catches accidentally-deleted folders before they hit GitHub.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from _paths import EXPERIMENT_IDS, EXPERIMENTS_DIR  # noqa: E402


# ── Layout exceptions ──────────────────────────────────────────────────
# E04, E05, E12, E13 use hardlinked code from canonical project trees;
# their `code/` may be empty but README must still exist. This is
# intentional and documented in EXPERIMENTS_INDEX.md.
_OPTIONAL_CODE = {"E04_crosscohort_transfer", "E05_v2_3d_feature",
                  "E12_llm_agent_ablation", "E13_5llm_cross_backbone"}

# E06 has four discriminator subfolders rather than a single code/results.
_E06_SUBFOLDERS = {"cptac_brca", "combat", "hma_relabel", "rank_transform"}


@pytest.mark.smoke
@pytest.mark.parametrize("eid", EXPERIMENT_IDS)
def test_experiment_directory_exists(eid: str):
    eroot = EXPERIMENTS_DIR / eid
    assert eroot.is_dir(), f"Missing experiment directory: {eroot}"


@pytest.mark.smoke
@pytest.mark.parametrize("eid", EXPERIMENT_IDS)
def test_experiment_has_readme(eid: str):
    readme = EXPERIMENTS_DIR / eid / "README.md"
    assert readme.is_file(), f"Missing README: {readme}"
    content = readme.read_text(encoding="utf-8", errors="replace")
    assert len(content.strip()) > 50, (
        f"{readme} is too short to be a real README "
        f"(< 50 non-whitespace chars)"
    )


@pytest.mark.smoke
@pytest.mark.parametrize("eid", EXPERIMENT_IDS)
def test_experiment_has_results_or_subfolders(eid: str):
    eroot = EXPERIMENTS_DIR / eid
    if eid == "E06_metabric_discriminators":
        present = [s for s in _E06_SUBFOLDERS if (eroot / s).is_dir()]
        assert present, (
            f"E06 should expose at least one of {_E06_SUBFOLDERS} as a "
            f"sub-folder; none found under {eroot}"
        )
    else:
        results = eroot / "results"
        assert results.is_dir(), f"Missing results/ folder for {eid}: {results}"


@pytest.mark.smoke
@pytest.mark.parametrize("eid", EXPERIMENT_IDS)
def test_experiment_has_code_dir(eid: str):
    if eid == "E06_metabric_discriminators":
        pytest.skip("E06 uses sub-folder layout; covered by other test")
    code = EXPERIMENTS_DIR / eid / "code"
    assert code.is_dir(), f"Missing code/ folder for {eid}"


@pytest.mark.smoke
@pytest.mark.parametrize("eid", EXPERIMENT_IDS)
def test_experiment_code_nonempty_or_documented(eid: str):
    """Code folder must contain at least one .py/.sh OR be in the
    documented hardlink-only set."""
    if eid == "E06_metabric_discriminators":
        pytest.skip("E06 uses sub-folder layout; covered by other test")
    code = EXPERIMENTS_DIR / eid / "code"
    if not code.exists():
        pytest.skip(f"{eid} has no code/ folder")
    files = list(code.rglob("*.py")) + list(code.rglob("*.sh")) \
        + list(code.rglob("*.ipynb")) + list(code.rglob("*.R"))
    if eid in _OPTIONAL_CODE and not files:
        pytest.skip(
            f"{eid} ships an empty code/ on purpose (hardlink layout); "
            f"see EXPERIMENTS_INDEX.md"
        )
    assert files, (
        f"{eid}/code/ contains no source files (.py/.sh/.ipynb/.R). "
        f"Either add the code or list {eid} in the _OPTIONAL_CODE set."
    )


@pytest.mark.smoke
def test_experiments_index_present():
    idx = EXPERIMENTS_DIR.parent / "EXPERIMENTS_INDEX.md"
    assert idx.is_file(), (
        f"Top-level EXPERIMENTS_INDEX.md missing at {idx} — required for "
        f"reproducibility (maps E01..E14 → manuscript sections)."
    )


@pytest.mark.smoke
def test_shared_data_present():
    shared = EXPERIMENTS_DIR.parent / "shared_data"
    assert shared.is_dir(), (
        f"shared_data/ root missing at {shared}. Even if subfolders are "
        f"empty placeholders, the directory must exist for path resolution."
    )

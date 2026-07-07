"""
Path constants and discovery helpers for the experiments test-suite.

Kept as a plain module (not conftest) so test files can `import` it
directly and use the constants in `@pytest.mark.parametrize` collection
without going through fixtures (which can't drive parametrize ids).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Iterator, List

# ─────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────

TESTS_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = TESTS_DIR.parent
JOURNALS_DIR = EXPERIMENTS_DIR.parent
SHARED_DATA_DIR = JOURNALS_DIR / "shared_data"
MANUSCRIPT_DIR = JOURNALS_DIR / "Manuscripts"

EXPERIMENT_IDS: List[str] = [
    "E01_headline_220",
    "E02_multiseed_220_hma",
    "E03_multiseed_624",
    "E04_crosscohort_transfer",
    "E05_v2_3d_feature",
    "E06_metabric_discriminators",
    "E07_msi_stratified",
    "E08_clustering_pilot_passF",
    "E09_seed_ensemble_passD",
    "E10_brca_cross_cancer",
    "E11_luad_cross_cancer",
    "E12_llm_agent_ablation",
    "E13_5llm_cross_backbone",
    "E14_fba_viability",
]


# ─────────────────────────────────────────────────────────────────────
# Discovery
# ─────────────────────────────────────────────────────────────────────


def python_files_under(root: Path) -> Iterator[Path]:
    """All *.py files under root (excluding __pycache__)."""
    if not root.exists():
        return
    for p in root.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        if p.name.startswith("."):
            continue
        yield p


def discover_experiment_python_files() -> List[Path]:
    """Return one absolute Path per *.py file across all E01..E14/code/."""
    out: List[Path] = []
    for eid in EXPERIMENT_IDS:
        eroot = EXPERIMENTS_DIR / eid
        if not eroot.exists():
            continue
        code_dir = eroot / "code"
        if code_dir.exists():
            out.extend(python_files_under(code_dir))
        for sub in ("cptac_brca", "combat", "hma_relabel", "rank_transform"):
            sd = eroot / sub
            if sd.exists():
                out.extend(python_files_under(sd))
    return out


# ─────────────────────────────────────────────────────────────────────
# Optional-dependency probes (used by skip gates)
# ─────────────────────────────────────────────────────────────────────


def have(module_name: str) -> bool:
    """True iff the named module is importable."""
    return importlib.util.find_spec(module_name) is not None

"""
Shared pytest fixtures for the MetaGNN experiments test-suite.

Layout assumed:
    Journals/
        experiments/E01_*..E14_*/        -- experiment trees
        experiments/tests/               -- this folder
        shared_data/                     -- the canonical data drop
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import List

# macOS + Python 3.12 has an Objective-C runtime initializer that
# refuses to spawn subprocesses after fork once certain frameworks
# have been touched. Setting this here, *before* any test imports
# numpy/pandas/etc, keeps subprocess.run() reliable for the
# byte-compile batch test.
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

# scikit-learn / OpenMP can race with Apple's libomp under Python
# 3.12 verbose pytest output, intermittently segfaulting late in
# KMeans. Pin a single thread for the test process — KMeans on the
# tiny synthetic blobs is plenty fast single-threaded.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

# `_paths.py` is a sibling module — pytest puts the conftest dir on
# sys.path automatically.
from _paths import (  # noqa: E402
    EXPERIMENT_IDS,
    EXPERIMENTS_DIR,
    JOURNALS_DIR,
    MANUSCRIPT_DIR,
    SHARED_DATA_DIR,
    have,
)


# ─────────────────────────────────────────────────────────────────────
# Path / file fixtures
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def experiments_dir() -> Path:
    return EXPERIMENTS_DIR


@pytest.fixture(scope="session")
def journals_dir() -> Path:
    return JOURNALS_DIR


@pytest.fixture(scope="session")
def shared_data_dir() -> Path:
    return SHARED_DATA_DIR


@pytest.fixture(scope="session")
def manuscript_dir() -> Path:
    return MANUSCRIPT_DIR


@pytest.fixture(scope="session")
def experiment_ids() -> List[str]:
    return list(EXPERIMENT_IDS)


# ─────────────────────────────────────────────────────────────────────
# Optional-dependency probes (skip gates)
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def have_torch() -> bool:
    return have("torch")


@pytest.fixture(scope="session")
def have_torch_geometric() -> bool:
    return have("torch_geometric")


@pytest.fixture(scope="session")
def have_cobra() -> bool:
    return have("cobra")


@pytest.fixture(scope="session")
def have_rdkit() -> bool:
    return have("rdkit")


@pytest.fixture(scope="session")
def have_h5py() -> bool:
    return have("h5py")


# ─────────────────────────────────────────────────────────────────────
# Synthetic data fixtures (no real TCGA / Recon3D required)
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def synthetic_predictions_csv(tmp_path_factory) -> Path:
    """
    A small per-(patient, reaction) predictions CSV in the same shape
    used by E07/E14 downstream consumers. 5 patients x 20 reactions.
    """
    rng = np.random.default_rng(42)
    n_pat, n_rxn = 5, 20
    rows = []
    for p in range(n_pat):
        pid = f"TCGA-AA-{1000 + p:04d}-01A"
        for r in range(n_rxn):
            score = float(rng.uniform(0, 1))
            sigma = float(rng.uniform(0, 0.2))
            label = int(score > 0.5)
            pred = int(score > 0.5)
            rows.append(
                {
                    "patient_id": pid,
                    "reaction_id": f"R_{r:05d}",
                    "metagnn_score": score,
                    "metagnn_sigma": sigma,
                    "hma_label": label,
                    "is_active_predicted": pred,
                }
            )
    df = pd.DataFrame(rows)
    out = tmp_path_factory.mktemp("preds") / "all_folds.csv"
    df.to_csv(out, index=False)
    return out


@pytest.fixture(scope="session")
def synthetic_msi_clinical(tmp_path_factory) -> Path:
    """A tiny MSI clinical TSV with the shape E07 expects."""
    rows = []
    for p in range(5):
        rows.append(
            {
                "tcga_barcode": f"TCGA-AA-{1000 + p:04d}-01A",
                "msi_status": "MSI-H" if p % 2 == 0 else "MSS",
            }
        )
    df = pd.DataFrame(rows)
    out = tmp_path_factory.mktemp("clin") / "clinical_metadata_msi.tsv"
    df.to_csv(out, sep="\t", index=False)
    return out


@pytest.fixture(scope="session")
def synthetic_seed_results(tmp_path_factory, synthetic_predictions_csv) -> Path:
    """
    A multi-seed multi-fold result tree of the shape E07's
    msi_stratified_624_configB.py expects:
        results_root/
            seed_0/
                fold_0.csv ... fold_4.csv
            seed_1/...
    """
    base = tmp_path_factory.mktemp("results_multiseed_624_configB")
    df = pd.read_csv(synthetic_predictions_csv)
    for seed in range(3):
        seed_dir = base / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        for fold in range(2):
            df.to_csv(seed_dir / f"fold_{fold}.csv", index=False)
    return base


@pytest.fixture(scope="session")
def synthetic_logits() -> np.ndarray:
    """Logit tensor of shape (n_seeds, n_folds, n_patients, n_reactions)."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((4, 2, 6, 10)).astype(np.float64)


@pytest.fixture(scope="session")
def synthetic_labels() -> np.ndarray:
    """Binary labels of shape (n_patients, n_reactions)."""
    rng = np.random.default_rng(1)
    return (rng.random((6, 10)) > 0.5).astype(np.int64)


# ─────────────────────────────────────────────────────────────────────
# Module loader util
# ─────────────────────────────────────────────────────────────────────


def load_module_from_path(name: str, path: Path):
    """
    Load a Python file as a module. Top-level `if __name__ == '__main__'`
    blocks are NOT executed because the module is given a non-`__main__`
    name. Returns the imported module or raises whatever the underlying
    import raised.
    """
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def module_loader():
    """Fixture-form of `load_module_from_path` for test functions."""
    return load_module_from_path

"""
Unit tests for E08_clustering_pilot_passF.

Two scripts are present (an early skeleton + the canonical
MetaGNN_PassF_ClusteringPilot_Script.py). We test the canonical one
against the pure helpers it ships:

  * cluster_kmeans   — silhouette-driven k sweep
  * cluster_hierarchical (consensus, no torch needed)
  * evaluate_cell    — ARI vs MSI/CMS/project + within-cluster F1 spread
  * _decide_go_no_go — Pass-F verdict logic
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from _paths import EXPERIMENTS_DIR, have

SCRIPT = (
    EXPERIMENTS_DIR
    / "E08_clustering_pilot_passF"
    / "code"
    / "MetaGNN_PassF_ClusteringPilot_Script.py"
)


@pytest.fixture(scope="module")
def passf_helpers():
    """
    Extract the pure-numpy / pure-sklearn helpers from the script
    without importing the torch-dependent representation extractors.
    """
    if not SCRIPT.exists():
        pytest.skip(f"E08 script not found: {SCRIPT}")
    src = SCRIPT.read_text(encoding="utf-8")

    # Block extractor: pulls the def block from `def foo(` until the
    # next top-level definition starts. The script uses standard
    # 4-space indentation so we anchor on column-0 starts.
    def _extract(symbol: str) -> str:
        # Match "def symbol(" through the next "^def " or "^class " or end of file
        pat = (
            rf"^def {re.escape(symbol)}\(.*?(?=^def |^class |\Z)"
        )
        m = re.search(pat, src, flags=re.MULTILINE | re.DOTALL)
        return m.group(0) if m else ""

    blocks = {
        name: _extract(name)
        for name in (
            "cluster_kmeans",
            "cluster_hierarchical",
            "evaluate_cell",
            "_decide_go_no_go",
        )
    }
    missing = [k for k, v in blocks.items() if not v.strip()]
    if missing:
        pytest.skip(f"Could not extract: {missing}")

    ns: dict = {}
    exec(
        "from __future__ import annotations\n"
        "from typing import Dict, List, Tuple\n"
        "import numpy as np\nimport pandas as pd\n"
        "from sklearn.cluster import KMeans\n"
        "from sklearn.metrics import adjusted_rand_score, silhouette_score\n"
        "from scipy.cluster.hierarchy import fcluster, linkage\n",
        ns,
    )
    for src_block in blocks.values():
        exec(src_block, ns)
    return ns


# ─────────────────────────────────────────────────────────────────────
# cluster_kmeans
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.requires_sklearn
def test_cluster_kmeans_recovers_obvious_blobs(passf_helpers):
    if not have("sklearn"):
        pytest.skip("scikit-learn not installed")
    rng = np.random.default_rng(0)
    # Two well-separated blobs in 4D
    a = rng.normal(loc=0.0, scale=0.1, size=(30, 4))
    b = rng.normal(loc=5.0, scale=0.1, size=(30, 4))
    X = np.vstack([a, b])
    labels, k, sil = passf_helpers["cluster_kmeans"](X, (2, 5))
    assert k == 2
    assert sil > 0.5  # well-separated → high silhouette
    # The two clusters must split the input in half (modulo label perm)
    counts = np.bincount(labels)
    assert sorted(counts.tolist()) == [30, 30]


@pytest.mark.unit
@pytest.mark.requires_sklearn
def test_cluster_kmeans_respects_k_range(passf_helpers):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 3))
    _, k, _ = passf_helpers["cluster_kmeans"](X, (2, 4))
    assert 2 <= k <= 4


# ─────────────────────────────────────────────────────────────────────
# cluster_hierarchical
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.requires_sklearn
def test_cluster_hierarchical_consensus(passf_helpers):
    rng = np.random.default_rng(0)
    # 3 well-separated clusters in 5D — keep small for test speed
    centroids = np.array([[0, 0, 0, 0, 0],
                          [10, 0, 0, 0, 0],
                          [0, 10, 0, 0, 0]], dtype=float)
    points = []
    for c in centroids:
        points.append(rng.normal(loc=c, scale=0.1, size=(8, 5)))
    X = np.vstack(points)
    labels, sil, consensus = passf_helpers["cluster_hierarchical"](
        X, k=3, n_boot=4, rng=np.random.default_rng(0),
    )
    assert labels.shape == (X.shape[0],)
    assert sil > 0.5
    # Consensus matrix must be square, symmetric-ish, and bounded in [0, 1]
    assert consensus.shape == (X.shape[0], X.shape[0])
    assert (consensus >= 0).all() and (consensus <= 1).all()


# ─────────────────────────────────────────────────────────────────────
# evaluate_cell
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.requires_sklearn
def test_evaluate_cell_perfect_msi_partition(passf_helpers):
    pids = [f"p{i}" for i in range(20)]
    msi = {p: ("MSI-H" if i < 10 else "MSS") for i, p in enumerate(pids)}
    cms = {p: "NA" for p in pids}
    project = {p: "NA" for p in pids}
    per_patient_f1 = {p: 0.8 + 0.001 * i for i, p in enumerate(pids)}
    # Cluster labels exactly track MSI status
    labels = np.array([0] * 10 + [1] * 10)
    out = passf_helpers["evaluate_cell"](
        pids, labels, msi, cms, project, per_patient_f1,
    )
    assert out["n_clusters"] == 2
    assert out["ari_msi"] == pytest.approx(1.0)
    # CMS / project are all NA → ARI returns None
    assert out["ari_cms"] is None
    assert out["ari_project"] is None
    # Within-cluster F1 means must be present and bounded
    assert all(0 <= v <= 1 for v in out["within_cluster_f1"].values())


@pytest.mark.unit
@pytest.mark.requires_sklearn
def test_evaluate_cell_handles_missing_f1(passf_helpers):
    pids = [f"p{i}" for i in range(10)]
    labels = np.array([0] * 5 + [1] * 5)
    msi = {p: "MSI-H" for p in pids}
    cms = {p: "NA" for p in pids}
    proj = {p: "NA" for p in pids}
    f1 = {p: float("nan") for p in pids}  # all missing
    out = passf_helpers["evaluate_cell"](pids, labels, msi, cms, proj, f1)
    # No cluster meets the >=5 valid-F1 threshold
    assert out["within_cluster_f1"] == {} or all(
        not np.isnan(v) for v in out["within_cluster_f1"].values()
    )


# ─────────────────────────────────────────────────────────────────────
# _decide_go_no_go — verdict logic for Pass F
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_decide_go_no_go_branches(passf_helpers):
    fn = passf_helpers["_decide_go_no_go"]

    def mk(silhouettes_3x3):
        # Build a 90-row summary fitting the (rep × algo × seed) layout
        rows = []
        reps = ["a_mean_layer3", "b_score_vector", "c_sigma_vector"]
        algos = ["kmeans", "leiden", "hierarchical"]
        for ri, rep in enumerate(reps):
            for ai, algo in enumerate(algos):
                for seed in range(10):
                    rows.append({
                        "representation": rep,
                        "algorithm": algo,
                        "seed": seed,
                        "silhouette": silhouettes_3x3[ri][ai],
                    })
        return rows

    pos = fn(mk([[0.5] * 3, [0.5] * 3, [0.5] * 3]))    # 9/9 strong
    null = fn(mk([[0.0] * 3, [0.0] * 3, [0.0] * 3]))   # 0/9 strong
    inc = fn(mk([[0.5, 0.0, 0.5], [0.0, 0.5, 0.0], [0.0, 0.0, 0.0]]))
    assert "POSITIVE" in pos
    assert "NULL" in null
    assert "INCONCLUSIVE" in inc

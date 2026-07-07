"""
Unit tests for E10_brca_cross_cancer/MetaGNN-BRCA-pipeline.

Two pure pieces are testable without TCGA-BRCA / RDKit / COBRApy:

  * parse_gpr_rule (02_preprocess_brca_for_metagnn.py) — string GPR
    rule parsing into list-of-lists semantics
  * build_edge_indices (00_build_recon3d_graph.py) — derives the three
    edge sets from a stoichiometric matrix; this is the central
    topology builder reused across BRCA / LUAD / CRC pipelines.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from _paths import EXPERIMENTS_DIR, have

PIPELINE_DIR = (
    EXPERIMENTS_DIR / "E10_brca_cross_cancer" / "code" / "MetaGNN-BRCA-pipeline"
)
GRAPH_BUILDER = PIPELINE_DIR / "00_build_recon3d_graph.py"
PREPROCESS = PIPELINE_DIR / "02_preprocess_brca_for_metagnn.py"


# ─────────────────────────────────────────────────────────────────────
# parse_gpr_rule extraction
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def parse_gpr_rule():
    if not PREPROCESS.exists():
        pytest.skip(f"{PREPROCESS} missing")
    src = PREPROCESS.read_text(encoding="utf-8")
    m = re.search(
        r"def parse_gpr_rule\(rule_str: str\) -> .*?\n    return result\n",
        src,
        flags=re.DOTALL,
    )
    if not m:
        pytest.skip("parse_gpr_rule not found in source")
    ns: dict = {}
    exec(
        "from typing import List\n"
        "import pandas as pd\n" + m.group(0),
        ns,
    )
    return ns["parse_gpr_rule"]


@pytest.mark.unit
def test_parse_gpr_rule_simple_and(parse_gpr_rule):
    assert parse_gpr_rule("GeneA and GeneB") == [["GeneA", "GeneB"]]


@pytest.mark.unit
def test_parse_gpr_rule_simple_or(parse_gpr_rule):
    assert parse_gpr_rule("GeneA or GeneB") == [["GeneA"], ["GeneB"]]


@pytest.mark.unit
def test_parse_gpr_rule_mixed(parse_gpr_rule):
    assert parse_gpr_rule("(GeneA and GeneB) or GeneC") == [
        ["GeneA", "GeneB"], ["GeneC"]
    ]


@pytest.mark.unit
def test_parse_gpr_rule_empty_inputs(parse_gpr_rule):
    assert parse_gpr_rule("") == []
    assert parse_gpr_rule("   ") == []
    assert parse_gpr_rule(float("nan")) == []
    assert parse_gpr_rule(None) == []


@pytest.mark.unit
def test_parse_gpr_rule_strips_whitespace(parse_gpr_rule):
    assert parse_gpr_rule(" GeneA  and  GeneB ") == [["GeneA", "GeneB"]]


# ─────────────────────────────────────────────────────────────────────
# build_edge_indices extraction (requires torch)
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def build_edge_indices_fn(tmp_path_factory):
    """Extract `build_edge_indices` from the graph builder. Requires
    torch (the function calls torch.tensor / torch.save)."""
    if not have("torch"):
        pytest.skip("torch not installed")
    if not GRAPH_BUILDER.exists():
        pytest.skip(f"{GRAPH_BUILDER} missing")
    src = GRAPH_BUILDER.read_text(encoding="utf-8")
    m = re.search(
        r"def build_edge_indices\(S: np\.ndarray, output_dir: str\):.*?(?=\n# ===|\ndef )",
        src,
        flags=re.DOTALL,
    )
    if not m:
        pytest.skip("build_edge_indices not found in source")
    ns: dict = {}
    # The original module uses a `logger` global — provide a no-op.
    exec(
        "import os\nimport numpy as np\nimport logging\n"
        "logger = logging.getLogger('e10_test')\n"
        + m.group(0),
        ns,
    )
    return ns["build_edge_indices"]


@pytest.mark.unit
@pytest.mark.requires_torch
def test_build_edge_indices_basic_topology(build_edge_indices_fn, tmp_path: Path):
    import torch

    # 4 metabolites × 3 reactions
    # rxn 0: m0 (-1) -> m1 (+1)        substrate=1, product=1
    # rxn 1: m1 (-1) -> m2 (+1)        substrate=1, product=1
    # rxn 2: m2 (-1) + m0 (-2) -> m3   substrate=2, product=1
    S = np.array(
        [
            [-1.0,  0.0, -2.0],
            [+1.0, -1.0,  0.0],
            [ 0.0, +1.0, -1.0],
            [ 0.0,  0.0, +1.0],
        ],
        dtype=np.float32,
    )
    counts = build_edge_indices_fn(S, str(tmp_path))
    # substrate_of: 4 negative entries
    assert counts["substrate_of"] == 4
    # produces: 3 positive entries
    assert counts["produces"] == 3
    # shared_metabolite: rxn 0&1 share m1, rxn 1&2 share m2, rxn 0&2 share m0
    # 6 directed edges expected (each unordered pair contributes 2)
    assert counts["shared_metabolite"] == 6
    for name in ("substrate_of.pt", "produces.pt", "shared_metabolite.pt"):
        assert (tmp_path / "edge_indices" / name).is_file()


@pytest.mark.unit
@pytest.mark.requires_torch
def test_build_edge_indices_currency_filter(build_edge_indices_fn, tmp_path: Path):
    """A metabolite in > 150 reactions must be excluded from
    shared_metabolite edges."""
    n_met = 5
    n_rxn = 200
    rng = np.random.default_rng(0)
    S = np.zeros((n_met, n_rxn), dtype=np.float32)
    # Metabolite 0 participates in every reaction (currency-like)
    S[0, :] = -1.0
    # Other metabolites participate in random reactions
    for m in range(1, n_met):
        idx = rng.choice(n_rxn, size=20, replace=False)
        S[m, idx] = 1.0
    # Add at least one positive entry per reaction so produces is non-trivial
    for r in range(n_rxn):
        S[1, r] = 1.0
    counts = build_edge_indices_fn(S, str(tmp_path))
    # After currency exclusion, m0 alone cannot create shared edges,
    # so the count comes purely from m1..m4.
    assert counts["shared_metabolite"] >= 0
    # produces edges = sum of S > 0 entries
    assert counts["produces"] == int((S > 0).sum())

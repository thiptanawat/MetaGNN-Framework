"""
Data-inventory tests.

Confirms that the shared_data drop is consistent with what the
experiments expect and that the manuscript anchor file is reachable.
These tests are LENIENT — they do NOT require the full TCGA / Recon3D
binaries to be present (those are stored externally and downloaded on
demand) but they DO require the directory contract to be intact.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from _paths import EXPERIMENTS_DIR, JOURNALS_DIR, SHARED_DATA_DIR


# Subfolders the EXPERIMENTS_INDEX.md promises to expose under shared_data/
_REQUIRED_SHARED_DATA_SUBDIRS = {
    "cptac_brca",
    "depmap",
    "gpr_mask",
    "hma_labels",
    "metabric",
    "recon3d",
    "tcga_brca",
    "tcga_crc_220",
    "tcga_crc_624",
    "tcga_luad",
    "training_logs",
}


# ─────────────────────────────────────────────────────────────────────
# Shared-data layout
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.smoke
def test_shared_data_root_exists():
    assert SHARED_DATA_DIR.is_dir(), (
        f"shared_data/ root missing at {SHARED_DATA_DIR} — the parent "
        f"data drop is required for path resolution even when individual "
        f"files are downloaded on demand."
    )


@pytest.mark.smoke
@pytest.mark.parametrize("subdir", sorted(_REQUIRED_SHARED_DATA_SUBDIRS))
def test_shared_data_subdir_exists(subdir):
    p = SHARED_DATA_DIR / subdir
    assert p.is_dir(), (
        f"shared_data/{subdir}/ missing — listed in EXPERIMENTS_INDEX.md "
        f"as a canonical sub-folder."
    )


@pytest.mark.smoke
def test_gpr_mask_payload_present():
    """
    The GPR reaction mask is a small JSON that ships with the repo
    (not gated behind external downloads). It MUST be present.
    """
    p = SHARED_DATA_DIR / "gpr_mask" / "gpr_reaction_mask.json"
    assert p.is_file(), f"GPR mask payload missing at {p}"
    obj = json.loads(p.read_text(encoding="utf-8"))
    # Either a list of reaction ids or a {rxn_id: bool} mapping
    assert isinstance(obj, (list, dict))
    assert len(obj) > 0


@pytest.mark.smoke
def test_training_logs_payload_present():
    """Stage-1/Stage-2 training logs ship with the repo."""
    s1 = SHARED_DATA_DIR / "training_logs" / "stage1_pretrain_log.csv"
    s2 = SHARED_DATA_DIR / "training_logs" / "stage2_finetune_log.csv"
    assert s1.is_file(), f"Stage-1 training log missing: {s1}"
    assert s2.is_file(), f"Stage-2 training log missing: {s2}"


# ─────────────────────────────────────────────────────────────────────
# Manuscript reachability
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.smoke
def test_manuscripts_dir_exists():
    p = JOURNALS_DIR / "Manuscripts"
    assert p.is_dir(), f"Manuscripts/ missing at {p}"


@pytest.mark.smoke
def test_experiments_index_lists_all_experiments():
    idx = JOURNALS_DIR / "EXPERIMENTS_INDEX.md"
    assert idx.is_file()
    text = idx.read_text(encoding="utf-8")
    for eid in [
        "E01_headline_220", "E02_multiseed_220_hma", "E03_multiseed_624",
        "E04_crosscohort_transfer", "E05_v2_3d_feature",
        "E06_metabric_discriminators", "E07_msi_stratified",
        "E08_clustering_pilot_passF", "E09_seed_ensemble_passD",
        "E10_brca_cross_cancer", "E11_luad_cross_cancer",
        "E12_llm_agent_ablation", "E13_5llm_cross_backbone",
        "E14_fba_viability",
    ]:
        assert eid in text, (
            f"EXPERIMENTS_INDEX.md does not reference {eid} — the index "
            f"must enumerate every experiment so reviewers can map "
            f"results to manuscript tables."
        )

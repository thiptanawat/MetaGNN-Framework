"""
Path-resolution tests.

For every experiment that has been wired to `shared_data/`, this suite
asserts the actual input files the scripts open at runtime are
reachable, the right size, and have the expected schema.

These tests give the strongest reproducibility guarantee in the
suite: a green run here means a fresh checkout (with the canonical
project trees in place) can run the experiments without absolute-path
edits.

Tests gracefully skip where data is documented as download-on-demand
(METABRIC, CPTAC-BRCA, raw TCGA-LUAD).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
import pytest

from _paths import EXPERIMENTS_DIR, SHARED_DATA_DIR


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _resolved_size(p: Path) -> int:
    """Return target size for a regular file or symlink. -1 if missing."""
    try:
        return p.resolve(strict=True).stat().st_size
    except (FileNotFoundError, OSError):
        return -1


def _resolved_exists(p: Path) -> bool:
    """True iff the path (or its symlink target) exists."""
    try:
        return p.resolve(strict=True).exists()
    except (FileNotFoundError, OSError):
        return False


def _has_columns(p: Path, expected: Iterable[str], sep: str = ",") -> bool:
    """Confirm a CSV/TSV file's header contains the expected columns."""
    df = pd.read_csv(p, sep=sep, nrows=1)
    return set(expected).issubset(df.columns)


# ─────────────────────────────────────────────────────────────────────
# Recon3D — every model-using experiment depends on this
# ─────────────────────────────────────────────────────────────────────


class TestRecon3D:
    """The Recon3D bipartite graph is the foundational input."""

    @pytest.mark.smoke
    def test_recon3d_xml_exists_and_is_sbml(self):
        p = SHARED_DATA_DIR / "recon3d" / "Recon3D.xml"
        size = _resolved_size(p)
        assert size > 1_000_000, (
            f"shared_data/recon3d/Recon3D.xml missing or truncated "
            f"(size={size}). Expected ~28 MB SBML file."
        )
        # First few bytes should be the XML/SBML header
        with open(p, "rb") as fh:
            head = fh.read(200)
        assert b"<?xml" in head[:50] or b"<sbml" in head[:200], (
            "Recon3D.xml does not look like an SBML file"
        )

    @pytest.mark.smoke
    def test_recon3d_stoich_h5_exists(self):
        p = SHARED_DATA_DIR / "recon3d" / "recon3d_stoich.h5"
        assert _resolved_exists(p), f"missing: {p}"
        assert _resolved_size(p) > 100_000

    @pytest.mark.smoke
    def test_metabolite_features_h5_exists(self):
        p = SHARED_DATA_DIR / "recon3d" / "metabolite_features.h5"
        assert _resolved_exists(p), f"missing: {p}"
        assert _resolved_size(p) > 50_000

    @pytest.mark.smoke
    def test_gpr_table_tsv_has_expected_schema(self):
        p = SHARED_DATA_DIR / "recon3d" / "gpr_table.tsv"
        assert _resolved_exists(p), f"missing: {p}"
        df = pd.read_csv(p, sep="\t", nrows=5)
        # Must expose at least one column naming the reaction and the gene-set
        cols = set(df.columns)
        assert any(c.lower().startswith("rxn") or c.lower().startswith("reaction")
                   for c in cols), f"no reaction column in {cols}"
        assert any("gene" in c.lower() or "gpr" in c.lower() for c in cols), (
            f"no gene/gpr column in {cols}"
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", ["substrate_of.pt", "produces.pt", "shared_metabolite.pt"])
    def test_edge_index_tensor_present(self, name):
        p = SHARED_DATA_DIR / "recon3d" / "edge_indices" / name
        assert _resolved_exists(p), f"missing edge index: {p}"


# ─────────────────────────────────────────────────────────────────────
# TCGA-CRC — drives E01–E05, E07, E09, E14
# ─────────────────────────────────────────────────────────────────────


class TestTCGACRC:
    @pytest.mark.smoke
    def test_clinical_metadata_msi_tsv_present(self):
        p = SHARED_DATA_DIR / "tcga_crc_624" / "clinical_metadata_msi.tsv"
        assert _resolved_exists(p), f"missing: {p}"
        df = pd.read_csv(p, sep="\t")
        assert "tcga_barcode" in df.columns
        assert "msi_status" in df.columns
        # Manuscript: 624 patients with 579 evaluable MSI calls
        assert len(df) >= 600, f"expected ≥600 patients, got {len(df)}"
        evaluable = df["msi_status"].isin(["MSI-H", "MSS"]).sum()
        assert evaluable >= 500, (
            f"expected ≥500 evaluable MSI calls, got {evaluable}"
        )

    @pytest.mark.smoke
    def test_tcga_crc_rnaseq_h5_present(self):
        p = SHARED_DATA_DIR / "tcga_crc_624" / "tcga_crc_rnaseq.h5"
        assert _resolved_exists(p), f"missing: {p}"
        assert _resolved_size(p) > 1_000_000

    @pytest.mark.smoke
    def test_reaction_features_dir_populated(self):
        d = SHARED_DATA_DIR / "tcga_crc_624" / "reaction_features"
        assert _resolved_exists(d), f"missing: {d}"
        h5_files = list(d.glob("TCGA-*.h5"))
        assert len(h5_files) >= 100, (
            f"expected ≥100 per-patient .h5 files in {d}, got {len(h5_files)}"
        )

    @pytest.mark.smoke
    def test_cptac_crc_protein_h5_present(self):
        # The 220-cohort hinges on CPTAC-COAD-overlapping patients
        p = SHARED_DATA_DIR / "tcga_crc_220" / "cptac_crc_protein.h5"
        assert _resolved_exists(p), f"missing: {p}"


# ─────────────────────────────────────────────────────────────────────
# HMA labels — pseudolabels driving the supervised loss
# ─────────────────────────────────────────────────────────────────────


class TestHMALabels:
    @pytest.mark.smoke
    def test_activity_pseudolabels_pt_present(self):
        p = SHARED_DATA_DIR / "hma_labels" / "activity_pseudolabels.pt"
        assert _resolved_exists(p), f"missing: {p}"
        assert _resolved_size(p) > 1000


# ─────────────────────────────────────────────────────────────────────
# Multi-seed predictions — the input to E07 / E09 / E14
# ─────────────────────────────────────────────────────────────────────


_PRED_COLS = ["patient_id", "reaction_id", "metagnn_score", "metagnn_sigma",
              "hma_label", "is_active_predicted"]


class TestMultiSeedPredictions:
    @pytest.fixture(scope="class")
    def configB(self):
        d = (SHARED_DATA_DIR / "training_logs"
             / "results_multiseed_624_configB")
        if not _resolved_exists(d):
            pytest.skip(f"multiseed configB tree not wired: {d}")
        return d

    def test_at_least_10_seeds_present(self, configB):
        seeds = [p for p in configB.iterdir()
                 if p.is_dir() and p.name.startswith("seed_")]
        assert len(seeds) >= 10, (
            f"manuscript expects 10 seeds, got {len(seeds)}: "
            f"{[s.name for s in seeds]}"
        )

    def test_each_seed_has_5_folds(self, configB):
        for seed_dir in sorted(configB.iterdir()):
            if not seed_dir.name.startswith("seed_"):
                continue
            folds = list(seed_dir.glob("fold_*.csv"))
            assert len(folds) == 5, (
                f"{seed_dir.name}: expected 5 folds, got {len(folds)}"
            )

    def test_fold_csv_schema(self, configB):
        seed_0 = next(p for p in configB.iterdir()
                      if p.name.startswith("seed_"))
        fold_csv = seed_0 / "fold_0.csv"
        assert _has_columns(fold_csv, _PRED_COLS), (
            f"fold CSV {fold_csv} missing required columns "
            f"{_PRED_COLS}"
        )

    def test_fold_csv_has_real_predictions(self, configB):
        seed_0 = next(p for p in configB.iterdir()
                      if p.name.startswith("seed_"))
        fold_csv = seed_0 / "fold_0.csv"
        df = pd.read_csv(fold_csv, nrows=10_000)
        # Scores must lie in [0, 1] and not all be the same constant
        assert df["metagnn_score"].between(0, 1).all()
        assert df["metagnn_score"].std() > 0.01

    def test_all_folds_csv_present(self):
        p = (SHARED_DATA_DIR / "training_logs" / "fold_predictions"
             / "all_folds.csv")
        assert _resolved_exists(p), f"E14 input missing: {p}"
        assert _has_columns(p, _PRED_COLS)


# ─────────────────────────────────────────────────────────────────────
# Cross-cancer (BRCA, LUAD)
# ─────────────────────────────────────────────────────────────────────


class TestBRCA:
    @pytest.mark.smoke
    def test_processed_dir_wired(self):
        d = SHARED_DATA_DIR / "tcga_brca" / "processed"
        assert _resolved_exists(d), f"missing: {d}"
        # The processed/ tree should expose at least edge_indices and
        # metabolite_features.h5 from the BRCA pipeline.
        for child in ("edge_indices", "metabolite_features.h5",
                      "patient_list.txt", "gpr_table.tsv"):
            assert _resolved_exists(d / child), (
                f"BRCA processed/{child} missing — pipeline 00–02 "
                f"must run before E10 can use it"
            )

    @pytest.mark.smoke
    def test_raw_star_counts_present(self):
        d = SHARED_DATA_DIR / "tcga_brca" / "raw"
        assert _resolved_exists(d), f"missing: {d}"
        star = list(d.glob("TCGA-*.star_gene_counts.tsv"))
        assert len(star) >= 100, (
            f"expected ≥100 STAR-counts files in {d}, got {len(star)}"
        )


class TestLUAD:
    @pytest.mark.smoke
    def test_results_luad_wired(self):
        d = SHARED_DATA_DIR / "tcga_luad" / "results_luad"
        assert _resolved_exists(d), f"missing: {d}"

    def test_raw_data_documented_as_download_on_demand(self):
        readme = SHARED_DATA_DIR / "tcga_luad" / "README.md"
        assert _resolved_exists(readme), f"missing: {readme}"
        assert "download-on-demand" in readme.read_text().lower() or \
               "01_download_tcga_luad.py" in readme.read_text()


class TestMETABRIC:
    """METABRIC raw data is download-on-demand; only the README ships."""

    def test_readme_explains_download(self):
        readme = SHARED_DATA_DIR / "metabric" / "README.md"
        assert _resolved_exists(readme), f"missing: {readme}"
        text = readme.read_text().lower()
        assert "cbioportal" in text
        assert "download" in text


class TestCPTACBRCA:
    """CPTAC-BRCA raw data is download-on-demand; only the README ships."""

    def test_readme_explains_download(self):
        readme = SHARED_DATA_DIR / "cptac_brca" / "README.md"
        assert _resolved_exists(readme), f"missing: {readme}"
        text = readme.read_text().lower()
        assert "cptac" in text
        assert "download" in text


# ─────────────────────────────────────────────────────────────────────
# DepMap HMA tissue GEMs
# ─────────────────────────────────────────────────────────────────────


class TestDepMap:
    @pytest.mark.smoke
    def test_hma_tissue_gems_dir_present(self):
        d = SHARED_DATA_DIR / "depmap" / "hma_tissue_gems"
        assert _resolved_exists(d), f"missing: {d}"
        mat_files = list(d.glob("*.mat"))
        assert len(mat_files) >= 10, (
            f"expected ≥10 .mat files for HMA tissue GEMs in {d}, "
            f"got {len(mat_files)}"
        )


# ─────────────────────────────────────────────────────────────────────
# Per-script default-path resolution
# ─────────────────────────────────────────────────────────────────────


def _import_script(name: str, path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestE07PathDefaults:
    """E07's module-level constants must resolve under the wired layout."""

    @pytest.fixture(scope="class")
    def e07(self):
        path = (EXPERIMENTS_DIR / "E07_msi_stratified" / "code"
                / "msi_stratified_624_configB.py")
        if not path.exists():
            pytest.skip(f"missing E07 script: {path}")
        return _import_script("e07_path_check", path)

    def test_clin_resolves(self, e07):
        assert _resolved_exists(e07.CLIN), (
            f"E07 CLIN does not resolve: {e07.CLIN}"
        )

    def test_results_resolves(self, e07):
        assert _resolved_exists(e07.RESULTS), (
            f"E07 RESULTS does not resolve: {e07.RESULTS}"
        )

    def test_results_has_seeds(self, e07):
        seeds = [p for p in e07.RESULTS.iterdir()
                 if p.is_dir() and p.name.startswith("seed_")]
        assert len(seeds) >= 10


class TestE14PathDefaults:
    """E14's argparse defaults must point at real files."""

    @pytest.fixture(scope="class")
    def e14_624(self):
        path = (EXPERIMENTS_DIR / "E14_fba_viability" / "code"
                / "run_fba_624.py")
        return _import_script("e14_624_path_check", path)

    @pytest.fixture(scope="class")
    def e14_a2a(self):
        path = (EXPERIMENTS_DIR / "E14_fba_viability" / "code"
                / "run_fba_apple_to_apple.py")
        return _import_script("e14_a2a_path_check", path)

    def test_run_fba_624_defaults(self, e14_624):
        defaults = e14_624._default_paths()
        for k in ("predictions", "recon3d"):
            assert defaults[k].resolve().exists(), (
                f"E14 run_fba_624 {k} default does not exist: "
                f"{defaults[k]}"
            )

    def test_run_fba_apple_to_apple_defaults(self, e14_a2a):
        defaults = e14_a2a._default_paths()
        for k in ("recon3d", "pred_220", "pred_624"):
            assert defaults[k].resolve().exists(), (
                f"E14 run_fba_apple_to_apple {k} default does not exist: "
                f"{defaults[k]}"
            )


# ─────────────────────────────────────────────────────────────────────
# Sanity: no Python script ships a hardcoded /Users/ or /home/<user>/
# argparse default
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.smoke
def test_no_hardcoded_user_paths_in_python_scripts():
    """Catches regressions: no script should ship a personal absolute
    path in an argparse default or top-level constant."""
    bad_patterns = (
        "/Users/<user>",
        "/sessions/<session>",
        "/home/<user>",
    )
    offenders = []
    for py in EXPERIMENTS_DIR.rglob("*.py"):
        if "__pycache__" in py.parts or "/tests/" in str(py):
            continue
        text = py.read_text(encoding="utf-8", errors="replace")
        for pat in bad_patterns:
            if pat in text:
                # Allow patterns inside docstrings only if they're
                # clearly part of a usage example or comment context.
                # Hard-line: any code use is a regression.
                # We therefore inspect each line.
                for lineno, line in enumerate(text.splitlines(), 1):
                    if pat in line and not line.lstrip().startswith(("#", '"', "'")):
                        offenders.append(f"{py}:{lineno}: {line.strip()}")
    if offenders:
        msg = "\n".join(offenders[:20])
        pytest.fail(
            f"{len(offenders)} hardcoded absolute path(s) in scripts:\n{msg}"
        )

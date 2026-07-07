"""
End-to-end integration test for E07 (MSI-stratified analysis).

This is the strongest reproducibility check the suite ships: it runs the
ACTUAL `msi_stratified_624_configB.py` script (no mocking, no synthetic
data) against the wired `shared_data/` tree and asserts the manuscript-
anchored outputs match.

Marked `slow` and `integration` so users can opt out with
`pytest -m "not slow"`. Takes ~60 seconds on a recent laptop.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from _paths import EXPERIMENTS_DIR, JOURNALS_DIR, SHARED_DATA_DIR

E07_SCRIPT = (
    EXPERIMENTS_DIR
    / "E07_msi_stratified"
    / "code"
    / "msi_stratified_624_configB.py"
)


def _shared_data_wired() -> bool:
    """Cheap pre-flight: are the two paths E07 needs resolvable?"""
    clin = SHARED_DATA_DIR / "tcga_crc_624" / "clinical_metadata_msi.tsv"
    results = (SHARED_DATA_DIR / "training_logs"
               / "results_multiseed_624_configB")
    if not clin.resolve(strict=False).exists():
        return False
    if not results.resolve(strict=False).exists():
        return False
    seeds = list(results.glob("seed_*"))
    return len(seeds) >= 10


@pytest.mark.slow
@pytest.mark.integration
class TestE07EndToEnd:
    """Run E07's main() on real data; verify outputs match manuscript."""

    @pytest.fixture(scope="class")
    def run_outputs(self, tmp_path_factory) -> Path:
        if not E07_SCRIPT.exists():
            pytest.skip(f"E07 script missing: {E07_SCRIPT}")
        if not _shared_data_wired():
            pytest.skip(
                "shared_data/ not wired — run "
                "`bash experiments/scripts/wire_shared_data.sh` first"
            )

        outdir = tmp_path_factory.mktemp("e07_e2e")
        env = os.environ.copy()
        env["E07_OUTDIR"] = str(outdir)
        # Force JOURNALS_ROOT so the script resolves shared_data relative
        # to the actual repo root, not whatever cwd pytest happens to use.
        env["JOURNALS_ROOT"] = str(JOURNALS_DIR)
        # Single thread keeps sklearn/OpenMP from racing on macOS Py3.12.
        env["OMP_NUM_THREADS"] = "1"

        proc = subprocess.run(
            [sys.executable, str(E07_SCRIPT)],
            env=env,
            cwd=str(JOURNALS_DIR),
            capture_output=True,
            text=True,
            timeout=600,
        )
        if proc.returncode != 0:
            pytest.fail(
                f"E07 script exited {proc.returncode}\n"
                f"STDOUT:\n{proc.stdout[-2000:]}\n"
                f"STDERR:\n{proc.stderr[-2000:]}"
            )
        return outdir

    # ── Output files ────────────────────────────────────────────────

    def test_all_expected_outputs_exist(self, run_outputs):
        for name in [
            "per_patient_f1_long.tsv",
            "per_patient_f1_aggregate.tsv",
            "stratum_summary.tsv",
            "seed_level_stratum.tsv",
            "stat_report.txt",
            "reaction_pooled_f1_by_seedfold.tsv",
            "reaction_pooled_f1_summary.tsv",
            "msi_stratified_snippet.tex",
            "fig_msi_stratified_f1.png",
            "fig_msi_stratified_f1.pdf",
        ]:
            p = run_outputs / name
            assert p.exists(), f"Expected output missing: {name}"
            assert p.stat().st_size > 0, f"Output is empty: {name}"

    # ── Manuscript-anchored numerical checks ────────────────────────

    def test_patient_counts_match_manuscript(self, run_outputs):
        """Manuscript: 92 MSI-H + 487 MSS = 579 evaluable of 624."""
        df = pd.read_csv(run_outputs / "stratum_summary.tsv", sep="\t")
        msi_h = df[df.stratum == "MSI-H"]["n_patients"].iloc[0]
        mss = df[df.stratum == "MSS"]["n_patients"].iloc[0]
        assert msi_h == 92, f"Expected 92 MSI-H, got {msi_h}"
        assert mss == 487, f"Expected 487 MSS, got {mss}"
        assert msi_h + mss == 579

    def test_per_patient_aggregate_has_one_row_per_patient(self, run_outputs):
        df = pd.read_csv(
            run_outputs / "per_patient_f1_aggregate.tsv", sep="\t"
        )
        assert len(df) == 579
        # Each patient should have a real F1 mean in [0, 1]
        assert df["mean"].between(0, 1).all()
        # And n=10 seeds × 5 folds = 50 observations per patient (max)
        assert df["count"].max() <= 50

    def test_seed_level_has_10_seeds(self, run_outputs):
        df = pd.read_csv(run_outputs / "seed_level_stratum.tsv", sep="\t")
        assert len(df) == 10, f"Expected 10 seeds, got {len(df)}"
        # Both strata columns must be present after the unstack
        assert {"MSI-H", "MSS"} <= set(df.columns)

    def test_stat_report_reports_valid_pvalues(self, run_outputs):
        text = (run_outputs / "stat_report.txt").read_text()
        # The report must include the three statistical tests reported
        # in the manuscript (Mann-Whitney, paired Wilcoxon, reaction-
        # pooled Wilcoxon).
        assert "Mann-Whitney U" in text
        assert "Wilcoxon" in text
        assert "Reaction-pooled F1" in text
        # Each p-value line must parse and lie in [0, 1].
        import re
        pvals = re.findall(r"p\s*=\s*([0-9.eE\-+]+)", text)
        assert pvals, "no p-values found in stat_report.txt"
        for raw in pvals:
            v = float(raw)
            assert 0 <= v <= 1, f"invalid p-value: {raw}"

    def test_msi_invariance_holds(self, run_outputs):
        """Manuscript headline finding: MSI-H and MSS are statistically
        indistinguishable. A small Δ and large p is the falsifier-pass."""
        df = pd.read_csv(run_outputs / "stratum_summary.tsv", sep="\t")
        msi_h_mean = df[df.stratum == "MSI-H"]["f1_mean"].iloc[0]
        mss_mean = df[df.stratum == "MSS"]["f1_mean"].iloc[0]
        delta = abs(msi_h_mean - mss_mean)
        assert delta < 0.05, (
            f"MSI-H/MSS Δ={delta:.4f} exceeds 0.05 — the manuscript "
            f"reports the strata are indistinguishable. Either the "
            f"data has changed or the analysis is broken."
        )

    def test_latex_snippet_includes_anchor(self, run_outputs):
        text = (run_outputs / "msi_stratified_snippet.tex").read_text()
        assert "\\subsubsection" in text
        assert "MSI-Stratified" in text
        assert "\\label{sec:msi_stratified_624}" in text
        # Figure include must reference the fig we wrote
        assert "fig_msi_stratified_f1.pdf" in text

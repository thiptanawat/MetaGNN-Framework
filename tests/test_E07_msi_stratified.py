"""
Unit tests for E07_msi_stratified — msi_stratified_624_configB.py

The script has hardcoded absolute paths that won't exist outside the
H100 environment, so we **cannot** import the script as-is (it calls
`OUTDIR.mkdir(...)` at module top-level, which writes outside tmp).

Strategy: read the source, extract & re-define the pure helper
functions (`tcga_patient_barcode`, `per_patient_f1`) using exec on the
helper-block so we test the actual code in the file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from _paths import EXPERIMENTS_DIR

SCRIPT = (
    EXPERIMENTS_DIR
    / "E07_msi_stratified"
    / "code"
    / "msi_stratified_624_configB.py"
)


@pytest.fixture(scope="module")
def helpers():
    if not SCRIPT.exists():
        pytest.skip(f"E07 script not found: {SCRIPT}")
    src = SCRIPT.read_text(encoding="utf-8")
    # Locate helper definitions and exec them in an isolated namespace.
    # The helpers are pure (no top-level path side-effects) so this is safe.
    # We pull `tcga_patient_barcode` and `per_patient_f1` only.
    ns: dict = {}
    # Provide imports the helpers depend on.
    exec(
        "from __future__ import annotations\n"
        "import numpy as np\nimport pandas as pd\n"
        "from sklearn.metrics import f1_score\n",
        ns,
    )
    # Find each helper by anchoring on its def line and extract until a
    # blank line at column 0 (heuristic — the file uses standard
    # double-blank separators between defs).
    def _extract(symbol_name: str) -> str:
        lines = src.splitlines()
        out_lines, capturing = [], False
        for line in lines:
            if line.startswith(f"def {symbol_name}(") and not capturing:
                capturing = True
                out_lines.append(line)
                continue
            if capturing:
                if line.startswith(("def ", "class ", "# ---", "ROOT", "if __name__")):
                    break
                out_lines.append(line)
        return "\n".join(out_lines)

    for name in ("tcga_patient_barcode", "per_patient_f1"):
        block = _extract(name)
        if not block.strip():
            pytest.skip(f"Could not extract helper {name} from {SCRIPT}")
        exec(block, ns)
    return ns


@pytest.mark.unit
def test_tcga_patient_barcode_three_field(helpers):
    f = helpers["tcga_patient_barcode"]
    assert f("TCGA-AA-1234-01A") == "TCGA-AA-1234"
    assert f("TCGA-AA-1234") == "TCGA-AA-1234"
    # Defensively handle short input
    assert f("FOO") == "FOO"


@pytest.mark.unit
def test_tcga_patient_barcode_handles_extra_segments(helpers):
    f = helpers["tcga_patient_barcode"]
    assert f("TCGA-AA-1234-01A-12-DX") == "TCGA-AA-1234"


@pytest.mark.unit
def test_per_patient_f1_returns_one_score_per_patient(helpers):
    f = helpers["per_patient_f1"]
    df = pd.DataFrame(
        {
            "patient_id": (["p1"] * 4) + (["p2"] * 4),
            "hma_label": [1, 0, 1, 1, 0, 1, 1, 0],
            "is_active_predicted": [1, 0, 1, 0, 0, 1, 0, 0],
        }
    )
    result = f(df)
    assert set(result.index) == {"p1", "p2"}
    # Scores in [0, 1] (or NaN); none should be negative
    assert (result.dropna() >= 0).all()
    assert (result.dropna() <= 1).all()


@pytest.mark.unit
def test_per_patient_f1_perfect_prediction(helpers):
    f = helpers["per_patient_f1"]
    df = pd.DataFrame({
        "patient_id": ["p1"] * 4,
        "hma_label": [1, 0, 1, 0],
        "is_active_predicted": [1, 0, 1, 0],
    })
    out = f(df)
    assert out.loc["p1"] == pytest.approx(1.0)


@pytest.mark.unit
def test_per_patient_f1_skips_all_negative(helpers):
    """If no positives in either truth or pred → returns NaN (per docstring)."""
    f = helpers["per_patient_f1"]
    df = pd.DataFrame({
        "patient_id": ["p1"] * 3,
        "hma_label": [0, 0, 0],
        "is_active_predicted": [0, 0, 0],
    })
    out = f(df)
    assert pd.isna(out.loc["p1"])


# ─────────────────────────────────────────────────────────────────────
# Integration sanity test against synthetic seed-results tree
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_msi_lookup_pattern_round_trip(
    synthetic_msi_clinical, synthetic_seed_results, helpers,
):
    """
    Mimic the lookup loop the main() function performs: load MSI
    labels, scan seed_*/fold_*.csv files, attach `msi`, compute F1
    per patient — without writing outputs.
    """
    f = helpers["per_patient_f1"]
    barcode = helpers["tcga_patient_barcode"]

    msi_df = pd.read_csv(synthetic_msi_clinical, sep="\t")
    msi_df["patient_id"] = msi_df["tcga_barcode"].map(barcode)
    msi_lookup = dict(zip(msi_df["patient_id"], msi_df["msi_status"]))

    rows = []
    for seed_dir in sorted(synthetic_seed_results.glob("seed_*")):
        for fold_csv in sorted(seed_dir.glob("fold_*.csv")):
            df = pd.read_csv(fold_csv)
            df["patient_id_short"] = df["patient_id"].map(barcode)
            df["msi"] = df["patient_id_short"].map(msi_lookup)
            df = df[df["msi"].isin(["MSI-H", "MSS"])]
            per_pat = f(df)
            for pid, val in per_pat.items():
                if pd.isna(val):
                    continue
                rows.append((seed_dir.name, fold_csv.stem, pid, val))

    assert len(rows) > 0, "No (seed, fold, patient, F1) rows produced"

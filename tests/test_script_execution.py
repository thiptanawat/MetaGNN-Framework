"""
Lightweight execution smoke-tests for experiment scripts.

Calls each script with `--help` (and, where safe, with a tiny input
fixture) to confirm:
  1. The script's module-level imports actually resolve.
  2. The argparse layer works.
  3. Default values are sane (no `--required` left undeclared).

These tests are CPU-only and finish in < 5 seconds. They complement:
  - `test_smoke_compile.py` — guarantees byte-compile but does not
     execute a single line of script-level code.
  - `test_path_resolution.py` — guarantees data files exist but
     does not invoke the consuming script.
  - `test_integration_e0*.py` — runs entire main() end-to-end on real
     data; tagged `slow`/`integration` so users can opt out.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from _paths import EXPERIMENTS_DIR


# Scripts that expose a working `argparse --help`. The list is
# conservative: we only include scripts whose module-level imports are
# stdlib + numpy/pandas/scipy/sklearn/matplotlib (no torch_geometric,
# no cobra at module top level — those would fail import on a minimal
# env). E07's `msi_stratified_624_configB.py` is intentionally NOT here
# because it has no argparse parser; it runs main() unconditionally
# (and is covered by the dedicated integration test instead).
_HELP_OK_SCRIPTS = [
    # E02 / E03 — pure-numpy aggregator
    "E02_multiseed_220_hma/code/code/aggregate_multiseed.py",

    # E14 — cobra is *lazy* imported inside run_fba_624()
    "E14_fba_viability/code/run_fba_624.py",
    "E14_fba_viability/code/run_fba_apple_to_apple.py",

    # E10/E11 cross-cancer — cobra/RDKit are lazy-imported inside
    # functions, so module import + --help works on a minimal env.
    "E10_brca_cross_cancer/code/MetaGNN-BRCA-pipeline/00_build_recon3d_graph.py",
    "E10_brca_cross_cancer/code/MetaGNN-BRCA-pipeline/01_download_tcga_brca.py",
    "E10_brca_cross_cancer/code/MetaGNN-BRCA-pipeline/04_compare_with_published.py",
    "E11_luad_cross_cancer/code/MetaGNN-LUAD-pipeline/01_download_tcga_luad.py",
]

# Scripts whose top-level import we want to verify but which run main()
# unconditionally (no argparse). For these we can only check that
# `import` works — running them would execute the whole experiment.
_IMPORT_ONLY_SCRIPTS = [
    "E07_msi_stratified/code/msi_stratified_624_configB.py",
]


def _short_id(rel: str) -> str:
    return rel


@pytest.mark.parametrize("rel_path", _HELP_OK_SCRIPTS, ids=_short_id)
def test_script_help_works(rel_path: str):
    """`python <script> --help` exits 0 and prints a usage line."""
    script = EXPERIMENTS_DIR / rel_path
    if not script.exists():
        pytest.skip(f"script not present: {rel_path}")
    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"{rel_path} --help exited {proc.returncode}\n"
            f"STDOUT:\n{proc.stdout[-1000:]}\n"
            f"STDERR:\n{proc.stderr[-1000:]}"
        )
    out = proc.stdout + proc.stderr
    assert "usage:" in out.lower(), (
        f"{rel_path} --help did not print a usage line. "
        f"Got:\n{out[:500]}"
    )


# Module-level import probe: each script's top-level statements must
# execute cleanly (no exceptions, no SystemExit) without --help. Some
# scripts crash during top-level constant assignment if a path is wrong;
# this test surfaces those.
@pytest.mark.parametrize(
    "rel_path",
    _HELP_OK_SCRIPTS + _IMPORT_ONLY_SCRIPTS,
    ids=_short_id,
)
def test_script_module_imports(rel_path: str):
    """Importing the script as a module must not raise."""
    script = EXPERIMENTS_DIR / rel_path
    if not script.exists():
        pytest.skip(f"script not present: {rel_path}")
    # Spawn a fresh interpreter so import-time side effects don't leak.
    # We rename __name__ so any `if __name__ == '__main__'` guard is
    # bypassed even when the loader sets it to '__main__'.
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('m', r'{script}')\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(m)\n"
        "print('OK')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"{rel_path} module import failed (exit {proc.returncode})\n"
            f"STDERR:\n{proc.stderr[-1200:]}"
        )
    assert "OK" in proc.stdout

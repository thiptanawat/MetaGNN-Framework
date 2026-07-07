"""
Smoke tests — every Python script in experiments/E01..E14 must parse
cleanly. This is the most basic 'is the code valid Python?' check and
catches stale syntax, missing colons, broken indentation, etc.

These tests do NOT execute top-level code (no `__main__` runs), so even
files with hardcoded absolute paths will pass.

Implementation note: we issue ONE subprocess that batch-compiles every
discovered file. Per-file in-process py_compile triggers an
intermittent CPython 3.12 segfault on macOS when the parametrized
test count climbs into the high-double-digits, and per-file subprocess
calls hit a separate selectors/fork interaction late in the run.
A single subprocess avoids both pitfalls and is dramatically faster.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from _paths import discover_experiment_python_files  # noqa: E402


_FILES = discover_experiment_python_files()


def _short_id(p: Path) -> str:
    """Build a short human-readable test id from an experiment file path."""
    parts = p.parts
    for i, part in enumerate(parts):
        if part.startswith("E0") or part.startswith("E1"):
            return "/".join(parts[i:])
    return p.name


# ─────────────────────────────────────────────────────────────────────
# 1) Files exist at all
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.smoke
def test_at_least_one_python_file_exists():
    assert len(_FILES) > 0, (
        "No Python files were discovered under any "
        "experiments/E*/code/ tree — the experiment layout has changed "
        "or the suite is being run from the wrong directory."
    )


# ─────────────────────────────────────────────────────────────────────
# 2) Every file parses with ast (catches syntax errors fast, with
#    rich error messages)
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.smoke
@pytest.mark.parametrize("py_file", _FILES, ids=[_short_id(p) for p in _FILES])
def test_parses_with_ast(py_file: Path):
    src = py_file.read_text(encoding="utf-8", errors="replace")
    try:
        ast.parse(src, filename=str(py_file))
    except SyntaxError as e:
        pytest.fail(
            f"SyntaxError in {py_file}:{e.lineno}:{e.offset}\n"
            f"  {e.msg}\n"
            f"  near: {e.text!r}"
        )


# ─────────────────────────────────────────────────────────────────────
# 3) Batch byte-compile — runs once at module-import time, results
#    cached and surfaced as one test. This avoids the macOS Python 3.12
#    segfault we see when running subprocess.run() at parametrized
#    test-execution time.
# ─────────────────────────────────────────────────────────────────────


def _run_batch_compile():
    """Spawn one subprocess that calls py_compile on every file.

    Runs once when this module is imported (test collection time),
    before pytest has loaded any large packages. Returns a list of
    ``{file, error}`` dicts (empty when everything compiled).
    """
    if not _FILES:
        return []

    import os
    import tempfile

    # Re-affirm the macOS fork-safety opt-out — applied early enough
    # that subprocess.run during collection is reliable.
    os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

    helper_src = textwrap.dedent("""
        import json, py_compile, sys
        out_path = sys.argv[1]
        files = sys.argv[2:]
        errors = []
        for f in files:
            try:
                py_compile.compile(f, doraise=True)
            except py_compile.PyCompileError as e:
                errors.append({"file": f, "error": str(e)})
            except Exception as e:
                errors.append({"file": f, "error": f"{type(e).__name__}: {e}"})
        with open(out_path, "w") as fh:
            json.dump(errors, fh)
    """).strip()

    with tempfile.TemporaryDirectory() as td:
        helper = Path(td) / "_compile_all.py"
        helper.write_text(helper_src)
        out_json = Path(td) / "errors.json"
        proc = subprocess.run(
            [sys.executable, str(helper), str(out_json), *map(str, _FILES)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0 and not out_json.exists():
            return [{"file": "<helper>", "error":
                f"helper subprocess failed: stdout={proc.stdout} "
                f"stderr={proc.stderr}"}]
        return json.loads(out_json.read_text())


_COMPILE_ERRORS = _run_batch_compile()


@pytest.mark.smoke
def test_all_files_byte_compile():
    """All discovered .py files must byte-compile."""
    if _COMPILE_ERRORS:
        msg = "\n".join(
            f"  {e['file']}: {e['error']}"
            for e in _COMPILE_ERRORS
        )
        pytest.fail(f"{len(_COMPILE_ERRORS)} file(s) failed to compile:\n{msg}")

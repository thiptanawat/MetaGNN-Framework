"""
Shared utilities for the per-experiment validators.

Each validator returns a `ValidationReport` describing what was
checked, what was expected (per the manuscript), what was observed,
and whether it passed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────


# This module sits at: experiments/validators/_common.py
EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
JOURNALS_DIR = EXPERIMENTS_DIR.parent
SHARED_DATA = JOURNALS_DIR / "shared_data"


def journals_root() -> Path:
    """Resolve the Journals root, honouring the JOURNALS_ROOT env var."""
    return Path(os.environ.get("JOURNALS_ROOT", JOURNALS_DIR))


# ─────────────────────────────────────────────────────────────────────
# Report dataclasses
# ─────────────────────────────────────────────────────────────────────


@dataclass
class Check:
    """One assertion within a validation report."""
    name: str
    passed: bool
    expected: Any = None
    observed: Any = None
    tolerance: Any = None
    note: str = ""

    def format(self, indent: int = 2) -> str:
        symbol = "✅" if self.passed else "❌"
        head = f"{' ' * indent}{symbol} {self.name}"
        bits = []
        if self.expected is not None:
            bits.append(f"expected={self.expected!r}")
        if self.observed is not None:
            bits.append(f"observed={self.observed!r}")
        if self.tolerance is not None:
            bits.append(f"tol={self.tolerance!r}")
        details = ", ".join(bits)
        if details:
            head = head + f"  ({details})"
        if self.note:
            head = head + f"\n{' ' * (indent + 4)}{self.note}"
        return head


@dataclass
class ValidationReport:
    """Aggregate report for one experiment."""
    experiment: str
    description: str
    manuscript_anchor: str
    checks: list[Check] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""

    def add(self, check: Check) -> Check:
        self.checks.append(check)
        return check

    @property
    def passed(self) -> bool:
        if self.skipped:
            return True
        return all(c.passed for c in self.checks)

    @property
    def n_passed(self) -> int:
        return sum(c.passed for c in self.checks)

    @property
    def n_failed(self) -> int:
        return sum(not c.passed for c in self.checks)

    def format(self) -> str:
        if self.skipped:
            return (f"⚪ {self.experiment} — SKIPPED\n"
                    f"   {self.skip_reason}")
        head_symbol = "✅" if self.passed else "❌"
        lines = [
            f"{head_symbol} {self.experiment}: {self.description}",
            f"   Manuscript: {self.manuscript_anchor}",
            f"   Checks: {self.n_passed} passed, {self.n_failed} failed",
        ]
        for c in self.checks:
            lines.append(c.format(indent=3))
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# Tolerance-comparison helpers
# ─────────────────────────────────────────────────────────────────────


def assert_close(name: str, observed: float, expected: float,
                 atol: float = 0.01, note: str = "") -> Check:
    """Numeric proximity check."""
    diff = abs(observed - expected)
    return Check(
        name=name,
        passed=bool(diff <= atol),
        expected=round(expected, 6),
        observed=round(observed, 6),
        tolerance=f"±{atol}",
        note=note,
    )


def assert_in_range(name: str, observed: float, lo: float, hi: float,
                    note: str = "") -> Check:
    """Range check."""
    return Check(
        name=name,
        passed=bool(lo <= observed <= hi),
        expected=f"[{lo}, {hi}]",
        observed=round(observed, 6),
        note=note,
    )


def assert_eq(name: str, observed: Any, expected: Any, note: str = "") -> Check:
    return Check(
        name=name,
        passed=observed == expected,
        expected=expected,
        observed=observed,
        note=note,
    )


def assert_gte(name: str, observed: float, lo: float, note: str = "") -> Check:
    return Check(
        name=name,
        passed=bool(observed >= lo),
        expected=f"≥ {lo}",
        observed=observed,
        note=note,
    )


def assert_exists(name: str, path: Path, note: str = "") -> Check:
    """File or directory existence check (follows symlinks)."""
    try:
        ok = Path(path).resolve(strict=False).exists()
    except OSError:
        ok = False
    return Check(
        name=name,
        passed=ok,
        expected="exists",
        observed=str(path),
        note=note,
    )


# ─────────────────────────────────────────────────────────────────────
# Metric helpers — identical math to the manuscript's evaluation
# ─────────────────────────────────────────────────────────────────────


def f1_macro_from_predictions(df: pd.DataFrame,
                              y_true_col: str = "hma_label",
                              y_pred_col: str = "is_active_predicted") -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(df[y_true_col].astype(int),
                          df[y_pred_col].astype(int),
                          average="macro", zero_division=0))


def auroc_from_predictions(df: pd.DataFrame,
                           y_true_col: str = "hma_label",
                           y_score_col: str = "metagnn_score") -> float:
    from sklearn.metrics import roc_auc_score
    y = df[y_true_col].astype(int)
    if y.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y, df[y_score_col]))


def auprc_from_predictions(df: pd.DataFrame,
                           y_true_col: str = "hma_label",
                           y_score_col: str = "metagnn_score") -> float:
    from sklearn.metrics import average_precision_score
    y = df[y_true_col].astype(int)
    if y.nunique() < 2:
        return float("nan")
    return float(average_precision_score(y, df[y_score_col]))


# ─────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────


def load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def load_predictions_csv(path: Path) -> pd.DataFrame:
    """Load a fold prediction CSV with the canonical schema."""
    return pd.read_csv(path)


def iter_seed_dirs(root: Path) -> Iterable[Path]:
    """Yield seed_*/ subdirectories in numeric order if possible."""
    if not root.exists():
        return
    seeds = [p for p in root.iterdir()
             if p.is_dir() and p.name.startswith("seed_")]

    def _key(p):
        try:
            return int(p.name.split("_", 1)[1])
        except (ValueError, IndexError):
            return 1 << 30
    yield from sorted(seeds, key=_key)


# ─────────────────────────────────────────────────────────────────────
# Manuscript-anchor table
# ─────────────────────────────────────────────────────────────────────


# Headline numbers per EXPERIMENTS_INDEX.md / manuscript Table N.
# These are the values reviewers should see when running the validators.
MANUSCRIPT = {
    # Experiment ID: dict of expected metrics + tolerances
    "E01": {
        "section": "§3.1 (Table 1, Figure 3)",
        "F1_220": 0.806, "F1_220_tol": 0.05,
        "AUROC_220": 0.866, "AUROC_220_tol": 0.05,
    },
    "E02": {
        "section": "§3.2 (Table 2)",
        "F1_220_mean": 0.806, "F1_220_std_max": 0.03,
    },
    "E03": {
        "section": "§3.2 (Table 3)",
        "F1_624_mean": 0.461, "F1_624_std_max": 0.03,
        "AUROC_624_mean": 0.558, "AUROC_624_std_max": 0.03,
    },
    "E04": {
        "section": "§3.3 (Table 4, Figure X1)",
        "directions": ("220_to_624", "624_to_220"),
    },
    "E05": {
        "section": "§3.4 (Table 5)",
    },
    "E06": {
        "section": "§3.15.5 (METABRIC four-discriminator framework)",
        # Manuscript Δ-AUROC values per discriminator, (all_reactions, gpr_only)
        "rank_transform_delta_all": +0.007,
        "rank_transform_delta_gpr": -0.001,
        "hma_relabel_delta_all":    +0.019,
        "hma_relabel_delta_gpr":    +0.038,
        "combat_delta_all":         -0.018,
        "combat_delta_gpr":         -0.027,
        "cptac_brca_auroc":         0.499,
        "cptac_brca_n_patients":    106,
        "delta_tol": 0.015,
    },
    "E07": {
        "section": "§3.5 (Table 6)",
        "n_msi_h": 92, "n_mss": 487, "n_evaluable": 579,
        "delta_max": 0.05,  # |F1_MSI-H - F1_MSS|
    },
    "E08": {
        "section": "Future Work (Pass F)",
    },
    "E09": {
        "section": "§3.6 (Table 7)",
        "auroc_624_B_baseline": 0.5584, "auroc_624_B_tol": 0.03,
    },
    "E10": {
        "section": "§3.7 (Table 8)",
    },
    "E11": {
        "section": "§3.7 (Table 9)",
    },
    "E12": {
        "section": "§3.10 (Table 11, tab:dual_llm_results)",
        "qwen3_v6_delta": +0.014,
        "gemma4_v6_delta": +0.052,
        "delta_tol": 0.015,
    },
    "E13": {
        "section": "§3.11 (Table 12, tab:cross_backbone_extended)",
        "gpt_oss_20b_delta": -0.001,
        "openbio_8b_delta":  -0.000,
        "deepseek_r1_32b_delta": -0.032,
        "delta_tol_loose": 0.020,
    },
    "E14": {
        "section": "§3.13 (Table 13)",
    },
}

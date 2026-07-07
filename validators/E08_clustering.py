"""
E08 — Clustering pilot validator (Pass F).

Manuscript anchor: Future Work section.

Validation strategy:
- The shipped summary.json is a list of (representation, algorithm,
  silhouette, ARI, within-cluster F1 spread) cells.
- Verify schema and that at least one (rep, algo) cell is present.
- Sanity-check: silhouette in [-1, 1], k ≥ 2 for valid clusterings.
- Verify the go/no-go verdict text is present.
"""

from __future__ import annotations

from . import _common as C
from ._common import (Check, ValidationReport, assert_eq, assert_exists,
                      assert_in_range, load_json, MANUSCRIPT)


def validate(exp_id: str = "E08", verbose: bool = False) -> ValidationReport:
    r = ValidationReport(
        experiment="E08",
        description="Clustering pilot Pass F",
        manuscript_anchor=MANUSCRIPT["E08"]["section"],
    )

    base = C.EXPERIMENTS_DIR / "E08_clustering_pilot_passF" / "results"
    if not base.exists():
        r.skipped = True
        r.skip_reason = f"results root missing: {base}"
        return r

    # Use the v2 (latest) result directory.
    snapshot = base / "v2"
    summary_json = snapshot / "summary.json"
    go_no_go = snapshot / "go_no_go.txt"
    cluster_csv = snapshot / "cluster_assignments.csv"

    r.add(assert_exists("v2/summary.json", summary_json))
    r.add(assert_exists("v2/go_no_go.txt", go_no_go))
    r.add(assert_exists("v2/cluster_assignments.csv", cluster_csv))

    if not summary_json.exists():
        return r

    rows = load_json(summary_json)
    r.add(Check(
        name="summary is a non-empty list of cells",
        passed=isinstance(rows, list) and len(rows) > 0,
        expected="non-empty list",
        observed=f"{len(rows) if isinstance(rows, list) else 'not a list'} cells",
    ))
    if not isinstance(rows, list) or not rows:
        return r

    expected_keys = {"representation", "algorithm", "silhouette",
                     "n_clusters", "within_cluster_f1_spread"}
    sample = rows[0]
    r.add(Check(
        name="cell schema includes (rep, algo, silhouette, n_clusters, spread)",
        passed=expected_keys.issubset(sample.keys()),
        expected=expected_keys,
        observed=set(sample.keys()),
    ))

    # ── Sanity checks per cell ────────────────────────────────────
    for i, cell in enumerate(rows):
        sil = cell.get("silhouette")
        n_clusters = cell.get("n_clusters")
        if sil is not None:
            r.add(assert_in_range(
                f"cell {i}: silhouette in [-1, 1]",
                observed=float(sil), lo=-1.0, hi=1.0,
            ))
        if n_clusters is not None:
            r.add(Check(
                name=f"cell {i}: n_clusters ≥ 2",
                passed=int(n_clusters) >= 2,
                expected="≥ 2",
                observed=int(n_clusters),
            ))

    # ── go/no-go verdict ───────────────────────────────────────────
    if go_no_go.exists():
        text = go_no_go.read_text()
        verdict_present = any(v in text.upper()
                              for v in ("POSITIVE", "NULL", "INCONCLUSIVE",
                                        "GO", "NO-GO", "NO GO"))
        r.add(Check(
            name="go/no-go verdict line present",
            passed=verdict_present,
            expected="POSITIVE / NULL / INCONCLUSIVE / GO / NO-GO",
            observed=text.strip().split("\n")[0][:80],
        ))

    return r

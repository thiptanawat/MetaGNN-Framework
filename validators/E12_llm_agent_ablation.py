"""
E12 — LLM agent v1–v6 ablation validator (Qwen3-32B + Gemma 4-31B-IT).

Manuscript anchor: §3.10, Table tab:dual_llm_results.

For each shipped backbone × version, we recompute the per-fold AUROC of
``metagnn_score`` (the GNN baseline) and ``s_final`` (the agent-corrected
score) from ``v6_langgraph_kg/fold_*/agent_results.json`` and check that
the ΔAUROC matches what the manuscript reports.

The headline numbers (Table 11, line 1262 ff.):

  Qwen3-32B v6:    ΔAUROC = +0.014 ± 0.013
  Gemma 4-31B v6:  ΔAUROC = +0.052 ± 0.031

These are the only two (backbone, version) cells that the manuscript
distinguishes as the v6 final result; the other v1–v5 cells in
Table tab:dual_llm_results are reported but with much higher
seed-fold variance, so we restrict the tight numerical check to v6.

If the ``sklearn`` import fails (rare on minimal CI), the validator
SKIPS rather than fails.
"""

from __future__ import annotations

from pathlib import Path

from . import _common as C
from ._common import (Check, ValidationReport, assert_close, assert_exists,
                      load_json)


def _fold_aurocs(v6_dir: Path) -> tuple[list[float], list[float]]:
    """Return (gnn_auroc_per_fold, sfinal_auroc_per_fold) over fold_* subdirs."""
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        raise RuntimeError("sklearn not available")
    gnn, fin = [], []
    for fold_dir in sorted(v6_dir.glob("fold_*")):
        res = fold_dir / "agent_results.json"
        if not res.exists():
            continue
        d = load_json(res)
        if not isinstance(d, dict) or not d:
            continue
        rows = list(d.values())
        y = [int(v.get("hma_label", 0)) for v in rows]
        if len(set(y)) < 2:
            continue
        sg = [float(v.get("metagnn_score", 0.0)) for v in rows]
        sf = [float(v.get("s_final", v.get("metagnn_score", 0.0))) for v in rows]
        gnn.append(float(roc_auc_score(y, sg)))
        fin.append(float(roc_auc_score(y, sf)))
    return gnn, fin


def validate(exp_id: str = "E12", verbose: bool = False) -> ValidationReport:
    r = ValidationReport(
        experiment="E12",
        description="LLM agent v1–v6 ablation (Qwen3 + Gemma 4)",
        manuscript_anchor="§3.10 (Table 11, tab:dual_llm_results)",
    )

    base = C.EXPERIMENTS_DIR / "E12_llm_agent_ablation" / "results"
    qwen_v6 = base / "qwen3_v6"     / "v6_langgraph_kg"
    gemma_v6 = base / "results_gemma4" / "v6_langgraph_kg"

    r.add(assert_exists("qwen3_v6/v6_langgraph_kg", qwen_v6))
    r.add(assert_exists("results_gemma4/v6_langgraph_kg", gemma_v6))

    try:
        # Qwen3-32B v6
        if qwen_v6.exists():
            gnn, fin = _fold_aurocs(qwen_v6)
            r.add(Check(
                name="Qwen3 v6: 5 folds present",
                passed=(len(gnn) == 5),
                expected=5, observed=len(gnn),
            ))
            if gnn:
                mean_delta = sum(f - g for f, g in zip(fin, gnn)) / len(gnn)
                mean_gnn = sum(gnn) / len(gnn)
                mean_fin = sum(fin) / len(fin)
                r.add(assert_close(
                    "Qwen3 v6: ΔAUROC ≈ +0.014",
                    observed=mean_delta, expected=+0.014, atol=0.010,
                    note="Manuscript Table 11 (line 1297-ish): instruction-tuned generalist with smallest positive lift.",
                ))
                r.add(assert_close(
                    "Qwen3 v6: GNN baseline AUROC ≈ 0.507",
                    observed=mean_gnn, expected=0.507, atol=0.015,
                ))
                r.add(assert_close(
                    "Qwen3 v6: s_final AUROC ≈ 0.521",
                    observed=mean_fin, expected=0.521, atol=0.015,
                ))

        # Gemma 4-31B v6
        if gemma_v6.exists():
            gnn, fin = _fold_aurocs(gemma_v6)
            r.add(Check(
                name="Gemma 4 v6: 5 folds present",
                passed=(len(gnn) == 5),
                expected=5, observed=len(gnn),
            ))
            if gnn:
                mean_delta = sum(f - g for f, g in zip(fin, gnn)) / len(gnn)
                mean_gnn = sum(gnn) / len(gnn)
                mean_fin = sum(fin) / len(fin)
                r.add(assert_close(
                    "Gemma 4 v6: ΔAUROC ≈ +0.052",
                    observed=mean_delta, expected=+0.052, atol=0.015,
                    note="Manuscript Table 11: best-performing backbone for the v6 agent loop.",
                ))
                r.add(assert_close(
                    "Gemma 4 v6: GNN baseline AUROC ≈ 0.507",
                    observed=mean_gnn, expected=0.507, atol=0.015,
                ))
                r.add(assert_close(
                    "Gemma 4 v6: s_final AUROC ≈ 0.559",
                    observed=mean_fin, expected=0.559, atol=0.015,
                ))

    except RuntimeError as e:
        r.skipped = True
        r.skip_reason = str(e)

    return r

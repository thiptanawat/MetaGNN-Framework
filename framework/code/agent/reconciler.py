"""
Score reconciliation module.

Implements weighted fusion of GNN and LLM predictions with configurable override logic.
Formula: s_final = w_GNN * s_r + w_LLM * p_r
Override: when LLM disagrees AND p_r > threshold, apply stronger correction.
"""

import logging
from typing import Dict, Optional

import pandas as pd


logger = logging.getLogger(__name__)


class Reconciler:
    """Reconciles GNN and LLM predictions using weighted fusion."""

    def __init__(
        self,
        w_gnn: float = 0.7,
        w_llm: float = 0.3,
        override_threshold: float = 0.8,
    ):
        """Initialize reconciler with weights and threshold.

        Args:
            w_gnn: Weight for GNN score (default 0.7).
            w_llm: Weight for LLM plausibility score (default 0.3).
            override_threshold: LLM plausibility threshold for override (default 0.8).

        Raises:
            ValueError: If weights don't sum to 1.0.
        """
        if not (0.98 < w_gnn + w_llm < 1.02):
            raise ValueError(
                f"Weights must sum to 1.0, got {w_gnn + w_llm}"
            )

        self.w_gnn = w_gnn
        self.w_llm = w_llm
        self.override_threshold = override_threshold

    def _check_disagreement(self, s_r: float, p_r: float) -> bool:
        """Check if GNN and LLM predictions disagree significantly.

        Disagreement defined as:
        - GNN active (s_r > 0.5) and LLM inactive (p_r < 0.5), or
        - GNN inactive (s_r < 0.5) and LLM active (p_r > 0.5)

        Args:
            s_r: GNN score.
            p_r: LLM plausibility score.

        Returns:
            True if disagreement detected.
        """
        gnn_active = s_r > 0.5
        llm_active = p_r > 0.5
        return gnn_active != llm_active

    def _compute_reconciled_score(
        self, s_r: float, p_r: float
    ) -> float:
        """Compute weighted reconciliation score.

        Args:
            s_r: GNN score.
            p_r: LLM plausibility score.

        Returns:
            Reconciled score between 0 and 1.
        """
        return self.w_gnn * s_r + self.w_llm * p_r

    def _apply_override_correction(
        self, s_r: float, p_r: float, reconciled: float
    ) -> float:
        """Apply stronger correction when LLM confidence is high.

        Override logic:
        - If LLM disagrees AND p_r > threshold, shift reconciled score
          toward LLM prediction with 60/40 weighting.

        Args:
            s_r: GNN score.
            p_r: LLM plausibility score.
            reconciled: Weighted reconciliation score.

        Returns:
            Corrected score with potential override applied.
        """
        if not self._check_disagreement(s_r, p_r):
            return reconciled

        if p_r > self.override_threshold:
            # Apply stronger correction: 60% LLM, 40% weighted baseline
            target = 0.9 if p_r > 0.5 else 0.1
            return 0.6 * target + 0.4 * reconciled

        return reconciled

    def reconcile(
        self, gnn_df: pd.DataFrame, llm_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Reconcile GNN and LLM predictions.

        Args:
            gnn_df: GNN predictions with columns: reaction_id, s_r, sigma_r, predicted_state.
            llm_df: LLM results with columns: reaction_id, p_r, suggested_action, etc.

        Returns:
            DataFrame with reconciled scores and metadata.
        """
        # Merge on reaction_id
        merged = gnn_df.merge(
            llm_df[["reaction_id", "p_r", "suggested_action", "reasoning"]],
            on="reaction_id",
            how="inner"
        )

        results = []

        for _, row in merged.iterrows():
            s_r = row["s_r"]
            p_r = row["p_r"]

            # Compute reconciled score
            reconciled = self._compute_reconciled_score(s_r, p_r)

            # Check for override
            if self._check_disagreement(s_r, p_r):
                corrected = self._apply_override_correction(s_r, p_r, reconciled)
                override_applied = True
            else:
                corrected = reconciled
                override_applied = False

            # Determine final state
            final_state = 1 if corrected > 0.5 else 0

            results.append({
                "reaction_id": row["reaction_id"],
                "s_r_gnn": s_r,
                "p_r_llm": p_r,
                "s_final": corrected,
                "predicted_state_gnn": row.get("predicted_state", 0),
                "predicted_state_final": final_state,
                "disagreement": self._check_disagreement(s_r, p_r),
                "override_applied": override_applied,
                "llm_action": row.get("suggested_action", "agree"),
                "reasoning": row.get("reasoning", ""),
            })

        result_df = pd.DataFrame(results)

        # Log statistics
        n_override = result_df["override_applied"].sum()
        n_disagreement = result_df["disagreement"].sum()
        n_state_change = (
            result_df["predicted_state_gnn"] != result_df["predicted_state_final"]
        ).sum()

        logger.info(
            f"Reconciliation complete: {len(result_df)} reactions, "
            f"{n_disagreement} disagreements, {n_override} overrides, "
            f"{n_state_change} state changes"
        )

        return result_df

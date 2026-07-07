"""
Main entry point for Module 6 LLM-based agentic validation.

Orchestrates prediction reconciliation using selected agent version.
Handles data loading, filtering, version dispatch, and result saving.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

# Import version handlers
from v1_bare_llm import BareAgent
from v2_enriched_prompt import EnrichedAgent
from v3_advocate_resolver import AdvocateResolverAgent
from v4_patient_rag import PatientRAGAgent
from v5_faiss_retrieval import FAISSAgent
from v6_langgraph_kg import LangGraphAgent
from reconciler import Reconciler


logger = logging.getLogger(__name__)


class AgentDispatcher:
    """Orchestrates agent version selection and execution pipeline."""

    AGENT_VERSIONS = {
        "v1": BareAgent,
        "v2": EnrichedAgent,
        "v3": AdvocateResolverAgent,
        "v4": PatientRAGAgent,
        "v5": FAISSAgent,
        "v6": LangGraphAgent,
    }

    def __init__(self, config_path: str):
        """Initialize dispatcher with configuration.

        Args:
            config_path: Path to agent.yaml configuration file.
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.reconciler = Reconciler(
            w_gnn=self.config.get("reconciler_weights", {}).get("gnn", 0.7),
            w_llm=self.config.get("reconciler_weights", {}).get("llm", 0.3),
            override_threshold=self.config.get("override_threshold", 0.8),
        )

    def load_gnn_predictions(
        self, checkpoint_path: str, data_root: str
    ) -> pd.DataFrame:
        """Load GNN predictions and uncertainty estimates.

        Expects Module 4/5 output format with columns:
        - reaction_id
        - s_r (prediction score)
        - sigma_r (uncertainty)
        - predicted_state (binary activity)

        Args:
            checkpoint_path: Path to Module 4/5 checkpoint directory.
            data_root: Root data directory.

        Returns:
            DataFrame with GNN predictions and metadata.
        """
        pred_file = Path(checkpoint_path) / "predictions.parquet"
        if not pred_file.exists():
            raise FileNotFoundError(f"Predictions not found: {pred_file}")

        df = pd.read_parquet(pred_file)
        logger.info(f"Loaded {len(df)} predictions from {pred_file}")
        return df

    def filter_boundary_reactions(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Filter reactions in uncertainty boundary zone.

        Criteria:
        - 0.3 < s_r < 0.7 (uncertain region)
        - sigma_r above median (high variance)
        - GPR-associated (has gene annotations)

        Args:
            df: DataFrame with all predictions.

        Returns:
            Filtered DataFrame of boundary reactions.
        """
        boundary_min = self.config.get("boundary_range", [0.3, 0.7])[0]
        boundary_max = self.config.get("boundary_range", [0.3, 0.7])[1]

        # Score range filter
        mask_score = (df["s_r"] > boundary_min) & (df["s_r"] < boundary_max)

        # Uncertainty filter (above median)
        sigma_median = df["sigma_r"].median()
        mask_sigma = df["sigma_r"] > sigma_median

        # GPR-associated filter
        mask_gpr = df["gpr"].notna() & (df["gpr"].str.len() > 0)

        filtered = df[mask_score & mask_sigma & mask_gpr].copy()
        logger.info(
            f"Filtered to {len(filtered)} boundary reactions "
            f"(score: {boundary_min}-{boundary_max}, sigma > {sigma_median:.4f}, GPR+)"
        )
        return filtered

    def dispatch_version(
        self, version: str, reactions: pd.DataFrame, output_dir: str
    ) -> pd.DataFrame:
        """Dispatch to appropriate agent version handler.

        Args:
            version: Version identifier (v1-v6).
            reactions: DataFrame of boundary reactions to validate.
            output_dir: Directory for version-specific outputs.

        Returns:
            DataFrame with agent validation results.

        Raises:
            ValueError: If version not recognized.
        """
        if version not in self.AGENT_VERSIONS:
            raise ValueError(
                f"Unknown version {version}. "
                f"Available: {', '.join(self.AGENT_VERSIONS.keys())}"
            )

        agent_class = self.AGENT_VERSIONS[version]
        agent = agent_class(self.config)

        logger.info(f"Running {version.upper()} agent on {len(reactions)} reactions")
        results = agent.validate(reactions)

        # Save version-specific results
        output_path = Path(output_dir) / f"{version}_results.parquet"
        results.to_parquet(output_path)
        logger.info(f"Saved {version.upper()} results to {output_path}")

        return results

    def reconcile_predictions(
        self, gnn_df: pd.DataFrame, llm_results: pd.DataFrame
    ) -> pd.DataFrame:
        """Reconcile GNN and LLM predictions using configured formula.

        Args:
            gnn_df: Original GNN predictions.
            llm_results: LLM agent validation results.

        Returns:
            DataFrame with reconciled predictions.
        """
        reconciled = self.reconciler.reconcile(gnn_df, llm_results)
        logger.info(f"Reconciled {len(reconciled)} predictions")
        return reconciled

    def run(
        self,
        version: str,
        checkpoint_path: str,
        data_root: str,
        output_dir: str,
    ):
        """Execute complete validation pipeline.

        Args:
            version: Agent version (v1-v6) or 'all' to run sequentially.
            checkpoint_path: Path to Module 4/5 checkpoint.
            data_root: Root data directory.
            output_dir: Output directory for results.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Load and filter predictions
        gnn_df = self.load_gnn_predictions(checkpoint_path, data_root)
        boundary_df = self.filter_boundary_reactions(gnn_df)

        if len(boundary_df) == 0:
            logger.warning("No boundary reactions found for validation")
            return

        # Execute version(s)
        if version.lower() == "all":
            versions = list(self.AGENT_VERSIONS.keys())
        else:
            versions = [version]

        all_results = []
        for v in versions:
            results = self.dispatch_version(v, boundary_df, output_dir)
            reconciled = self.reconcile_predictions(gnn_df, results)
            all_results.append(reconciled)

        # Save master reconciliation
        final_df = pd.concat(all_results, axis=0).drop_duplicates(subset=["reaction_id"])
        final_path = Path(output_dir) / "reconciled_predictions.parquet"
        final_df.to_parquet(final_path)
        logger.info(f"Saved reconciled predictions to {final_path}")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Module 6: LLM-based agentic validation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/agent.yaml",
        help="Path to agent configuration file",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v6",
        choices=["v1", "v2", "v3", "v4", "v5", "v6", "all"],
        help="Agent version to run",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to Module 4/5 checkpoint directory",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Output directory for results",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        dispatcher = AgentDispatcher(args.config)
        dispatcher.run(
            version=args.version,
            checkpoint_path=args.checkpoint,
            data_root=args.data_root,
            output_dir=args.output_dir,
        )
        logger.info("Agent validation complete")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

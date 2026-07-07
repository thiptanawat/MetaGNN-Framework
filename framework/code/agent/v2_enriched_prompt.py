"""
Version 2: Enhanced single-agent with Recon3D biochemical context.

Extends v1 with structured metadata: reaction name, BiGG ID, subsystem,
reversibility, substrates/products, GPR gene IDs.
Single LLM call per reaction.
"""

import json
import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from utils.llm_serving import LLMClient


logger = logging.getLogger(__name__)


class EnrichedAgent:
    """Single-agent with enriched biochemical metadata."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize agent with configuration.

        Args:
            config: Configuration dictionary from agent.yaml.
        """
        self.config = config
        self.llm = LLMClient(
            endpoint=config.get("llm_endpoint", "http://localhost:8000/v1"),
            model=config.get("llm_model", "Qwen3-32B-AWQ"),
            temperature=config.get("temperature_resolver", 0.3),
        )

    def _format_metabolites(self, metabolites: Optional[str]) -> str:
        """Format metabolite list for display.

        Args:
            metabolites: Comma-separated metabolite IDs.

        Returns:
            Formatted string or "None".
        """
        if not metabolites or (isinstance(metabolites, float) and pd.isna(metabolites)):
            return "None"
        return str(metabolites)

    def _build_prompt(self, reaction: pd.Series) -> str:
        """Build enriched prompt with Recon3D metadata.

        Args:
            reaction: Row from boundary reactions DataFrame.

        Returns:
            Prompt string for LLM.
        """
        prompt = f"""Evaluate metabolic reaction activity using biochemical context:

PREDICTION METRICS
Reaction ID: {reaction.get('reaction_id', 'UNKNOWN')}
BiGG ID: {reaction.get('bigg_id', 'N/A')}
MetaGNN Score: {reaction.get('s_r', 0.5):.4f}
Uncertainty: {reaction.get('sigma_r', 0.0):.4f}
Predicted State: {'Active' if reaction.get('predicted_state', 0) else 'Inactive'}

BIOCHEMICAL METADATA
Reaction Name: {reaction.get('name', 'Unknown')}
Subsystem: {reaction.get('subsystem', 'Unknown')}
Reversibility: {reaction.get('reversible', 'Unknown')}

STOICHIOMETRY
Substrates: {self._format_metabolites(reaction.get('substrates', None))}
Products: {self._format_metabolites(reaction.get('products', None))}

GENETICS
Associated Genes: {reaction.get('gpr', 'None')}

Given this metabolic context, assess the plausibility of the MetaGNN prediction.
Consider subsystem function, genetic association, and biochemical feasibility.

Respond with JSON:
{{
    "plausibility_score": <float 0-1>,
    "reasoning": "<explanation incorporating metabolic context>",
    "suggested_action": "<agree|activate|deactivate>"
}}
"""
        return prompt

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response into structured format.

        Args:
            response_text: Raw LLM response text.

        Returns:
            Dictionary with parsed fields.
        """
        try:
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                result = json.loads(json_str)
            else:
                result = {
                    "plausibility_score": 0.5,
                    "reasoning": "Parse error",
                    "suggested_action": "agree",
                }
        except json.JSONDecodeError:
            result = {
                "plausibility_score": 0.5,
                "reasoning": "JSON parse error",
                "suggested_action": "agree",
            }

        # Validate and normalize
        result["plausibility_score"] = float(result.get("plausibility_score", 0.5))
        result["plausibility_score"] = max(0.0, min(1.0, result["plausibility_score"]))
        result["reasoning"] = str(result.get("reasoning", ""))
        result["suggested_action"] = str(result.get("suggested_action", "agree")).lower()

        return result

    def validate(self, reactions: pd.DataFrame) -> pd.DataFrame:
        """Validate reactions using enriched LLM agent.

        Args:
            reactions: DataFrame of boundary reactions with Recon3D metadata.

        Returns:
            DataFrame with validation results.
        """
        results = []

        for idx, reaction in reactions.iterrows():
            try:
                prompt = self._build_prompt(reaction)
                response = self.llm.generate(prompt)
                parsed = self._parse_response(response)

                results.append({
                    "reaction_id": reaction.get("reaction_id", f"rxn_{idx}"),
                    "s_r": reaction.get("s_r", 0.5),
                    "sigma_r": reaction.get("sigma_r", 0.0),
                    "predicted_state": reaction.get("predicted_state", 0),
                    "p_r": parsed["plausibility_score"],
                    "reasoning": parsed["reasoning"],
                    "suggested_action": parsed["suggested_action"],
                })

                if (idx + 1) % 10 == 0:
                    logger.info(f"Processed {idx + 1}/{len(reactions)} reactions")

            except Exception as e:
                logger.error(f"Error processing reaction {idx}: {e}")
                results.append({
                    "reaction_id": reaction.get("reaction_id", f"rxn_{idx}"),
                    "s_r": reaction.get("s_r", 0.5),
                    "sigma_r": reaction.get("sigma_r", 0.0),
                    "predicted_state": reaction.get("predicted_state", 0),
                    "p_r": 0.5,
                    "reasoning": f"Error: {str(e)}",
                    "suggested_action": "agree",
                })

        return pd.DataFrame(results)

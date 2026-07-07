"""
Version 3: Multi-agent debate framework.

Three-stage pipeline: activation advocate, deactivation advocate, resolver.
Each advocate presents evidence for their position, resolver synthesizes.
Includes self-reflection and bias checking in resolver.
Three LLM calls per reaction.
"""

import json
import logging
from typing import Any, Dict, Optional

import pandas as pd

from utils.llm_serving import LLMClient


logger = logging.getLogger(__name__)


class AdvocateResolverAgent:
    """Multi-agent debate with activation/deactivation advocates and resolver."""

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

    def _advocate_activate(self, reaction: pd.Series) -> str:
        """Generate activation advocate argument.

        Args:
            reaction: Reaction data.

        Returns:
            Advocate's argument for activity.
        """
        prompt = f"""You are an activation advocate. Argue why reaction {reaction.get('reaction_id')} 
should be ACTIVE in the colorectal cancer context.

Focus on:
- CRC-upregulated pathways supporting this reaction
- Biochemical feasibility
- Gene expression evidence if available

Be concise but convincing. Reaction: {reaction.get('name', 'Unknown')}
Subsystem: {reaction.get('subsystem', 'Unknown')}
GPR: {reaction.get('gpr', 'None')}

Provide your argument:"""

        response = self.llm.generate(prompt)
        return response

    def _advocate_deactivate(self, reaction: pd.Series) -> str:
        """Generate deactivation advocate argument.

        Args:
            reaction: Reaction data.

        Returns:
            Advocate's argument for inactivity.
        """
        prompt = f"""You are a deactivation advocate. Argue why reaction {reaction.get('reaction_id')} 
should be INACTIVE in the colorectal cancer context.

Focus on:
- Tissue-specific inactivity patterns
- Biochemical constraints
- Lack of supporting gene expression

Be concise but convincing. Reaction: {reaction.get('name', 'Unknown')}
Subsystem: {reaction.get('subsystem', 'Unknown')}
GPR: {reaction.get('gpr', 'None')}

Provide your argument:"""

        response = self.llm.generate(prompt)
        return response

    def _resolve_debate(
        self, reaction: pd.Series, activate_arg: str, deactivate_arg: str
    ) -> Dict[str, Any]:
        """Synthesize advocate arguments with self-reflection and bias checking.

        Args:
            reaction: Reaction data.
            activate_arg: Activation advocate's argument.
            deactivate_arg: Deactivation advocate's argument.

        Returns:
            Resolver's decision with plausibility score and reasoning.
        """
        prompt = f"""You are a resolver mediating a debate on reaction {reaction.get('reaction_id')} activity.

REACTION CONTEXT
Name: {reaction.get('name', 'Unknown')}
Subsystem: {reaction.get('subsystem', 'Unknown')}
MetaGNN Score: {reaction.get('s_r', 0.5):.4f}

ACTIVATION ADVOCATE'S ARGUMENT:
{activate_arg}

DEACTIVATION ADVOCATE'S ARGUMENT:
{deactivate_arg}

As resolver, evaluate both arguments. Consider:
1. Which argument is more supported?
2. Are there logical biases or gaps?
3. What does the evidence truly suggest?

Respond with JSON:
{{
    "plausibility_score": <float 0-1>,
    "reasoning": "<synthesis of both arguments with self-reflection>",
    "bias_check": "<any identified biases or counterarguments>",
    "suggested_action": "<agree|activate|deactivate>"
}}
"""

        response = self.llm.generate(prompt)
        return self._parse_response(response)

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse resolver response into structured format.

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
                    "bias_check": "",
                    "suggested_action": "agree",
                }
        except json.JSONDecodeError:
            result = {
                "plausibility_score": 0.5,
                "reasoning": "JSON parse error",
                "bias_check": "",
                "suggested_action": "agree",
            }

        result["plausibility_score"] = float(result.get("plausibility_score", 0.5))
        result["plausibility_score"] = max(0.0, min(1.0, result["plausibility_score"]))
        result["reasoning"] = str(result.get("reasoning", ""))
        result["bias_check"] = str(result.get("bias_check", ""))
        result["suggested_action"] = str(result.get("suggested_action", "agree")).lower()

        return result

    def validate(self, reactions: pd.DataFrame) -> pd.DataFrame:
        """Validate reactions using three-stage debate framework.

        Args:
            reactions: DataFrame of boundary reactions.

        Returns:
            DataFrame with validation results from resolver.
        """
        results = []

        for idx, reaction in reactions.iterrows():
            try:
                # Stage 1: Activation advocate
                activate_arg = self._advocate_activate(reaction)

                # Stage 2: Deactivation advocate
                deactivate_arg = self._advocate_deactivate(reaction)

                # Stage 3: Resolver synthesis
                decision = self._resolve_debate(reaction, activate_arg, deactivate_arg)

                results.append({
                    "reaction_id": reaction.get("reaction_id", f"rxn_{idx}"),
                    "s_r": reaction.get("s_r", 0.5),
                    "sigma_r": reaction.get("sigma_r", 0.0),
                    "predicted_state": reaction.get("predicted_state", 0),
                    "p_r": decision["plausibility_score"],
                    "reasoning": decision["reasoning"],
                    "bias_check": decision.get("bias_check", ""),
                    "suggested_action": decision["suggested_action"],
                    "activate_argument": activate_arg,
                    "deactivate_argument": deactivate_arg,
                })

                if (idx + 1) % 5 == 0:
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
                    "bias_check": "",
                    "suggested_action": "agree",
                    "activate_argument": "",
                    "deactivate_argument": "",
                })

        return pd.DataFrame(results)

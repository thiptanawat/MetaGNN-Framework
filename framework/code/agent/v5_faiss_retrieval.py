"""
Version 5: Multi-agent debate with FAISS semantic retrieval.

Extends v4 with FAISS index built from metabolic knowledge base.
Uses sentence-transformers/all-MiniLM-L6-v2 (384-dim) for embeddings.
Top-5 passages per reaction via cosine similarity.
Approximately 70-80% evidence coverage (GPR-independent).
"""

import json
import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from utils.faiss_index import FAISSIndex
from utils.kegg_client import KEGGClient
from utils.llm_serving import LLMClient


logger = logging.getLogger(__name__)


class FAISSAgent:
    """Multi-agent with semantic retrieval via FAISS."""

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
        self.kegg = KEGGClient(
            cache_dir=config.get("kegg_cache_dir", "./cache/kegg")
        )
        self.faiss_index = FAISSIndex(
            index_path=config.get("faiss_index_path", "./cache/faiss"),
            model_name=config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
        )

    def _retrieve_passages(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """Retrieve top-k passages from FAISS index.

        Args:
            query: Query string (reaction name, subsystem, description).
            top_k: Number of passages to retrieve (default from config).

        Returns:
            List of retrieved passages.
        """
        if top_k is None:
            top_k = self.config.get("faiss_top_k", 5)

        try:
            passages = self.faiss_index.search(query, top_k=top_k)
            return passages
        except Exception as e:
            logger.warning(f"FAISS retrieval failed for '{query}': {e}")
            return []

    def _advocate_activate(
        self, reaction: pd.Series, retrieved_passages: List[str]
    ) -> str:
        """Generate activation advocate argument with FAISS context.

        Args:
            reaction: Reaction data.
            retrieved_passages: Semantically similar passages from knowledge base.

        Returns:
            Advocate's argument for activity.
        """
        passage_context = "\n".join(
            [f"- {passage[:200]}" for passage in retrieved_passages[:3]]
        )

        prompt = f"""You are an activation advocate with metabolic knowledge.

Argue why reaction {reaction.get('reaction_id')} should be ACTIVE in CRC.

REACTION
Name: {reaction.get('name', 'Unknown')}
Subsystem: {reaction.get('subsystem', 'Unknown')}

METABOLIC KNOWLEDGE
{passage_context if passage_context else 'No direct evidence found.'}

PREDICTION
MetaGNN Score: {reaction.get('s_r', 0.5):.4f}

Provide your argument with knowledge support:"""

        response = self.llm.generate(prompt)
        return response

    def _advocate_deactivate(
        self, reaction: pd.Series, retrieved_passages: List[str]
    ) -> str:
        """Generate deactivation advocate argument with FAISS context.

        Args:
            reaction: Reaction data.
            retrieved_passages: Semantically similar passages from knowledge base.

        Returns:
            Advocate's argument for inactivity.
        """
        passage_context = "\n".join(
            [f"- {passage[:200]}" for passage in retrieved_passages[:3]]
        )

        prompt = f"""You are a deactivation advocate with metabolic knowledge.

Argue why reaction {reaction.get('reaction_id')} should be INACTIVE in CRC.

REACTION
Name: {reaction.get('name', 'Unknown')}
Subsystem: {reaction.get('subsystem', 'Unknown')}

METABOLIC KNOWLEDGE
{passage_context if passage_context else 'No direct evidence found.'}

PREDICTION
MetaGNN Score: {reaction.get('s_r', 0.5):.4f}

Provide your argument with knowledge support:"""

        response = self.llm.generate(prompt)
        return response

    def _resolve_debate(
        self,
        reaction: pd.Series,
        activate_arg: str,
        deactivate_arg: str,
        retrieved_passages: List[str],
    ) -> Dict[str, Any]:
        """Synthesize arguments with FAISS evidence context.

        Args:
            reaction: Reaction data.
            activate_arg: Activation advocate's argument.
            deactivate_arg: Deactivation advocate's argument.
            retrieved_passages: Retrieved knowledge base passages.

        Returns:
            Resolver's decision.
        """
        passage_summary = f"{len(retrieved_passages)} related passages retrieved"

        prompt = f"""Resolve debate on reaction {reaction.get('reaction_id')} activity.

METABOLIC KNOWLEDGE ({passage_summary})
{chr(10).join([f'- {p[:150]}' for p in retrieved_passages[:3]])}

ACTIVATION ADVOCATE:
{activate_arg}

DEACTIVATION ADVOCATE:
{deactivate_arg}

Synthesize both arguments with knowledge base evidence. Respond with JSON:
{{
    "plausibility_score": <float 0-1>,
    "reasoning": "<synthesis with knowledge evidence>",
    "bias_check": "<identified biases>",
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
        """Validate reactions with FAISS semantic retrieval.

        Args:
            reactions: DataFrame of boundary reactions.

        Returns:
            DataFrame with validation results.
        """
        results = []

        for idx, reaction in reactions.iterrows():
            try:
                # Retrieve semantically similar passages
                query = f"{reaction.get('name', '')} {reaction.get('subsystem', '')}"
                retrieved_passages = self._retrieve_passages(query)

                # Debate stages
                activate_arg = self._advocate_activate(reaction, retrieved_passages)
                deactivate_arg = self._advocate_deactivate(reaction, retrieved_passages)
                decision = self._resolve_debate(
                    reaction, activate_arg, deactivate_arg, retrieved_passages
                )

                results.append({
                    "reaction_id": reaction.get("reaction_id", f"rxn_{idx}"),
                    "s_r": reaction.get("s_r", 0.5),
                    "sigma_r": reaction.get("sigma_r", 0.0),
                    "predicted_state": reaction.get("predicted_state", 0),
                    "p_r": decision["plausibility_score"],
                    "reasoning": decision["reasoning"],
                    "bias_check": decision.get("bias_check", ""),
                    "suggested_action": decision["suggested_action"],
                    "faiss_passages_count": len(retrieved_passages),
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
                    "faiss_passages_count": 0,
                })

        return pd.DataFrame(results)

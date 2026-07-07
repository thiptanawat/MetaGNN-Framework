"""
LLM interface and serving client.

Wraps vLLM client for local model deployment.
Supports multiple backends (Qwen3, Gemma, OpenBioLLM, etc.).
Handles structured JSON output parsing and retries.
"""

import json
import logging
import time
from typing import Any, Dict, Optional

try:
    from openai import OpenAI, APIError, APIConnectionError
except ImportError:
    OpenAI = None
    APIError = Exception
    APIConnectionError = Exception


logger = logging.getLogger(__name__)


class LLMClient:
    """Client for vLLM-served language models."""

    def __init__(
        self,
        endpoint: str = "http://localhost:8000/v1",
        model: str = "Qwen3-32B-AWQ",
        temperature: float = 0.3,
        max_tokens: int = 1024,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        """Initialize LLM client.

        Args:
            endpoint: vLLM API endpoint URL.
            model: Model identifier.
            temperature: Sampling temperature (0.0-1.0).
            max_tokens: Maximum response length.
            timeout: Request timeout in seconds.
            max_retries: Maximum retry attempts.

        Raises:
            ImportError: If openai package not installed.
        """
        if OpenAI is None:
            raise ImportError(
                "openai package required. Install with: pip install openai"
            )

        self.endpoint = endpoint
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries

        self.client = OpenAI(
            api_key="not-needed",  # vLLM doesn't require auth
            base_url=endpoint,
            timeout=timeout,
        )

        logger.info(
            f"Initialized LLM client: {model} at {endpoint} "
            f"(temp={temperature}, max_tokens={max_tokens})"
        )

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate response from LLM with retry logic.

        Args:
            prompt: Input prompt text.
            temperature: Optional override for sampling temperature.
            max_tokens: Optional override for max tokens.

        Returns:
            Generated text response.

        Raises:
            RuntimeError: If max retries exceeded.
        """
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        for attempt in range(self.max_retries):
            try:
                response = self.client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    temperature=temp,
                    max_tokens=tokens,
                    top_p=0.95,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                )

                return response.choices[0].text.strip()

            except (APIConnectionError, APIError) as e:
                wait_time = 2 ** attempt
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"LLM request failed (attempt {attempt + 1}/"
                        f"{self.max_retries}): {e}. Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(
                        f"LLM request failed after {self.max_retries} attempts: {e}"
                    ) from e

    def parse_json_response(self, response: str) -> Dict[str, Any]:
        """Extract and parse JSON from LLM response.

        Args:
            response: Raw LLM response text.

        Returns:
            Parsed JSON dictionary.

        Raises:
            ValueError: If JSON extraction/parsing fails.
        """
        try:
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx < 0 or end_idx <= start_idx:
                raise ValueError("No JSON object found in response")

            json_str = response[start_idx:end_idx]
            return json.loads(json_str)

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {e}") from e

    def health_check(self) -> bool:
        """Check if LLM service is healthy.

        Returns:
            True if service is responsive, False otherwise.
        """
        try:
            response = self.client.completions.create(
                model=self.model,
                prompt="Hello",
                max_tokens=1,
                timeout=5,
            )
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

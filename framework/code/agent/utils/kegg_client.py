"""
KEGG REST API wrapper with caching and rate limiting.

Fetches enzyme classifications and related reactions.
Implements caching to reduce API calls.
Includes error handling for timeouts and network issues.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import requests
except ImportError:
    requests = None


logger = logging.getLogger(__name__)


class KEGGClient:
    """Client for KEGG REST API with local caching."""

    BASE_URL = "http://rest.kegg.jp"
    RATE_LIMIT_DELAY = 1.0  # seconds between requests
    REQUEST_TIMEOUT = 10

    def __init__(self, cache_dir: str = "./cache/kegg"):
        """Initialize KEGG client with optional caching.

        Args:
            cache_dir: Directory for caching API responses.

        Raises:
            ImportError: If requests package not installed.
        """
        if requests is None:
            raise ImportError(
                "requests package required. Install with: pip install requests"
            )

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.last_request_time = 0

        logger.info(f"Initialized KEGG client with cache at {self.cache_dir}")

    def _respect_rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.RATE_LIMIT_DELAY:
            time.sleep(self.RATE_LIMIT_DELAY - elapsed)
        self.last_request_time = time.time()

    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for a key.

        Args:
            key: Cache key (e.g., 'enzyme_class_R00001').

        Returns:
            Path to cache file.
        """
        return self.cache_dir / f"{key}.json"

    def _load_cache(self, key: str) -> Optional[Any]:
        """Load value from cache.

        Args:
            key: Cache key.

        Returns:
            Cached value or None if not found.
        """
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.debug(f"Cache read failed for {key}: {e}")
        return None

    def _save_cache(self, key: str, value: Any):
        """Save value to cache.

        Args:
            key: Cache key.
            value: Value to cache.
        """
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, 'w') as f:
                json.dump(value, f)
        except IOError as e:
            logger.debug(f"Cache write failed for {key}: {e}")

    def get_enzyme_class(self, reaction_id: str) -> Optional[str]:
        """Fetch enzyme classification for reaction.

        Args:
            reaction_id: KEGG reaction ID (e.g., 'R00001').

        Returns:
            Enzyme classification string or None.
        """
        cache_key = f"enzyme_class_{reaction_id}"
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        try:
            self._respect_rate_limit()
            url = f"{self.BASE_URL}/get/{reaction_id}"
            response = requests.get(url, timeout=self.REQUEST_TIMEOUT)
            response.raise_for_status()

            # Parse response to extract enzyme info
            lines = response.text.split('\n')
            enzyme_class = None
            for line in lines:
                if line.startswith('ENZYME'):
                    parts = line.split()
                    if len(parts) > 1:
                        enzyme_class = parts[1]
                        break

            self._save_cache(cache_key, enzyme_class)
            return enzyme_class

        except Exception as e:
            logger.debug(f"Enzyme class lookup failed for {reaction_id}: {e}")
            return None

    def get_related_reactions(
        self, reaction_id: str, limit: int = 10
    ) -> List[str]:
        """Fetch related reactions (by shared substrates/products).

        Args:
            reaction_id: KEGG reaction ID.
            limit: Maximum number of related reactions to return.

        Returns:
            List of related reaction IDs.
        """
        cache_key = f"related_reactions_{reaction_id}_{limit}"
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        try:
            self._respect_rate_limit()
            url = f"{self.BASE_URL}/link/reaction/{reaction_id}"
            response = requests.get(url, timeout=self.REQUEST_TIMEOUT)
            response.raise_for_status()

            # Parse tab-delimited response
            related = []
            for line in response.text.strip().split('\n'):
                if line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        related_id = parts[1].replace('rn:', '')
                        related.append(related_id)
                        if len(related) >= limit:
                            break

            self._save_cache(cache_key, related)
            return related

        except Exception as e:
            logger.debug(f"Related reactions lookup failed for {reaction_id}: {e}")
            return []

    def clear_cache(self):
        """Clear all cached data."""
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
            logger.info("Cleared KEGG cache")
        except Exception as e:
            logger.warning(f"Cache clear failed: {e}")

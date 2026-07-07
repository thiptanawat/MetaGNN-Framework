"""
FAISS semantic search index for metabolic knowledge base.

Builds index using sentence-transformers/all-MiniLM-L6-v2 (384-dim).
Supports top-k retrieval by cosine similarity.
Includes index persistence and rebuild capabilities.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import faiss
    import numpy as np
except ImportError:
    faiss = None
    np = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


logger = logging.getLogger(__name__)


class FAISSIndex:
    """FAISS-based semantic search for metabolic knowledge."""

    def __init__(
        self,
        index_path: str = "./cache/faiss",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """Initialize FAISS index.

        Args:
            index_path: Directory for index persistence.
            model_name: Sentence-transformers model identifier.

        Raises:
            ImportError: If faiss or sentence_transformers not installed.
        """
        if faiss is None or SentenceTransformer is None:
            raise ImportError(
                "faiss and sentence_transformers required. "
                "Install with: pip install faiss-cpu sentence-transformers"
            )

        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

        # Load embedding model
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        # Load or initialize index
        self._load_index()

        logger.info(
            f"Initialized FAISS index: {self.embedding_dim}-dim, "
            f"{self.index.ntotal} passages"
        )

    def _load_index(self):
        """Load FAISS index from disk or create new."""
        index_file = self.index_path / "metabolic_kb.index"
        metadata_file = self.index_path / "metabolic_kb_metadata.json"

        if index_file.exists() and metadata_file.exists():
            try:
                self.index = faiss.read_index(str(index_file))
                with open(metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded FAISS index with {self.index.ntotal} passages")
            except Exception as e:
                logger.warning(f"Failed to load index: {e}. Creating new index.")
                self._create_new_index()
        else:
            self._create_new_index()

    def _create_new_index(self):
        """Create new empty FAISS index."""
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.metadata = {"passages": [], "passage_ids": []}
        logger.info("Created new FAISS index")

    def add_passages(self, passages: List[str], passage_ids: Optional[List[str]] = None):
        """Add passages to index.

        Args:
            passages: List of text passages.
            passage_ids: Optional list of passage identifiers.
        """
        if passage_ids is None:
            passage_ids = [f"passage_{i}" for i in range(len(passages))]

        try:
            # Encode passages
            embeddings = self.model.encode(passages, show_progress_bar=True)
            embeddings = np.array(embeddings, dtype=np.float32)

            # Add to index
            self.index.add(embeddings)

            # Update metadata
            self.metadata["passages"].extend(passages)
            self.metadata["passage_ids"].extend(passage_ids)

            logger.info(f"Added {len(passages)} passages to index")

        except Exception as e:
            logger.error(f"Error adding passages: {e}")
            raise

    def search(self, query: str, top_k: int = 5) -> List[str]:
        """Search for similar passages.

        Args:
            query: Query string.
            top_k: Number of results to return.

        Returns:
            List of top-k similar passages.
        """
        try:
            # Encode query
            query_embedding = self.model.encode([query])[0]
            query_embedding = np.array([query_embedding], dtype=np.float32)

            # Search
            _, indices = self.index.search(query_embedding, top_k)

            results = []
            for idx in indices[0]:
                if idx >= 0 and idx < len(self.metadata["passages"]):
                    results.append(self.metadata["passages"][idx])

            return results

        except Exception as e:
            logger.warning(f"Search failed: {e}")
            return []

    def search_with_scores(
        self, query: str, top_k: int = 5
    ) -> Tuple[List[str], List[float]]:
        """Search with similarity scores.

        Args:
            query: Query string.
            top_k: Number of results to return.

        Returns:
            Tuple of (passages, scores) where scores are in [0, 1].
        """
        try:
            # Encode query
            query_embedding = self.model.encode([query])[0]
            query_embedding = np.array([query_embedding], dtype=np.float32)

            # Normalize query for cosine similarity
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)

            # Get all distances
            distances, indices = self.index.search(query_embedding, top_k)

            # Convert L2 distances to cosine similarity scores
            passages = []
            scores = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx >= 0 and idx < len(self.metadata["passages"]):
                    # L2 distance to cosine similarity
                    similarity = 1.0 / (1.0 + dist)
                    passages.append(self.metadata["passages"][idx])
                    scores.append(similarity)

            return passages, scores

        except Exception as e:
            logger.warning(f"Search with scores failed: {e}")
            return [], []

    def save(self):
        """Persist index to disk."""
        try:
            index_file = self.index_path / "metabolic_kb.index"
            metadata_file = self.index_path / "metabolic_kb_metadata.json"

            faiss.write_index(self.index, str(index_file))
            with open(metadata_file, 'w') as f:
                json.dump(self.metadata, f)

            logger.info(f"Saved FAISS index with {self.index.ntotal} passages")

        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise

    def get_stats(self) -> dict:
        """Get index statistics.

        Returns:
            Dictionary with index metrics.
        """
        return {
            "total_passages": self.index.ntotal,
            "embedding_dim": self.embedding_dim,
            "model": self.model_name,
            "index_path": str(self.index_path),
        }

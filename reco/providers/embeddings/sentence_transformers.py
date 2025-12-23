"""Sentence Transformers embedding provider."""

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer


class SentenceTransformerProvider:
    """Local embedding provider using sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with a sentence-transformers model.
        
        Args:
            model_name: HuggingFace model name. Default is a small, fast model.
                       Other options: 'all-mpnet-base-v2' (better quality, slower)
        """
        self.model = SentenceTransformer(model_name)
        self._dimension = self.model.get_sentence_embedding_dimension()
    
    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of strings to embed.
            
        Returns:
            numpy array of shape (len(texts), dimension)
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self._dimension)
        
        embeddings = self.model.encode(
            texts, 
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embeddings.astype(np.float32)
    
    @property
    def dimension(self) -> int:
        """Dimension of the embedding vectors."""
        return self._dimension

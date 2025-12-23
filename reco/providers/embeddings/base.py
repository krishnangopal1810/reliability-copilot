"""Base embedding provider protocol."""

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""
    
    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        """Generate embeddings for a list of texts."""
        ...
    
    @property
    def dimension(self) -> int:
        """Dimension of the embedding vectors."""
        ...

"""Storage protocol for persistence."""

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, Optional
from uuid import UUID

import numpy as np
from numpy.typing import NDArray

from ..core.models import EvalRun, FailureCluster


@dataclass
class ClusterMatch:
    """A historical cluster that matches a current one."""
    cluster_id: UUID
    label: str
    similarity: float  # 0-1 cosine similarity
    run_id: UUID
    run_date: datetime
    occurrence_count: int
    severity: Optional["Severity"] = None  # For label reuse


class Storage(Protocol):
    """Protocol for storage backends."""
    
    def save_run(self, run: EvalRun) -> None:
        """Persist an eval run."""
        ...
    
    def save_clusters(
        self, 
        clusters: list[FailureCluster], 
        run_id: UUID,
        embeddings: list[NDArray[np.float32]]
    ) -> None:
        """Save clusters with their embeddings for similarity matching."""
        ...
    
    def find_similar_clusters(
        self, 
        embedding: NDArray[np.float32], 
        threshold: float = 0.85,
        limit: int = 10
    ) -> list[ClusterMatch]:
        """Find historically similar clusters by embedding similarity."""
        ...
    
    def count_recent_occurrences(
        self, 
        label: str, 
        last_n_runs: int = 5
    ) -> tuple[int, Optional[datetime]]:
        """Count occurrences of a label in recent runs.
        
        Returns:
            Tuple of (count, first_seen_date)
        """
        ...
    
    # Phase 2: Profile methods
    def get_failure_mode_stats(
        self, 
        limit_runs: int = 10
    ) -> list[tuple[str, int, int]]:
        """Get failure mode stats aggregated from recent runs.
        
        Returns:
            List of (label, occurrence_count, total_failures) tuples
        """
        ...
    
    def get_pass_rate_history(
        self, 
        limit: int = 10
    ) -> list[tuple[datetime, float, int]]:
        """Get pass rate history from recent runs.
        
        Returns:
            List of (date, pass_rate, total_responses) tuples
        """
        ...
    
    def get_recent_runs(
        self,
        limit: int = 5
    ) -> list[EvalRun]:
        """Get recent eval runs for comparison.
        
        Returns:
            List of EvalRun objects, most recent first.
        """
        ...
    
    def close(self) -> None:
        """Close storage connection."""
        ...


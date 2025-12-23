"""Clusterer for grouping related failures."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import hdbscan
import numpy as np

from .models import Response, FailureCluster, Severity

if TYPE_CHECKING:
    from ..providers.llm.base import LLMProvider
    from ..providers.embeddings.base import EmbeddingProvider


@dataclass
class ClusterConfig:
    """Configuration for clustering behavior."""
    min_cluster_size: int = 2   # Minimum failures to form a cluster
    min_samples: int = 1        # Density threshold (lower = more clusters)
    metric: str = "euclidean"   # Distance metric


class Clusterer:
    """Clusters failure responses to identify patterns."""
    
    def __init__(
        self, 
        embedder: "EmbeddingProvider", 
        llm: "LLMProvider",
        config: ClusterConfig | None = None
    ):
        self.embedder = embedder
        self.llm = llm
        self.config = config or ClusterConfig()
    
    def cluster(self, failures: list[Response]) -> list[FailureCluster]:
        """Cluster failures by semantic similarity.
        
        Args:
            failures: List of failed Response objects
            
        Returns:
            List of FailureCluster objects, sorted by severity then size
        """
        if not failures:
            return []
        
        if len(failures) == 1:
            return [self._single_cluster(failures)]
        
        if len(failures) < self.config.min_cluster_size:
            # Not enough to cluster meaningfully
            return [self._create_cluster(-1, failures)]
        
        # 1. Get text to embed (failure reason or output)
        reasons = [f.failure_reason or f.output[:500] for f in failures]
        
        # 2. Generate embeddings
        embeddings = self.embedder.embed(reasons)
        
        # 3. Cluster with HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.config.min_cluster_size,
            min_samples=self.config.min_samples,
            metric=self.config.metric,
        )
        labels = clusterer.fit_predict(embeddings)
        
        # 4. Group failures by cluster label
        clusters: dict[int, list[Response]] = {}
        for failure, label in zip(failures, labels):
            clusters.setdefault(label, []).append(failure)
        
        # 5. Create cluster objects with LLM-generated labels
        result = []
        for label, cluster_failures in sorted(clusters.items()):
            cluster = self._create_cluster(label, cluster_failures)
            result.append(cluster)
        
        # Sort by severity (descending), then by size (descending)
        return sorted(
            result, 
            key=lambda c: (-self._severity_rank(c.severity), -len(c.response_ids))
        )
    
    def _create_cluster(self, label: int, failures: list[Response]) -> FailureCluster:
        """Create a labeled cluster from a group of failures.
        
        Args:
            label: Cluster label from HDBSCAN (-1 for noise/uncategorized)
            failures: List of failures in this cluster
            
        Returns:
            FailureCluster with LLM-generated label
        """
        response_ids = [f.id for f in failures]
        
        if label == -1:
            return FailureCluster(
                label="Uncategorized",
                description="Failures that don't fit a clear pattern",
                severity=Severity.LOW,
                response_ids=response_ids,
            )
        
        # Generate label via LLM
        reasons = [f.failure_reason or f.output[:200] for f in failures[:5]]
        label_text, severity = self._generate_label(reasons)
        
        return FailureCluster(
            label=label_text,
            description=f"{len(failures)} failures with similar pattern",
            severity=severity,
            response_ids=response_ids,
        )
    
    def _generate_label(self, reasons: list[str]) -> tuple[str, Severity]:
        """Use LLM to generate a cluster label and assess severity.
        
        Args:
            reasons: Sample of failure reasons from the cluster
            
        Returns:
            Tuple of (label string, Severity enum)
        """
        prompt = f"""These are failure reasons from an AI system that are semantically related.
They represent a pattern of failures that keep occurring.

Failure examples:
{chr(10).join(f"- {r[:200]}" for r in reasons)}

Tasks:
1. Generate a short, descriptive label (3-5 words) that names this failure pattern.
2. Assess severity:
   - LOW: Minor issues, cosmetic problems, edge cases
   - MEDIUM: Noticeable problems that affect user experience
   - HIGH: Significant failures, wrong answers, broken functionality
   - CRITICAL: Dangerous, unsafe, or compliance-violating behavior

Format your response exactly as:
LABEL: <your label here>
SEVERITY: <LOW|MEDIUM|HIGH|CRITICAL>"""

        try:
            response = self.llm.complete(prompt, max_tokens=50)
            return self._parse_label_response(response)
        except Exception:
            return "Unknown Pattern", Severity.MEDIUM
    
    def _parse_label_response(self, response: str) -> tuple[str, Severity]:
        """Parse the LLM response for label and severity."""
        label = "Unknown Pattern"
        severity = Severity.MEDIUM
        
        for line in response.strip().split("\n"):
            line = line.strip()
            if line.upper().startswith("LABEL:"):
                label = line.split(":", 1)[1].strip()
            elif line.upper().startswith("SEVERITY:"):
                sev_str = line.split(":", 1)[1].strip().upper()
                if sev_str in Severity.__members__:
                    severity = Severity[sev_str]
        
        return label, severity
    
    def _single_cluster(self, failures: list[Response]) -> FailureCluster:
        """Handle edge case of a single failure."""
        failure = failures[0]
        return FailureCluster(
            label="Single Failure",
            description=failure.failure_reason or "Single failure case",
            severity=Severity.MEDIUM,
            response_ids=[failure.id],
        )
    
    @staticmethod
    def _severity_rank(severity: Severity) -> int:
        """Convert severity to numeric rank for sorting."""
        return {
            Severity.LOW: 0, 
            Severity.MEDIUM: 1, 
            Severity.HIGH: 2, 
            Severity.CRITICAL: 3
        }[severity]

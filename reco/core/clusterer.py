"""Clusterer for grouping related failures."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import hdbscan
import numpy as np
from numpy.typing import NDArray

from .models import Response, FailureCluster, Severity

if TYPE_CHECKING:
    from ..providers.llm.base import LLMProvider
    from ..providers.embeddings.base import EmbeddingProvider
    from ..storage.base import Storage


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
        config: ClusterConfig | None = None,
        storage: Optional["Storage"] = None
    ):
        self.embedder = embedder
        self.llm = llm
        self.config = config or ClusterConfig()
        self.storage = storage
    
    def cluster(self, failures: list[Response], run_id=None) -> list[FailureCluster]:
        """Cluster failures by semantic similarity.
        
        Args:
            failures: List of failed Response objects
            run_id: Optional run ID for storage
            
        Returns:
            List of FailureCluster objects, sorted by severity then size
        """
        if not failures:
            return []
        
        if len(failures) == 1:
            clusters = [self._single_cluster(failures)]
            return clusters
        
        if len(failures) < self.config.min_cluster_size:
            # Not enough to cluster meaningfully
            return [self._create_cluster(-1, failures, None)]
        
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
        clusters_by_label: dict[int, list[tuple[Response, NDArray]]] = {}
        for failure, label, emb in zip(failures, labels, embeddings):
            clusters_by_label.setdefault(label, []).append((failure, emb))
        
        # 5. Create cluster objects with LLM-generated labels
        result = []
        cluster_embeddings = []
        
        for label, items in sorted(clusters_by_label.items()):
            cluster_failures = [f for f, _ in items]
            # Average embedding for the cluster
            cluster_emb = np.mean([e for _, e in items], axis=0).astype(np.float32)
            
            cluster = self._create_cluster(label, cluster_failures, cluster_emb)
            result.append(cluster)
            cluster_embeddings.append(cluster_emb)
        
        # 6. Add history context if storage available
        if self.storage:
            self._add_history_context(result, cluster_embeddings)
            
            # Save to storage
            if run_id:
                self.storage.save_clusters(result, run_id, cluster_embeddings)
        
        # Sort by severity (descending), then by size (descending)
        return sorted(
            result, 
            key=lambda c: (-self._severity_rank(c.severity), -len(c.response_ids))
        )
    
    def _create_cluster(
        self, 
        label: int, 
        failures: list[Response], 
        embedding: Optional[NDArray] = None
    ) -> FailureCluster:
        """Create a labeled cluster from a group of failures.
        
        Args:
            label: Cluster label from HDBSCAN (-1 for noise/uncategorized)
            failures: List of failures in this cluster
            embedding: Optional cluster embedding for history matching
            
        Returns:
            FailureCluster with LLM-generated label or reused historical label
        """
        response_ids = [f.id for f in failures]
        
        if label == -1:
            return FailureCluster(
                label="Uncategorized",
                description="Failures that don't fit a clear pattern",
                severity=Severity.LOW,
                response_ids=response_ids,
            )
        
        # Check for existing label from historical matches (embedding similarity)
        if self.storage and embedding is not None:
            matches = self.storage.find_similar_clusters(embedding, threshold=0.85)
            if matches:
                # Reuse existing label for consistency
                best_match = matches[0]
                return FailureCluster(
                    label=best_match.label,
                    description=f"{len(failures)} failures matching known pattern",
                    severity=best_match.severity,
                    response_ids=response_ids,
                    is_recurring=True,
                )
        
        # No match found - generate new label via LLM
        reasons = [f.failure_reason or f.output[:200] for f in failures[:5]]
        label_text, severity = self._generate_label(reasons)
        
        return FailureCluster(
            label=label_text,
            description=f"{len(failures)} failures with similar pattern",
            severity=severity,
            response_ids=response_ids,
        )
    
    def _generate_label(self, reasons: list[str]) -> tuple[str, Severity]:
        """Classify failures into taxonomy category.
        
        Uses predefined taxonomy for consistent labeling. Falls back to
        LLM-generated label only if no category fits.
        
        Args:
            reasons: Sample of failure reasons from the cluster
            
        Returns:
            Tuple of (category name, Severity enum)
        """
        from .taxonomy import load_taxonomy, format_taxonomy_for_prompt
        
        categories = load_taxonomy()
        taxonomy_text = format_taxonomy_for_prompt(categories)
        
        prompt = f"""Classify these AI system failures into ONE category from the taxonomy.

TAXONOMY:
{taxonomy_text}

FAILURES:
{chr(10).join(f"- {r[:200]}" for r in reasons)}

RULES:
1. Pick the BEST matching category from the taxonomy
2. If none fit well, respond with: OTHER: [brief 2-4 word label]
3. Assess severity:
   - LOW: Minor issues, cosmetic, edge cases
   - MEDIUM: Noticeable problems affecting user experience
   - HIGH: Significant failures, wrong answers
   - CRITICAL: Dangerous, unsafe, or compliance-violating

RESPONSE FORMAT:
CATEGORY: [exact category name from taxonomy OR "OTHER: your label"]
SEVERITY: [LOW|MEDIUM|HIGH|CRITICAL]"""

        try:
            response = self.llm.complete(prompt, max_tokens=50)
            return self._parse_label_response(response, categories)
        except Exception:
            return "Unknown Pattern", Severity.MEDIUM
    
    def _parse_label_response(
        self, 
        response: str, 
        categories: list
    ) -> tuple[str, Severity]:
        """Parse the LLM classification response.
        
        Validates that the returned category exists in the taxonomy.
        """
        label = "Unknown Pattern"
        severity = Severity.MEDIUM
        
        category_names = {cat.name.lower(): cat.name for cat in categories}
        
        for line in response.strip().split("\n"):
            line = line.strip()
            if line.upper().startswith("CATEGORY:"):
                raw_label = line.split(":", 1)[1].strip()
                
                # Check if it's an "OTHER" response
                if raw_label.upper().startswith("OTHER:"):
                    label = raw_label.split(":", 1)[1].strip()
                else:
                    # Validate against taxonomy (case-insensitive)
                    normalized = raw_label.lower()
                    if normalized in category_names:
                        label = category_names[normalized]
                    else:
                        # Use as-is if not found (LLM might have used exact match)
                        label = raw_label
                        
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
    
    def _add_history_context(
        self, 
        clusters: list[FailureCluster], 
        embeddings: list[NDArray]
    ) -> None:
        """Add historical context to clusters by checking for similar patterns."""
        if not self.storage:
            return
        
        for cluster, embedding in zip(clusters, embeddings):
            if cluster.label == "Uncategorized":
                continue
            
            # Check for similar historical clusters by embedding
            matches = self.storage.find_similar_clusters(embedding, threshold=0.85)
            
            # Also check by exact label match
            count, first_seen = self.storage.count_recent_occurrences(
                cluster.label, last_n_runs=5
            )
            
            if matches or count > 0:
                cluster.is_recurring = True
                cluster.occurrence_count = count + 1  # Including current
                cluster.first_seen = first_seen

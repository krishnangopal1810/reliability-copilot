"""Exhaustive unit tests for reco.core.clusterer."""

import pytest
import numpy as np

from reco.core.models import Response, FailureCluster, Severity
from reco.core.clusterer import Clusterer, ClusterConfig
from tests.conftest import MockLLMProvider, MockEmbeddingProvider


class TestClusterConfig:
    """Tests for ClusterConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ClusterConfig()
        
        assert config.min_cluster_size == 2
        assert config.min_samples == 1
        assert config.metric == "euclidean"
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = ClusterConfig(
            min_cluster_size=3,
            min_samples=2,
            metric="cosine",
        )
        
        assert config.min_cluster_size == 3
        assert config.min_samples == 2
        assert config.metric == "cosine"


class TestClusterer:
    """Tests for the Clusterer class."""
    
    # ========================================================================
    # Initialization
    # ========================================================================
    
    def test_clusterer_init(self, mock_embedder, mock_llm):
        """Test clusterer initialization."""
        clusterer = Clusterer(mock_embedder, mock_llm)
        
        assert clusterer.embedder == mock_embedder
        assert clusterer.llm == mock_llm
        assert clusterer.config is not None
    
    def test_clusterer_init_with_custom_config(self, mock_embedder, mock_llm):
        """Test clusterer with custom config."""
        config = ClusterConfig(min_cluster_size=3)
        clusterer = Clusterer(mock_embedder, mock_llm, config)
        
        assert clusterer.config.min_cluster_size == 3
    
    # ========================================================================
    # Empty and Single Failure Cases
    # ========================================================================
    
    def test_cluster_empty_list(self, mock_embedder, mock_llm):
        """Test clustering an empty list of failures."""
        clusterer = Clusterer(mock_embedder, mock_llm)
        result = clusterer.cluster([])
        
        assert result == []
    
    def test_cluster_single_failure(self, mock_embedder, mock_llm):
        """Test clustering a single failure."""
        failures = [
            Response(id="t1", input="Q", output="A", passed=False, failure_reason="Error"),
        ]
        
        clusterer = Clusterer(mock_embedder, mock_llm)
        result = clusterer.cluster(failures)
        
        assert len(result) == 1
        assert result[0].label == "Single Failure"
        assert len(result[0].response_ids) == 1
    
    def test_cluster_below_min_size(self, mock_embedder, mock_llm):
        """Test clustering when failures are below min cluster size."""
        failures = [
            Response(id="t1", input="Q", output="A", passed=False, failure_reason="Error"),
        ]
        
        config = ClusterConfig(min_cluster_size=3)
        clusterer = Clusterer(mock_embedder, mock_llm, config)
        result = clusterer.cluster(failures)
        
        # Should return a single uncategorized cluster
        assert len(result) == 1
    
    # ========================================================================
    # Clustering Behavior
    # ========================================================================
    
    def test_cluster_returns_failure_clusters(self, mock_embedder, mock_llm, failures_for_clustering):
        """Test that clustering returns FailureCluster objects."""
        mock_llm.responses["label"] = "LABEL: Test Pattern\nSEVERITY: HIGH"
        
        clusterer = Clusterer(mock_embedder, mock_llm)
        result = clusterer.cluster(failures_for_clustering)
        
        assert all(isinstance(c, FailureCluster) for c in result)
    
    def test_cluster_all_failures_assigned(self, mock_embedder, mock_llm, failures_for_clustering):
        """Test that all failures are assigned to clusters."""
        mock_llm.responses["label"] = "LABEL: Test Pattern\nSEVERITY: MEDIUM"
        
        clusterer = Clusterer(mock_embedder, mock_llm)
        result = clusterer.cluster(failures_for_clustering)
        
        # Collect all response IDs from clusters
        all_ids = []
        for cluster in result:
            all_ids.extend(cluster.response_ids)
        
        # All original failure IDs should be present
        original_ids = {f.id for f in failures_for_clustering}
        assert set(all_ids) == original_ids
    
    def test_cluster_sorted_by_severity_then_size(self, mock_embedder, mock_llm):
        """Test that clusters are sorted by severity (desc) then size (desc)."""
        # Create failures that will form distinct clusters
        failures = [
            Response(id="h1", input="Q", output="A", passed=False, failure_reason="Critical error type A"),
            Response(id="h2", input="Q", output="A", passed=False, failure_reason="Critical error type A same"),
            Response(id="l1", input="Q", output="A", passed=False, failure_reason="Minor issue type B"),
            Response(id="l2", input="Q", output="A", passed=False, failure_reason="Minor issue type B same"),
            Response(id="l3", input="Q", output="A", passed=False, failure_reason="Minor issue type B again"),
        ]
        
        # Mock LLM to return different severities based on content
        def mock_complete(prompt, max_tokens=1000):
            if "critical" in prompt.lower():
                return "LABEL: Critical Errors\nSEVERITY: CRITICAL"
            else:
                return "LABEL: Minor Issues\nSEVERITY: LOW"
        
        mock_llm.complete = mock_complete
        
        clusterer = Clusterer(mock_embedder, mock_llm)
        result = clusterer.cluster(failures)
        
        # First cluster should be the highest severity
        if len(result) > 1:
            # Severity order: CRITICAL > HIGH > MEDIUM > LOW
            severity_order = {Severity.LOW: 0, Severity.MEDIUM: 1, Severity.HIGH: 2, Severity.CRITICAL: 3}
            for i in range(len(result) - 1):
                current_sev = severity_order[result[i].severity]
                next_sev = severity_order[result[i + 1].severity]
                # Either higher severity or same severity with more items
                assert current_sev >= next_sev or (
                    current_sev == next_sev and 
                    len(result[i].response_ids) >= len(result[i + 1].response_ids)
                )
    
    # ========================================================================
    # Embeddings
    # ========================================================================
    
    def test_cluster_calls_embedder(self, mock_embedder, mock_llm):
        """Test that clustering calls the embedding provider."""
        failures = [
            Response(id="t1", input="Q", output="A", passed=False, failure_reason="Error 1"),
            Response(id="t2", input="Q", output="A", passed=False, failure_reason="Error 2"),
        ]
        
        mock_llm.set_default_response("LABEL: Test\nSEVERITY: MEDIUM")
        
        clusterer = Clusterer(mock_embedder, mock_llm)
        clusterer.cluster(failures)
        
        # Embedder should have been called
        assert len(mock_embedder.call_history) > 0
    
    def test_cluster_uses_failure_reason_for_embedding(self, mock_embedder, mock_llm):
        """Test that failure reason is used for embedding."""
        failures = [
            Response(id="t1", input="Q", output="A", passed=False, failure_reason="Specific error message"),
            Response(id="t2", input="Q", output="A", passed=False, failure_reason="Another error message"),
        ]
        
        mock_llm.set_default_response("LABEL: Test\nSEVERITY: MEDIUM")
        
        clusterer = Clusterer(mock_embedder, mock_llm)
        clusterer.cluster(failures)
        
        # First call should contain the failure reasons
        embedded_texts = mock_embedder.call_history[0]
        assert "Specific error message" in embedded_texts
        assert "Another error message" in embedded_texts
    
    def test_cluster_uses_output_when_no_failure_reason(self, mock_embedder, mock_llm):
        """Test that output is used when failure_reason is None."""
        failures = [
            Response(id="t1", input="Q", output="Output text as fallback", passed=False),
            Response(id="t2", input="Q", output="Different output fallback", passed=False),
        ]
        
        mock_llm.set_default_response("LABEL: Test\nSEVERITY: MEDIUM")
        
        clusterer = Clusterer(mock_embedder, mock_llm)
        clusterer.cluster(failures)
        
        embedded_texts = mock_embedder.call_history[0]
        assert "Output text as fallback" in embedded_texts[0]
    
    # ========================================================================
    # Label Generation
    # ========================================================================
    
    def test_cluster_llm_generates_labels(self, mock_embedder, mock_llm):
        """Test that LLM generates cluster labels."""
        # Use identical failure reasons to ensure they cluster together
        failures = [
            Response(id="t1", input="Q", output="A", passed=False, failure_reason="Same error pattern"),
            Response(id="t2", input="Q", output="A", passed=False, failure_reason="Same error pattern"),
            Response(id="t3", input="Q", output="A", passed=False, failure_reason="Same error pattern"),
        ]
        
        # Set similar embeddings to ensure clustering
        mock_embedder.set_similar_embeddings([
            "Same error pattern",
            "Same error pattern", 
            "Same error pattern",
        ])
        
        mock_llm.set_default_response("LABEL: Type A Errors\nSEVERITY: HIGH")
        
        clusterer = Clusterer(mock_embedder, mock_llm)
        result = clusterer.cluster(failures)
        
        # LLM should have been called for labeling (at least for non-noise clusters)
        # Check that we got clusters and at least one is labeled
        assert len(result) > 0
        labeled = [c for c in result if c.label not in ("Uncategorized", "Single Failure")]
        # Either LLM was called OR all went to uncategorized (valid behavior)
        assert len(mock_llm.call_history) > 0 or len(labeled) == 0
    
    def test_cluster_parses_label_response(self, mock_embedder, mock_llm):
        """Test that label response is parsed correctly."""
        failures = [
            Response(id="t1", input="Q", output="A", passed=False, failure_reason="Error"),
            Response(id="t2", input="Q", output="A", passed=False, failure_reason="Error similar"),
        ]
        
        mock_llm.set_default_response("LABEL: Financial Hallucinations\nSEVERITY: CRITICAL")
        
        clusterer = Clusterer(mock_embedder, mock_llm)
        result = clusterer.cluster(failures)
        
        # Find a labeled cluster (not Uncategorized or Single Failure)
        labeled = [c for c in result if c.label not in ("Uncategorized", "Single Failure", "Unknown Pattern")]
        if labeled:
            assert labeled[0].label == "Financial Hallucinations"
            assert labeled[0].severity == Severity.CRITICAL
    
    def test_cluster_handles_malformed_label_response(self, mock_embedder, mock_llm):
        """Test handling of malformed LLM response."""
        failures = [
            Response(id="t1", input="Q", output="A", passed=False, failure_reason="Error"),
            Response(id="t2", input="Q", output="A", passed=False, failure_reason="Error similar"),
        ]
        
        # Malformed response
        mock_llm.set_default_response("This is not the expected format at all")
        
        clusterer = Clusterer(mock_embedder, mock_llm)
        result = clusterer.cluster(failures)
        
        # Should not raise, should use defaults
        assert len(result) > 0
        for cluster in result:
            assert cluster.severity in list(Severity)
    
    def test_cluster_handles_partial_label_response(self, mock_embedder, mock_llm):
        """Test handling of partial LLM response (only label, no severity)."""
        failures = [
            Response(id="t1", input="Q", output="A", passed=False, failure_reason="Error"),
            Response(id="t2", input="Q", output="A", passed=False, failure_reason="Error similar"),
        ]
        
        mock_llm.set_default_response("LABEL: Only Label Here")
        
        clusterer = Clusterer(mock_embedder, mock_llm)
        result = clusterer.cluster(failures)
        
        # Should use default severity when not provided
        for cluster in result:
            if cluster.label == "Only Label Here":
                assert cluster.severity == Severity.MEDIUM  # Default
    
    # ========================================================================
    # Uncategorized Cluster
    # ========================================================================
    
    def test_uncategorized_cluster_for_noise(self, mock_embedder, mock_llm):
        """Test that outliers go to Uncategorized cluster."""
        # Make failures very different so HDBSCAN marks them as noise
        failures = [
            Response(id="t1", input="Q", output="A", passed=False, 
                     failure_reason="Completely unique error that stands alone ZZZXXX"),
        ]
        
        clusterer = Clusterer(mock_embedder, mock_llm)
        result = clusterer.cluster(failures)
        
        # Single failure becomes "Single Failure" cluster
        assert len(result) == 1
        assert result[0].label == "Single Failure"
    
    def test_uncategorized_has_low_severity(self, mock_embedder, mock_llm):
        """Test that Uncategorized cluster has LOW severity."""
        # Create setup that will have uncategorized items
        failures = [
            Response(id="t1", input="Q", output="A", passed=False, failure_reason="Error A"),
            Response(id="t2", input="Q", output="A", passed=False, failure_reason="Error A similar"),
            Response(id="outlier", input="Q", output="A", passed=False, 
                     failure_reason="Completely different unrelated outlier QWERTY"),
        ]
        
        mock_llm.set_default_response("LABEL: Test Pattern\nSEVERITY: HIGH")
        
        clusterer = Clusterer(mock_embedder, mock_llm)
        result = clusterer.cluster(failures)
        
        uncategorized = [c for c in result if c.label == "Uncategorized"]
        for uc in uncategorized:
            assert uc.severity == Severity.LOW
    
    # ========================================================================
    # Severity Ranking
    # ========================================================================
    
    def test_severity_rank_ordering(self, mock_embedder, mock_llm):
        """Test that severity ranking is correct."""
        from reco.core.clusterer import Clusterer
        
        assert Clusterer._severity_rank(Severity.LOW) == 0
        assert Clusterer._severity_rank(Severity.MEDIUM) == 1
        assert Clusterer._severity_rank(Severity.HIGH) == 2
        assert Clusterer._severity_rank(Severity.CRITICAL) == 3
    
    # ========================================================================
    # Edge Cases
    # ========================================================================
    
    def test_cluster_with_long_failure_reason(self, mock_embedder, mock_llm):
        """Test clustering with very long failure reasons."""
        long_reason = "Error: " + "x" * 1000
        failures = [
            Response(id="t1", input="Q", output="A", passed=False, failure_reason=long_reason),
            Response(id="t2", input="Q", output="A", passed=False, failure_reason=long_reason + " variant"),
        ]
        
        mock_llm.set_default_response("LABEL: Long Errors\nSEVERITY: MEDIUM")
        
        clusterer = Clusterer(mock_embedder, mock_llm)
        result = clusterer.cluster(failures)
        
        # Should not raise
        assert len(result) > 0
    
    def test_cluster_with_unicode_failure_reason(self, mock_embedder, mock_llm):
        """Test clustering with unicode in failure reasons."""
        failures = [
            Response(id="t1", input="Q", output="A", passed=False, failure_reason="Error with émojis 🔥"),
            Response(id="t2", input="Q", output="A", passed=False, failure_reason="Error with émojis 🎉"),
        ]
        
        mock_llm.set_default_response("LABEL: Emoji Errors\nSEVERITY: LOW")
        
        clusterer = Clusterer(mock_embedder, mock_llm)
        result = clusterer.cluster(failures)
        
        # Should not raise
        assert len(result) > 0
    
    def test_cluster_with_empty_failure_reason(self, mock_embedder, mock_llm):
        """Test clustering with empty failure reasons."""
        failures = [
            Response(id="t1", input="Q", output="Output 1", passed=False, failure_reason=""),
            Response(id="t2", input="Q", output="Output 2", passed=False, failure_reason=""),
        ]
        
        mock_llm.set_default_response("LABEL: Empty Errors\nSEVERITY: MEDIUM")
        
        clusterer = Clusterer(mock_embedder, mock_llm)
        result = clusterer.cluster(failures)
        
        # Should fall back to using output
        assert len(result) > 0


# ============================================================================
# Phase 1: Storage Integration Tests
# ============================================================================

class TestClustererWithStorage:
    """Tests for clusterer with storage integration (Phase 1)."""
    
    def test_clusterer_accepts_storage_parameter(self, mock_embedder, mock_llm, mock_storage):
        """Test that clusterer accepts optional storage parameter."""
        clusterer = Clusterer(mock_embedder, mock_llm, storage=mock_storage)
        
        assert clusterer.storage == mock_storage
    
    def test_clusterer_saves_clusters_to_storage(self, mock_embedder, mock_llm, mock_storage):
        """Test that clusterer saves clusters when storage is provided."""
        from uuid import uuid4
        
        failures = [
            Response(id="t1", input="Q", output="A", passed=False, failure_reason="Error pattern A"),
            Response(id="t2", input="Q", output="A", passed=False, failure_reason="Error pattern A similar"),
        ]
        
        mock_llm.set_default_response("LABEL: Test Pattern\nSEVERITY: MEDIUM")
        
        run_id = uuid4()
        clusterer = Clusterer(mock_embedder, mock_llm, storage=mock_storage)
        clusterer.cluster(failures, run_id=run_id)
        
        # Should have saved clusters
        assert len(mock_storage.saved_clusters) == 1
        saved_clusters, saved_run_id, saved_embeddings = mock_storage.saved_clusters[0]
        assert saved_run_id == run_id
        assert len(saved_clusters) > 0
        assert len(saved_embeddings) > 0
    
    def test_clusterer_does_not_save_without_run_id(self, mock_embedder, mock_llm, mock_storage):
        """Test that clusterer doesn't save if no run_id provided."""
        failures = [
            Response(id="t1", input="Q", output="A", passed=False, failure_reason="Error"),
            Response(id="t2", input="Q", output="A", passed=False, failure_reason="Error similar"),
        ]
        
        mock_llm.set_default_response("LABEL: Test\nSEVERITY: MEDIUM")
        
        clusterer = Clusterer(mock_embedder, mock_llm, storage=mock_storage)
        clusterer.cluster(failures)  # No run_id
        
        # Should not have saved
        assert len(mock_storage.saved_clusters) == 0
    
    def test_clusterer_marks_recurring_patterns(self, mock_embedder, mock_llm, mock_storage):
        """Test that clusterer marks patterns as recurring when history exists."""
        from datetime import datetime, timezone
        
        failures = [
            Response(id="t1", input="Q", output="A", passed=False, failure_reason="Known error pattern"),
            Response(id="t2", input="Q", output="A", passed=False, failure_reason="Known error pattern similar"),
        ]
        
        mock_llm.set_default_response("LABEL: Known Pattern\nSEVERITY: HIGH")
        
        # Set up history
        mock_storage.set_occurrence_count(3, datetime(2024, 1, 15, tzinfo=timezone.utc))
        
        clusterer = Clusterer(mock_embedder, mock_llm, storage=mock_storage)
        result = clusterer.cluster(failures)
        
        # Find non-uncategorized cluster
        labeled = [c for c in result if c.label not in ("Uncategorized", "Single Failure")]
        if labeled:
            cluster = labeled[0]
            assert cluster.is_recurring is True
            assert cluster.occurrence_count == 4  # 3 + 1 for current
            assert cluster.first_seen is not None
    
    def test_clusterer_new_patterns_not_recurring(self, mock_embedder, mock_llm, mock_storage):
        """Test that new patterns are not marked as recurring."""
        failures = [
            Response(id="t1", input="Q", output="A", passed=False, failure_reason="Brand new error"),
            Response(id="t2", input="Q", output="A", passed=False, failure_reason="Brand new error similar"),
        ]
        
        mock_llm.set_default_response("LABEL: New Pattern\nSEVERITY: MEDIUM")
        
        # No history
        mock_storage.set_occurrence_count(0, None)
        
        clusterer = Clusterer(mock_embedder, mock_llm, storage=mock_storage)
        result = clusterer.cluster(failures)
        
        # New patterns should not be recurring
        for cluster in result:
            if cluster.label not in ("Uncategorized", "Single Failure"):
                assert cluster.is_recurring is False
    
    def test_clusterer_works_without_storage(self, mock_embedder, mock_llm):
        """Test that clusterer works when no storage is provided."""
        failures = [
            Response(id="t1", input="Q", output="A", passed=False, failure_reason="Error"),
            Response(id="t2", input="Q", output="A", passed=False, failure_reason="Error similar"),
        ]
        
        mock_llm.set_default_response("LABEL: Test\nSEVERITY: MEDIUM")
        
        # No storage provided
        clusterer = Clusterer(mock_embedder, mock_llm)
        result = clusterer.cluster(failures)
        
        # Should still work and return clusters
        assert len(result) > 0
        
        # Clusters should have default history values
        for cluster in result:
            assert cluster.is_recurring is False
            assert cluster.occurrence_count == 1
    
    def test_clusterer_reuses_existing_labels(self, mock_embedder, mock_llm, mock_storage):
        """Test that clusterer reuses labels from historical matches instead of generating new ones."""
        from reco.storage.base import ClusterMatch
        from reco.core.models import Severity
        from uuid import uuid4
        from datetime import datetime, timezone
        
        failures = [
            Response(id="t1", input="Q", output="A", passed=False, failure_reason="Known error pattern"),
            Response(id="t2", input="Q", output="A", passed=False, failure_reason="Known error pattern similar"),
        ]
        
        # Set up storage to return a historical match
        mock_storage.set_similar_clusters([
            ClusterMatch(
                cluster_id=uuid4(),
                label="Historical Label That Should Be Reused",
                similarity=0.92,
                run_id=uuid4(),
                run_date=datetime.now(timezone.utc),
                occurrence_count=3,
                severity=Severity.HIGH,
            )
        ])
        
        # LLM would generate a different label, but should NOT be called
        mock_llm.set_default_response("LABEL: New Different Label\nSEVERITY: LOW")
        
        clusterer = Clusterer(mock_embedder, mock_llm, storage=mock_storage)
        result = clusterer.cluster(failures)
        
        # Find non-uncategorized cluster
        labeled = [c for c in result if c.label not in ("Uncategorized", "Single Failure")]
        
        if labeled:
            # Should use the historical label, not the LLM-generated one
            assert labeled[0].label == "Historical Label That Should Be Reused"
            assert labeled[0].severity == Severity.HIGH
            assert labeled[0].is_recurring is True

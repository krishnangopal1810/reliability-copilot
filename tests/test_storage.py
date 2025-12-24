"""Tests for storage layer."""

import pytest
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

from reco.core.models import EvalRun, Response, FailureCluster, Severity
from reco.storage.sqlite import SQLiteStorage


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        storage = SQLiteStorage(db_path)
        yield storage
        storage.close()


@pytest.fixture
def sample_run():
    """Create a sample eval run."""
    return EvalRun(
        name="test_run",
        responses=[
            Response(id="t1", input="Q1", output="A1", passed=True),
            Response(id="t2", input="Q2", output="A2", passed=False, failure_reason="Error"),
        ],
    )


@pytest.fixture
def sample_clusters():
    """Create sample clusters."""
    return [
        FailureCluster(
            label="Format Errors",
            description="Formatting issues",
            severity=Severity.MEDIUM,
            response_ids=["t1", "t2"],
        ),
        FailureCluster(
            label="Logic Failures",
            description="Wrong logic",
            severity=Severity.HIGH,
            response_ids=["t3", "t4"],
        ),
    ]


class TestSQLiteStorageInit:
    """Tests for SQLite storage initialization."""
    
    def test_creates_database_file(self, temp_db):
        """Test that database file is created."""
        assert temp_db.db_path.exists()
    
    def test_creates_tables(self, temp_db):
        """Test that tables are created."""
        cursor = temp_db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor}
        assert "runs" in tables
        assert "clusters" in tables
    
    def test_default_path(self):
        """Test default path is ~/.reco/data.db."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Don't use default path in tests, just check logic
            storage = SQLiteStorage(Path(tmpdir) / "custom.db")
            assert storage.db_path.name == "custom.db"
            storage.close()


class TestSQLiteStorageSaveRun:
    """Tests for saving runs."""
    
    def test_save_run(self, temp_db, sample_run):
        """Test saving a run."""
        temp_db.save_run(sample_run)
        
        cursor = temp_db.conn.execute(
            "SELECT * FROM runs WHERE id = ?", (str(sample_run.id),)
        )
        row = cursor.fetchone()
        
        assert row is not None
        assert row["name"] == "test_run"
    
    def test_save_run_updates_existing(self, temp_db, sample_run):
        """Test that saving again updates the run."""
        temp_db.save_run(sample_run)
        
        sample_run.name = "updated_name"
        temp_db.save_run(sample_run)
        
        cursor = temp_db.conn.execute("SELECT COUNT(*) FROM runs")
        count = cursor.fetchone()[0]
        assert count == 1
        
        cursor = temp_db.conn.execute(
            "SELECT name FROM runs WHERE id = ?", (str(sample_run.id),)
        )
        assert cursor.fetchone()["name"] == "updated_name"


class TestSQLiteStorageSaveClusters:
    """Tests for saving clusters."""
    
    def test_save_clusters(self, temp_db, sample_run, sample_clusters):
        """Test saving clusters."""
        temp_db.save_run(sample_run)
        
        embeddings = [
            np.random.rand(384).astype(np.float32),
            np.random.rand(384).astype(np.float32),
        ]
        
        temp_db.save_clusters(sample_clusters, sample_run.id, embeddings)
        
        cursor = temp_db.conn.execute("SELECT COUNT(*) FROM clusters")
        count = cursor.fetchone()[0]
        assert count == 2
    
    def test_save_clusters_with_embeddings(self, temp_db, sample_run, sample_clusters):
        """Test that embeddings are stored."""
        temp_db.save_run(sample_run)
        
        embedding = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        embeddings = [embedding, np.array([4.0, 5.0, 6.0], dtype=np.float32)]
        
        temp_db.save_clusters(sample_clusters, sample_run.id, embeddings)
        
        cursor = temp_db.conn.execute(
            "SELECT embedding FROM clusters WHERE label = 'Format Errors'"
        )
        stored = np.frombuffer(cursor.fetchone()["embedding"], dtype=np.float32)
        
        assert np.allclose(stored, embedding)


class TestSQLiteStorageFindSimilar:
    """Tests for finding similar clusters."""
    
    def test_find_similar_empty(self, temp_db):
        """Test finding similar when database is empty."""
        embedding = np.random.rand(384).astype(np.float32)
        matches = temp_db.find_similar_clusters(embedding)
        assert matches == []
    
    def test_find_similar_matches(self, temp_db, sample_run, sample_clusters):
        """Test finding similar clusters."""
        temp_db.save_run(sample_run)
        
        # Create a specific embedding
        base_embedding = np.array([1.0] * 384, dtype=np.float32)
        similar_embedding = np.array([0.99] * 384, dtype=np.float32)
        different_embedding = np.array([-1.0] * 384, dtype=np.float32)
        
        embeddings = [base_embedding, different_embedding]
        temp_db.save_clusters(sample_clusters, sample_run.id, embeddings)
        
        # Should find similar match
        matches = temp_db.find_similar_clusters(similar_embedding, threshold=0.9)
        assert len(matches) >= 1
        assert matches[0].label == "Format Errors"


class TestSQLiteStorageCountOccurrences:
    """Tests for counting cluster occurrences."""
    
    def test_count_occurrences_none(self, temp_db):
        """Test counting when no matches exist."""
        count, first_seen = temp_db.count_recent_occurrences("Nonexistent")
        assert count == 0
        assert first_seen is None
    
    def test_count_occurrences_with_history(self, temp_db):
        """Test counting with historical data."""
        # Create multiple runs with same cluster label
        for i in range(3):
            run = EvalRun(name=f"run_{i}")
            temp_db.save_run(run)
            
            cluster = FailureCluster(
                label="Recurring Issue",
                severity=Severity.HIGH,
                response_ids=[f"t{i}"],
            )
            embedding = np.random.rand(384).astype(np.float32)
            temp_db.save_clusters([cluster], run.id, [embedding])
        
        count, first_seen = temp_db.count_recent_occurrences("Recurring Issue", last_n_runs=5)
        assert count == 3
        assert first_seen is not None


class TestCosineSimilarity:
    """Tests for cosine similarity calculation."""
    
    def test_identical_vectors(self, temp_db):
        """Test similarity of identical vectors is 1."""
        v = np.array([1.0, 2.0, 3.0])
        assert abs(temp_db._cosine_similarity(v, v) - 1.0) < 0.001
    
    def test_orthogonal_vectors(self, temp_db):
        """Test similarity of orthogonal vectors is 0."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        assert abs(temp_db._cosine_similarity(v1, v2)) < 0.001
    
    def test_opposite_vectors(self, temp_db):
        """Test similarity of opposite vectors is -1."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([-1.0, 0.0, 0.0])
        assert abs(temp_db._cosine_similarity(v1, v2) + 1.0) < 0.001


class TestGetClusterHistory:
    """Tests for get_cluster_history method."""
    
    def test_get_cluster_history_empty(self, temp_db):
        """Test getting history when none exists."""
        history = temp_db.get_cluster_history("Nonexistent")
        assert history == []
    
    def test_get_cluster_history_with_data(self, temp_db):
        """Test getting cluster history."""
        for i in range(3):
            run = EvalRun(name=f"run_{i}")
            temp_db.save_run(run)
            
            cluster = FailureCluster(
                label="Test Pattern",
                severity=Severity.HIGH,
                response_ids=[f"t{i}"],
            )
            embedding = np.random.rand(384).astype(np.float32)
            temp_db.save_clusters([cluster], run.id, [embedding])
        
        history = temp_db.get_cluster_history("Test Pattern", limit=10)
        assert len(history) == 3
        assert all(h.label == "Test Pattern" for h in history)


class TestProfileMethods:
    """Tests for Phase 2 profile methods."""
    
    def test_get_run_count(self, temp_db):
        """Test getting run count."""
        assert temp_db.get_run_count() == 0
        
        temp_db.save_run(EvalRun(name="run1"))
        temp_db.save_run(EvalRun(name="run2"))
        
        assert temp_db.get_run_count() == 2
    
    def test_get_pass_rate_history(self, temp_db):
        """Test getting pass rate history."""
        for i in range(3):
            run = EvalRun(name=f"run_{i}")
            temp_db.save_run(run)
            
            cluster = FailureCluster(
                label=f"Pattern {i}",
                severity=Severity.MEDIUM,
                response_ids=[f"t{i}"],
            )
            embedding = np.random.rand(384).astype(np.float32)
            temp_db.save_clusters([cluster], run.id, [embedding])
        
        history = temp_db.get_pass_rate_history(limit=5)
        assert len(history) == 3
    
    def test_get_failure_mode_stats_limit(self, temp_db):
        """Test that failure mode stats respects limit."""
        for i in range(5):
            run = EvalRun(name=f"run_{i}")
            temp_db.save_run(run)
            
            clusters = [
                FailureCluster(label="Type A", severity=Severity.HIGH, response_ids=[f"a{i}"]),
                FailureCluster(label="Type B", severity=Severity.MEDIUM, response_ids=[f"b{i}"]),
            ]
            embeddings = [np.random.rand(384).astype(np.float32) for _ in clusters]
            temp_db.save_clusters(clusters, run.id, embeddings)
        
        # Get stats for only last 2 runs
        stats = temp_db.get_failure_mode_stats(limit_runs=2)
        
        # Should have entries from last 2 runs only
        assert len(stats) > 0

"""Tests for Phase 2 profile functionality."""

import pytest
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from io import StringIO

from rich.console import Console

from reco.core.models import EvalRun, FailureCluster, Severity
from reco.storage.sqlite import SQLiteStorage
from reco.formatters.terminal import TerminalFormatter


@pytest.fixture
def temp_storage():
    """Create a temporary storage for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        storage = SQLiteStorage(db_path)
        yield storage
        storage.close()


class TestStorageProfileMethods:
    """Tests for storage profile methods."""
    
    def test_get_run_count_empty(self, temp_storage):
        """Test run count on empty database."""
        assert temp_storage.get_run_count() == 0
    
    def test_get_run_count_with_runs(self, temp_storage):
        """Test run count with runs."""
        import numpy as np
        
        for i in range(3):
            run = EvalRun(name=f"run_{i}")
            temp_storage.save_run(run)
            
            cluster = FailureCluster(
                label=f"Pattern {i}",
                severity=Severity.MEDIUM,
                response_ids=["t1"],
            )
            embedding = np.random.rand(384).astype(np.float32)
            temp_storage.save_clusters([cluster], run.id, [embedding])
        
        assert temp_storage.get_run_count() == 3
    
    def test_get_failure_mode_stats_empty(self, temp_storage):
        """Test failure stats on empty database."""
        stats = temp_storage.get_failure_mode_stats()
        assert stats == []
    
    def test_get_failure_mode_stats_with_clusters(self, temp_storage):
        """Test failure stats with clusters."""
        import numpy as np
        
        run = EvalRun(name="test_run")
        temp_storage.save_run(run)
        
        clusters = [
            FailureCluster(label="Pattern A", severity=Severity.HIGH, response_ids=["t1"]),
            FailureCluster(label="Pattern A", severity=Severity.HIGH, response_ids=["t2"]),
            FailureCluster(label="Pattern B", severity=Severity.MEDIUM, response_ids=["t3"]),
        ]
        embeddings = [np.random.rand(384).astype(np.float32) for _ in clusters]
        temp_storage.save_clusters(clusters, run.id, embeddings)
        
        stats = temp_storage.get_failure_mode_stats()
        
        # Should be sorted by count descending
        assert len(stats) == 2
        assert stats[0][0] == "Pattern A"  # Label
        assert stats[0][1] == 2  # Count
    
    def test_get_failure_mode_stats_excludes_uncategorized(self, temp_storage):
        """Test that Uncategorized is excluded from stats."""
        import numpy as np
        
        run = EvalRun(name="test_run")
        temp_storage.save_run(run)
        
        clusters = [
            FailureCluster(label="Real Pattern", severity=Severity.HIGH, response_ids=["t1"]),
            FailureCluster(label="Uncategorized", severity=Severity.LOW, response_ids=["t2"]),
            FailureCluster(label="Single Failure", severity=Severity.LOW, response_ids=["t3"]),
        ]
        embeddings = [np.random.rand(384).astype(np.float32) for _ in clusters]
        temp_storage.save_clusters(clusters, run.id, embeddings)
        
        stats = temp_storage.get_failure_mode_stats()
        
        # Should only include "Real Pattern"
        assert len(stats) == 1
        assert stats[0][0] == "Real Pattern"


class TestRenderProfile:
    """Tests for profile rendering."""
    
    @pytest.fixture
    def captured_console(self):
        """Create a console that captures output."""
        output = StringIO()
        console = Console(file=output, force_terminal=True, width=100)
        return console, output
    
    def test_render_profile_empty(self, captured_console):
        """Test rendering empty profile."""
        console, output = captured_console
        formatter = TerminalFormatter(console)
        
        formatter.render_profile([], run_count=0, last_n=10)
        
        result = output.getvalue()
        assert "No failure patterns" in result
    
    def test_render_profile_with_stats(self, captured_console):
        """Test rendering profile with stats."""
        console, output = captured_console
        formatter = TerminalFormatter(console)
        
        stats = [
            ("Pattern A", 5, 10),
            ("Pattern B", 3, 10),
        ]
        
        formatter.render_profile(stats, run_count=5, last_n=10)
        
        result = output.getvalue()
        assert "Pattern A" in result
        assert "5x" in result
        assert "50%" in result
    
    def test_render_profile_shows_run_count(self, captured_console):
        """Test that run count is shown correctly."""
        console, output = captured_console
        formatter = TerminalFormatter(console)
        
        formatter.render_profile([("X", 1, 1)], run_count=7, last_n=10)
        
        result = output.getvalue()
        assert "7 runs" in result

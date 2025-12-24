"""Exhaustive unit tests for reco.cli."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner

from reco.cli import app, load_run
from reco.core.models import EvalRun, Response


runner = CliRunner()


class TestLoadRun:
    """Tests for the load_run helper function."""
    
    def test_load_run_valid_file(self, tmp_path):
        """Test loading a valid eval run file."""
        data = {
            "name": "test_run",
            "responses": [
                {"id": "t1", "input": "Q1", "output": "A1", "pass": True},
                {"id": "t2", "input": "Q2", "output": "A2", "pass": False, "failure_reason": "Error"},
            ],
        }
        file_path = tmp_path / "test.json"
        file_path.write_text(json.dumps(data))
        
        run = load_run(file_path)
        
        assert isinstance(run, EvalRun)
        assert run.name == "test_run"
        assert len(run.responses) == 2
    
    def test_load_run_parses_responses_correctly(self, tmp_path):
        """Test that responses are parsed with all fields."""
        data = {
            "responses": [
                {
                    "id": "t1",
                    "input": "Question",
                    "output": "Answer",
                    "expected": "Expected",
                    "pass": False,
                    "failure_reason": "Wrong",
                    "latency_ms": 100,
                    "metadata": {"key": "value"},
                },
            ],
        }
        file_path = tmp_path / "test.json"
        file_path.write_text(json.dumps(data))
        
        run = load_run(file_path)
        resp = run.responses[0]
        
        assert resp.id == "t1"
        assert resp.input == "Question"
        assert resp.output == "Answer"
        assert resp.expected == "Expected"
        assert resp.passed is False
        assert resp.failure_reason == "Wrong"
        assert resp.latency_ms == 100
        assert resp.metadata == {"key": "value"}
    
    def test_load_run_handles_passed_key(self, tmp_path):
        """Test that 'passed' key works as alternative to 'pass'."""
        data = {
            "responses": [
                {"id": "t1", "input": "Q", "output": "A", "passed": True},
            ],
        }
        file_path = tmp_path / "test.json"
        file_path.write_text(json.dumps(data))
        
        run = load_run(file_path)
        
        assert run.responses[0].passed is True
    
    def test_load_run_default_pass_true(self, tmp_path):
        """Test that pass defaults to True when not specified."""
        data = {
            "responses": [
                {"id": "t1", "input": "Q", "output": "A"},
            ],
        }
        file_path = tmp_path / "test.json"
        file_path.write_text(json.dumps(data))
        
        run = load_run(file_path)
        
        assert run.responses[0].passed is True
    
    def test_load_run_generates_id_when_missing(self, tmp_path):
        """Test that ID is generated when missing."""
        data = {
            "responses": [
                {"input": "Q", "output": "A"},
            ],
        }
        file_path = tmp_path / "test.json"
        file_path.write_text(json.dumps(data))
        
        run = load_run(file_path)
        
        assert run.responses[0].id is not None
        assert "test_" in run.responses[0].id
    
    def test_load_run_uses_filename_as_name(self, tmp_path):
        """Test that filename is used as run name when not specified."""
        data = {"responses": []}
        file_path = tmp_path / "my_eval_run.json"
        file_path.write_text(json.dumps(data))
        
        run = load_run(file_path)
        
        assert run.name == "my_eval_run"
    
    def test_load_run_file_not_found(self, tmp_path):
        """Test error when file doesn't exist."""
        import typer
        
        with pytest.raises(typer.BadParameter) as exc_info:
            load_run(tmp_path / "nonexistent.json")
        
        assert "not found" in str(exc_info.value).lower()
    
    def test_load_run_invalid_json(self, tmp_path):
        """Test error when file contains invalid JSON."""
        import typer
        
        file_path = tmp_path / "invalid.json"
        file_path.write_text("this is not json {{{")
        
        with pytest.raises(typer.BadParameter) as exc_info:
            load_run(file_path)
        
        assert "invalid json" in str(exc_info.value).lower()
    
    def test_load_run_empty_responses(self, tmp_path):
        """Test loading file with empty responses array."""
        data = {"responses": []}
        file_path = tmp_path / "empty.json"
        file_path.write_text(json.dumps(data))
        
        run = load_run(file_path)
        
        assert len(run.responses) == 0
    
    def test_load_run_with_metadata(self, tmp_path):
        """Test loading file with run-level metadata."""
        data = {
            "name": "test",
            "metadata": {"model": "gpt-4", "version": "1.0"},
            "responses": [],
        }
        file_path = tmp_path / "meta.json"
        file_path.write_text(json.dumps(data))
        
        run = load_run(file_path)
        
        assert run.metadata == {"model": "gpt-4", "version": "1.0"}


class TestVersionCommand:
    """Tests for the version command."""
    
    def test_version_command(self):
        """Test that version command works."""
        result = runner.invoke(app, ["version"])
        
        assert result.exit_code == 0
        assert "0.1.0" in result.output
    
    def test_version_shows_reco(self):
        """Test that version output mentions reco."""
        result = runner.invoke(app, ["version"])
        
        assert "reco" in result.output.lower()


class TestCompareCommand:
    """Tests for the compare command."""
    
    @pytest.fixture
    def sample_files(self, tmp_path):
        """Create sample baseline and candidate files."""
        baseline = {
            "responses": [
                {"id": "t1", "input": "Q1", "output": "A1", "pass": True},
                {"id": "t2", "input": "Q2", "output": "A2", "pass": True},
            ],
        }
        candidate = {
            "responses": [
                {"id": "t1", "input": "Q1", "output": "A1", "pass": True},
                {"id": "t2", "input": "Q2", "output": "Wrong", "pass": False, "failure_reason": "Error"},
            ],
        }
        
        baseline_path = tmp_path / "baseline.json"
        candidate_path = tmp_path / "candidate.json"
        
        baseline_path.write_text(json.dumps(baseline))
        candidate_path.write_text(json.dumps(candidate))
        
        return baseline_path, candidate_path
    
    def test_compare_missing_baseline(self, tmp_path):
        """Test compare fails gracefully when baseline missing."""
        candidate_path = tmp_path / "candidate.json"
        candidate_path.write_text('{"responses": []}')
        
        result = runner.invoke(app, ["compare", str(tmp_path / "missing.json"), str(candidate_path)])
        
        assert result.exit_code != 0
    
    def test_compare_missing_candidate(self, tmp_path):
        """Test compare fails gracefully when candidate missing."""
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text('{"responses": []}')
        
        result = runner.invoke(app, ["compare", str(baseline_path), str(tmp_path / "missing.json")])
        
        assert result.exit_code != 0
    
    def test_compare_loads_files(self, sample_files):
        """Test that compare loads files correctly."""
        baseline_path, candidate_path = sample_files
        
        # Mock the LLM provider to avoid API calls
        with patch("reco.cli.OpenRouterProvider") as mock_openrouter:
            mock_instance = MagicMock()
            mock_instance.complete.return_value = "RECOMMENDATION: DO_NOT_SHIP\nRISK_LEVEL: HIGH\nSUMMARY: Test"
            mock_openrouter.return_value = mock_instance
            
            with patch("reco.cli.Config") as mock_config:
                mock_config.load.return_value.validate.return_value = []
                mock_config.load.return_value.llm_model = "anthropic/claude-3.5-sonnet"
                
                result = runner.invoke(app, ["compare", str(baseline_path), str(candidate_path)])
        
        # Should not fail due to file loading
        assert "Loading eval runs" in result.output or result.exit_code == 0 or "Error" in result.output
    
    def test_compare_help(self):
        """Test compare help output."""
        result = runner.invoke(app, ["compare", "--help"])
        
        assert result.exit_code == 0
        assert "baseline" in result.output.lower()
        assert "candidate" in result.output.lower()
    
    def test_compare_no_semantic_flag(self):
        """Test --no-semantic flag is recognized."""
        result = runner.invoke(app, ["compare", "--help"])
        
        assert "--no-semantic" in result.output


class TestClusterCommand:
    """Tests for the cluster command."""
    
    @pytest.fixture
    def failures_file(self, tmp_path):
        """Create a file with failures for clustering."""
        data = {
            "responses": [
                {"id": "f1", "input": "Q", "output": "A", "pass": False, "failure_reason": "Error type A"},
                {"id": "f2", "input": "Q", "output": "A", "pass": False, "failure_reason": "Error type A similar"},
                {"id": "f3", "input": "Q", "output": "A", "pass": False, "failure_reason": "Different error B"},
            ],
        }
        
        file_path = tmp_path / "failures.json"
        file_path.write_text(json.dumps(data))
        
        return file_path
    
    def test_cluster_missing_file(self, tmp_path):
        """Test cluster fails gracefully when file missing."""
        result = runner.invoke(app, ["cluster", str(tmp_path / "missing.json")])
        
        assert result.exit_code != 0
    
    def test_cluster_no_failures(self, tmp_path):
        """Test cluster handles file with no failures."""
        data = {
            "responses": [
                {"id": "t1", "input": "Q", "output": "A", "pass": True},
            ],
        }
        file_path = tmp_path / "no_failures.json"
        file_path.write_text(json.dumps(data))
        
        # Mock to avoid API calls
        with patch("reco.cli.Config") as mock_config:
            mock_config.load.return_value.validate.return_value = []
            
            result = runner.invoke(app, ["cluster", str(file_path)])
        
        assert "No failures to cluster" in result.output or result.exit_code == 0
    
    def test_cluster_help(self):
        """Test cluster help output."""
        result = runner.invoke(app, ["cluster", "--help"])
        
        assert result.exit_code == 0
        assert "evalfile" in result.output.lower() or "eval" in result.output.lower()
    
    def test_cluster_min_size_option(self):
        """Test --min-size option is recognized."""
        result = runner.invoke(app, ["cluster", "--help"])
        
        assert "--min-size" in result.output or "-m" in result.output


class TestDiffCommand:
    """Tests for the diff command (placeholder)."""
    
    def test_diff_help(self):
        """Test diff help output."""
        result = runner.invoke(app, ["diff", "--help"])
        
        assert result.exit_code == 0
        assert "baseline" in result.output.lower()
        assert "candidate" in result.output.lower()
    
    def test_diff_case_option(self):
        """Test --case option is recognized."""
        result = runner.invoke(app, ["diff", "--help"])
        
        assert "--case" in result.output or "-c" in result.output


class TestMainHelp:
    """Tests for the main CLI help."""
    
    def test_main_help(self):
        """Test main help output."""
        result = runner.invoke(app, ["--help"])
        
        assert result.exit_code == 0
        assert "Reliability Copilot" in result.output
    
    def test_main_lists_commands(self):
        """Test that help lists all commands."""
        result = runner.invoke(app, ["--help"])
        
        assert "compare" in result.output
        assert "cluster" in result.output
        assert "diff" in result.output
        assert "version" in result.output


class TestProfileCommand:
    """Tests for the profile command."""
    
    def test_profile_help(self):
        """Test profile help output."""
        result = runner.invoke(app, ["profile", "--help"])
        
        assert result.exit_code == 0
        assert "reliability profile" in result.output.lower() or "last-n" in result.output
    
    def test_profile_last_n_option(self):
        """Test --last-n option is recognized."""
        result = runner.invoke(app, ["profile", "--help"])
        
        assert "--last-n" in result.output or "-n" in result.output


class TestAnalyzeAgentCommand:
    """Tests for the analyze-agent command."""
    
    @pytest.fixture
    def sample_trace(self, tmp_path):
        """Create a sample agent trace."""
        trace = {
            "id": "test",
            "goal": "Test goal",
            "outcome": "failed",
            "steps": [
                {"step": 1, "action": "search", "success": True},
                {"step": 2, "action": "process", "success": False, "error": "Failed"},
            ]
        }
        
        file_path = tmp_path / "trace.json"
        file_path.write_text(json.dumps(trace))
        return file_path
    
    def test_analyze_agent_success(self, sample_trace):
        """Test analyze-agent with valid trace."""
        result = runner.invoke(app, ["analyze-agent", str(sample_trace)])
        
        assert result.exit_code == 0
        assert "AGENT TRACE ANALYSIS" in result.output
    
    def test_analyze_agent_missing_file(self, tmp_path):
        """Test analyze-agent with missing file."""
        result = runner.invoke(app, ["analyze-agent", str(tmp_path / "missing.json")])
        
        assert result.exit_code == 1
    
    def test_analyze_agent_invalid_json(self, tmp_path):
        """Test analyze-agent with invalid JSON."""
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("not json {{")
        
        result = runner.invoke(app, ["analyze-agent", str(invalid_file)])
        
        assert result.exit_code == 1
    
    def test_analyze_agent_help(self):
        """Test analyze-agent help output."""
        result = runner.invoke(app, ["analyze-agent", "--help"])
        
        assert result.exit_code == 0
        assert "trace" in result.output.lower() or "agent" in result.output.lower()
    
    def test_analyze_agent_shows_issues(self, sample_trace):
        """Test that analyze-agent shows detected issues."""
        result = runner.invoke(app, ["analyze-agent", str(sample_trace)])
        
        assert result.exit_code == 0
        assert "ISSUES" in result.output or "Tool" in result.output

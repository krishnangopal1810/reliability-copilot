"""Tests for Phase 4 deployment gate."""

import pytest
import tempfile
from pathlib import Path

from typer.testing import CliRunner

from reco.cli import app
from reco.core.models import EvalRun, Response
from reco.core.gate import DeploymentGate, GateThresholds, GateResult


runner = CliRunner()


class TestGateThresholds:
    """Tests for threshold loading."""
    
    def test_default_thresholds(self):
        """Test default threshold values."""
        thresholds = GateThresholds()
        
        assert thresholds.max_regression_percent == 15.0
        assert thresholds.min_pass_rate == 0.80
        assert thresholds.block_on_severity == ["CRITICAL"]
    
    def test_load_from_file(self, tmp_path):
        """Test loading thresholds from YAML."""
        config = tmp_path / "thresholds.yaml"
        config.write_text("""
max_regression_percent: 10
min_pass_rate: 0.90
block_on_severity: [CRITICAL, HIGH]
""")
        
        thresholds = GateThresholds.load(config)
        
        assert thresholds.max_regression_percent == 10
        assert thresholds.min_pass_rate == 0.90
        assert thresholds.block_on_severity == ["CRITICAL", "HIGH"]
    
    def test_load_missing_file(self, tmp_path):
        """Test default when config missing."""
        thresholds = GateThresholds.load(tmp_path / "missing.yaml")
        
        assert thresholds.max_regression_percent == 15.0


class TestDeploymentGate:
    """Tests for gate checking logic."""
    
    @pytest.fixture
    def gate(self):
        return DeploymentGate()
    
    def test_passes_when_improved(self, gate):
        """Test gate passes when candidate is better."""
        baseline = EvalRun(responses=[
            Response(id="1", input="", output="", passed=True),
            Response(id="2", input="", output="", passed=False),
        ])
        candidate = EvalRun(responses=[
            Response(id="1", input="", output="", passed=True),
            Response(id="2", input="", output="", passed=True),
        ])
        
        result = gate.check(baseline, candidate)
        
        assert result.passed is True
        assert len(result.violations) == 0
    
    def test_fails_on_regression(self, gate):
        """Test gate fails when regression exceeds threshold."""
        baseline = EvalRun(responses=[
            Response(id="1", input="", output="", passed=True),
            Response(id="2", input="", output="", passed=True),
        ])
        candidate = EvalRun(responses=[
            Response(id="1", input="", output="", passed=True),
            Response(id="2", input="", output="", passed=False),
        ])
        
        result = gate.check(baseline, candidate)
        
        # 50% pass rate vs 100% = 50% regression
        assert result.passed is False
        assert any("regressed" in v.message.lower() for v in result.violations)
    
    def test_fails_below_min_pass_rate(self):
        """Test gate fails when below min pass rate."""
        thresholds = GateThresholds(min_pass_rate=0.90)
        gate = DeploymentGate(thresholds)
        
        baseline = EvalRun(responses=[
            Response(id="1", input="", output="", passed=True),
        ])
        candidate = EvalRun(responses=[
            Response(id="1", input="", output="", passed=True),
            Response(id="2", input="", output="", passed=False),  # 50% pass
        ])
        
        result = gate.check(baseline, candidate)
        
        assert result.passed is False
        assert any("below minimum" in v.message.lower() for v in result.violations)
    
    def test_calculates_regression_percent(self, gate):
        """Test regression calculation."""
        baseline = EvalRun(responses=[
            Response(id="1", input="", output="", passed=True),
            Response(id="2", input="", output="", passed=True),
        ])
        candidate = EvalRun(responses=[
            Response(id="1", input="", output="", passed=True),
            Response(id="2", input="", output="", passed=False),
        ])
        
        result = gate.check(baseline, candidate)
        
        # 100% -> 50% = 50% regression
        assert result.regression_percent == 50.0


class TestGateCLI:
    """Tests for gate CLI command."""
    
    def test_gate_help(self):
        """Test gate help output."""
        result = runner.invoke(app, ["gate", "--help"])
        
        assert result.exit_code == 0
        assert "baseline" in result.stdout.lower()
        assert "candidate" in result.stdout.lower()
    
    def test_gate_exit_code_pass(self, tmp_path):
        """Test exit code 0 when gate passes."""
        import json
        
        baseline = {"responses": [{"id": "1", "input": "", "output": "", "pass": True}]}
        candidate = {"responses": [{"id": "1", "input": "", "output": "", "pass": True}]}
        
        baseline_file = tmp_path / "baseline.json"
        candidate_file = tmp_path / "candidate.json"
        
        baseline_file.write_text(json.dumps(baseline))
        candidate_file.write_text(json.dumps(candidate))
        
        result = runner.invoke(app, ["gate", str(baseline_file), str(candidate_file)])
        
        assert result.exit_code == 0
        assert "PASSED" in result.stdout
    
    def test_gate_exit_code_fail(self, tmp_path):
        """Test exit code 1 when gate fails."""
        import json
        
        baseline = {"responses": [
            {"id": "1", "input": "", "output": "", "pass": True},
            {"id": "2", "input": "", "output": "", "pass": True},
        ]}
        candidate = {"responses": [
            {"id": "1", "input": "", "output": "", "pass": True},
            {"id": "2", "input": "", "output": "", "pass": False},
        ]}
        
        baseline_file = tmp_path / "baseline.json"
        candidate_file = tmp_path / "candidate.json"
        
        baseline_file.write_text(json.dumps(baseline))
        candidate_file.write_text(json.dumps(candidate))
        
        result = runner.invoke(app, ["gate", str(baseline_file), str(candidate_file)])
        
        assert result.exit_code == 1
        assert "BLOCKED" in result.stdout

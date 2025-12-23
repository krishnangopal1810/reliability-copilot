"""Exhaustive unit tests for reco.core.models."""

import pytest
from datetime import datetime, timezone
from uuid import UUID

from reco.core.models import (
    Response,
    EvalRun,
    FailureCluster,
    Comparison,
    Judgment,
    Severity,
)


class TestSeverity:
    """Tests for the Severity enum."""
    
    def test_severity_values(self):
        """Test that all severity levels have correct string values."""
        assert Severity.LOW.value == "low"
        assert Severity.MEDIUM.value == "medium"
        assert Severity.HIGH.value == "high"
        assert Severity.CRITICAL.value == "critical"
    
    def test_severity_from_string(self):
        """Test creating severity from string name."""
        assert Severity["LOW"] == Severity.LOW
        assert Severity["MEDIUM"] == Severity.MEDIUM
        assert Severity["HIGH"] == Severity.HIGH
        assert Severity["CRITICAL"] == Severity.CRITICAL
    
    def test_severity_comparison(self):
        """Test that severity levels are distinct."""
        severities = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
        assert len(set(severities)) == 4
    
    def test_invalid_severity_raises(self):
        """Test that invalid severity names raise KeyError."""
        with pytest.raises(KeyError):
            Severity["INVALID"]


class TestResponse:
    """Tests for the Response data class."""
    
    def test_response_minimal(self):
        """Test creating a response with minimal fields."""
        r = Response(id="t1", input="Q", output="A")
        
        assert r.id == "t1"
        assert r.input == "Q"
        assert r.output == "A"
        assert r.expected is None
        assert r.passed is True  # Default
        assert r.failure_reason is None
        assert r.latency_ms is None
        assert r.metadata == {}
    
    def test_response_full(self):
        """Test creating a response with all fields."""
        r = Response(
            id="t1",
            input="Question",
            output="Answer",
            expected="Expected",
            passed=False,
            failure_reason="Wrong answer",
            latency_ms=100,
            metadata={"model": "gpt-4"},
        )
        
        assert r.id == "t1"
        assert r.input == "Question"
        assert r.output == "Answer"
        assert r.expected == "Expected"
        assert r.passed is False
        assert r.failure_reason == "Wrong answer"
        assert r.latency_ms == 100
        assert r.metadata == {"model": "gpt-4"}
    
    def test_response_passed_default_true(self):
        """Test that passed defaults to True."""
        r = Response(id="t1", input="Q", output="A")
        assert r.passed is True
    
    def test_response_metadata_default_empty(self):
        """Test that metadata defaults to empty dict."""
        r = Response(id="t1", input="Q", output="A")
        assert r.metadata == {}
        assert isinstance(r.metadata, dict)
    
    def test_response_metadata_mutable(self):
        """Test that metadata can be modified after creation."""
        r = Response(id="t1", input="Q", output="A")
        r.metadata["key"] = "value"
        assert r.metadata["key"] == "value"
    
    def test_response_equality(self):
        """Test response equality."""
        r1 = Response(id="t1", input="Q", output="A")
        r2 = Response(id="t1", input="Q", output="A")
        assert r1 == r2
    
    def test_response_inequality_different_id(self):
        """Test responses with different IDs are not equal."""
        r1 = Response(id="t1", input="Q", output="A")
        r2 = Response(id="t2", input="Q", output="A")
        assert r1 != r2


class TestEvalRun:
    """Tests for the EvalRun data class."""
    
    def test_eval_run_minimal(self):
        """Test creating an eval run with defaults."""
        run = EvalRun()
        
        assert isinstance(run.id, UUID)
        assert run.name == ""
        assert run.responses == []
        assert isinstance(run.created_at, datetime)
        assert run.metadata == {}
    
    def test_eval_run_with_responses(self):
        """Test creating an eval run with responses."""
        responses = [
            Response(id="t1", input="Q1", output="A1", passed=True),
            Response(id="t2", input="Q2", output="A2", passed=False),
        ]
        run = EvalRun(name="test_run", responses=responses)
        
        assert run.name == "test_run"
        assert len(run.responses) == 2
    
    def test_eval_run_failures_property(self):
        """Test the failures property filters correctly."""
        responses = [
            Response(id="t1", input="Q1", output="A1", passed=True),
            Response(id="t2", input="Q2", output="A2", passed=False, failure_reason="Error"),
            Response(id="t3", input="Q3", output="A3", passed=True),
            Response(id="t4", input="Q4", output="A4", passed=False, failure_reason="Error"),
        ]
        run = EvalRun(responses=responses)
        
        failures = run.failures
        assert len(failures) == 2
        assert all(not f.passed for f in failures)
        assert {f.id for f in failures} == {"t2", "t4"}
    
    def test_eval_run_failures_empty_when_all_pass(self, all_passing_run):
        """Test failures is empty when all tests pass."""
        assert len(all_passing_run.failures) == 0
    
    def test_eval_run_failures_all_when_none_pass(self, all_failing_run):
        """Test failures contains all when none pass."""
        assert len(all_failing_run.failures) == 3
    
    def test_eval_run_pass_rate_all_passing(self, all_passing_run):
        """Test pass rate is 1.0 when all pass."""
        assert all_passing_run.pass_rate == 1.0
    
    def test_eval_run_pass_rate_all_failing(self, all_failing_run):
        """Test pass rate is 0.0 when all fail."""
        assert all_failing_run.pass_rate == 0.0
    
    def test_eval_run_pass_rate_mixed(self, mixed_run):
        """Test pass rate for mixed results."""
        assert mixed_run.pass_rate == 0.5  # 2 pass, 2 fail
    
    def test_eval_run_pass_rate_empty(self, empty_eval_run):
        """Test pass rate is 0.0 for empty run."""
        assert empty_eval_run.pass_rate == 0.0
    
    def test_eval_run_id_unique(self):
        """Test that each run gets a unique ID."""
        run1 = EvalRun()
        run2 = EvalRun()
        assert run1.id != run2.id
    
    def test_eval_run_created_at_auto(self):
        """Test that created_at is set automatically."""
        before = datetime.now(timezone.utc)
        run = EvalRun()
        after = datetime.now(timezone.utc)
        
        assert before <= run.created_at <= after


class TestFailureCluster:
    """Tests for the FailureCluster data class."""
    
    def test_failure_cluster_minimal(self):
        """Test creating a cluster with defaults."""
        cluster = FailureCluster()
        
        assert isinstance(cluster.id, UUID)
        assert cluster.label == ""
        assert cluster.description == ""
        assert cluster.severity == Severity.MEDIUM  # Default
        assert cluster.response_ids == []
        assert cluster.first_seen is None
        assert cluster.occurrence_count == 1
    
    def test_failure_cluster_full(self):
        """Test creating a cluster with all fields."""
        now = datetime.now(timezone.utc)
        cluster = FailureCluster(
            label="Financial Hallucinations",
            description="Model invents numbers",
            severity=Severity.HIGH,
            response_ids=["t1", "t2", "t3"],
            first_seen=now,
            occurrence_count=5,
        )
        
        assert cluster.label == "Financial Hallucinations"
        assert cluster.description == "Model invents numbers"
        assert cluster.severity == Severity.HIGH
        assert len(cluster.response_ids) == 3
        assert cluster.first_seen == now
        assert cluster.occurrence_count == 5
    
    def test_failure_cluster_severity_default_medium(self):
        """Test that severity defaults to MEDIUM."""
        cluster = FailureCluster()
        assert cluster.severity == Severity.MEDIUM
    
    def test_failure_cluster_response_ids_mutable(self):
        """Test that response_ids can be modified."""
        cluster = FailureCluster(response_ids=["t1"])
        cluster.response_ids.append("t2")
        assert len(cluster.response_ids) == 2


class TestComparison:
    """Tests for the Comparison data class."""
    
    def test_comparison_creation(self, baseline_run, candidate_run_worse):
        """Test creating a comparison."""
        comp = Comparison(
            baseline=baseline_run,
            candidate=candidate_run_worse,
            improvements=["t4"],
            regressions=["t1"],
            unchanged=["t2", "t3", "t5"],
        )
        
        assert comp.baseline == baseline_run
        assert comp.candidate == candidate_run_worse
        assert comp.improvements == ["t4"]
        assert comp.regressions == ["t1"]
        assert len(comp.unchanged) == 3
    
    def test_comparison_defaults(self, baseline_run, candidate_run_worse):
        """Test comparison default values."""
        comp = Comparison(
            baseline=baseline_run,
            candidate=candidate_run_worse,
        )
        
        assert comp.improvements == []
        assert comp.regressions == []
        assert comp.unchanged == []
        assert comp.summary == ""
        assert comp.recommendation == ""
        assert comp.confidence == 0.0
    
    def test_comparison_recommendation_mutable(self, baseline_run, candidate_run_worse):
        """Test that recommendation can be set."""
        comp = Comparison(baseline=baseline_run, candidate=candidate_run_worse)
        comp.recommendation = "do_not_ship"
        assert comp.recommendation == "do_not_ship"


class TestJudgment:
    """Tests for the Judgment data class."""
    
    def test_judgment_creation(self, sample_comparison):
        """Test creating a judgment."""
        judgment = Judgment(
            comparison=sample_comparison,
            narrative="This is the analysis.",
            risk_level=Severity.HIGH,
            key_findings=["Finding 1", "Finding 2"],
            action_items=["Action 1"],
        )
        
        assert judgment.comparison == sample_comparison
        assert judgment.narrative == "This is the analysis."
        assert judgment.risk_level == Severity.HIGH
        assert len(judgment.key_findings) == 2
        assert len(judgment.action_items) == 1
    
    def test_judgment_defaults(self, sample_comparison):
        """Test judgment default values."""
        judgment = Judgment(comparison=sample_comparison)
        
        assert judgment.narrative == ""
        assert judgment.risk_level == Severity.MEDIUM
        assert judgment.key_findings == []
        assert judgment.action_items == []
    
    def test_judgment_risk_level_default_medium(self, sample_comparison):
        """Test that risk_level defaults to MEDIUM."""
        judgment = Judgment(comparison=sample_comparison)
        assert judgment.risk_level == Severity.MEDIUM

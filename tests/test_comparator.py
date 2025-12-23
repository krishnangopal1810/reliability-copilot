"""Exhaustive unit tests for reco.core.comparator."""

import pytest

from reco.core.models import Response, EvalRun, Comparison
from reco.core.comparator import Comparator, ComparisonConfig
from tests.conftest import MockLLMProvider


class TestComparisonConfig:
    """Tests for ComparisonConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ComparisonConfig()
        
        assert config.semantic_diff is True
        assert config.strict_matching is False
        assert config.ignore_unchanged is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = ComparisonConfig(
            semantic_diff=False,
            strict_matching=True,
            ignore_unchanged=False,
        )
        
        assert config.semantic_diff is False
        assert config.strict_matching is True
        assert config.ignore_unchanged is False


class TestComparator:
    """Tests for the Comparator class."""
    
    # ========================================================================
    # Initialization
    # ========================================================================
    
    def test_comparator_init_with_llm(self, mock_llm):
        """Test comparator initialization with LLM provider."""
        comparator = Comparator(mock_llm)
        
        assert comparator.llm == mock_llm
        assert comparator.config is not None
    
    def test_comparator_init_with_custom_config(self, mock_llm):
        """Test comparator initialization with custom config."""
        config = ComparisonConfig(semantic_diff=False)
        comparator = Comparator(mock_llm, config)
        
        assert comparator.config.semantic_diff is False
    
    # ========================================================================
    # Basic Comparisons
    # ========================================================================
    
    def test_compare_identical_runs(self, mock_llm, all_passing_run):
        """Test comparing identical runs."""
        comparator = Comparator(mock_llm)
        result = comparator.compare(all_passing_run, all_passing_run)
        
        assert isinstance(result, Comparison)
        assert result.baseline == all_passing_run
        assert result.candidate == all_passing_run
        assert len(result.improvements) == 0
        assert len(result.regressions) == 0
        assert len(result.unchanged) == 3
    
    def test_compare_empty_runs(self, mock_llm, empty_eval_run):
        """Test comparing empty runs."""
        comparator = Comparator(mock_llm)
        result = comparator.compare(empty_eval_run, empty_eval_run)
        
        assert len(result.improvements) == 0
        assert len(result.regressions) == 0
        assert len(result.unchanged) == 0
    
    def test_compare_returns_comparison_object(self, mock_llm, baseline_run, candidate_run_better):
        """Test that compare returns a Comparison object."""
        comparator = Comparator(mock_llm)
        result = comparator.compare(baseline_run, candidate_run_better)
        
        assert isinstance(result, Comparison)
    
    # ========================================================================
    # Improvement Detection
    # ========================================================================
    
    def test_detect_improvement_failing_to_passing(self, mock_llm, baseline_run, candidate_run_better):
        """Test detecting improvement when failing test starts passing."""
        comparator = Comparator(mock_llm)
        result = comparator.compare(baseline_run, candidate_run_better)
        
        # t4 was failing in baseline, passing in candidate
        assert "t4" in result.improvements
    
    def test_detect_multiple_improvements(self, mock_llm):
        """Test detecting multiple improvements."""
        baseline = EvalRun(responses=[
            Response(id="t1", input="Q1", output="A1", passed=False, failure_reason="Error"),
            Response(id="t2", input="Q2", output="A2", passed=False, failure_reason="Error"),
            Response(id="t3", input="Q3", output="A3", passed=False, failure_reason="Error"),
        ])
        candidate = EvalRun(responses=[
            Response(id="t1", input="Q1", output="A1", passed=True),
            Response(id="t2", input="Q2", output="A2", passed=True),
            Response(id="t3", input="Q3", output="A3", passed=False, failure_reason="Error"),
        ])
        
        comparator = Comparator(mock_llm)
        result = comparator.compare(baseline, candidate)
        
        assert len(result.improvements) == 2
        assert "t1" in result.improvements
        assert "t2" in result.improvements
    
    # ========================================================================
    # Regression Detection
    # ========================================================================
    
    def test_detect_regression_passing_to_failing(self, mock_llm, baseline_run, candidate_run_worse):
        """Test detecting regression when passing test starts failing."""
        comparator = Comparator(mock_llm)
        result = comparator.compare(baseline_run, candidate_run_worse)
        
        # t1 and t3 were passing in baseline, failing in candidate
        assert "t1" in result.regressions
        assert "t3" in result.regressions
    
    def test_detect_multiple_regressions(self, mock_llm):
        """Test detecting multiple regressions."""
        baseline = EvalRun(responses=[
            Response(id="t1", input="Q1", output="A1", passed=True),
            Response(id="t2", input="Q2", output="A2", passed=True),
            Response(id="t3", input="Q3", output="A3", passed=True),
        ])
        candidate = EvalRun(responses=[
            Response(id="t1", input="Q1", output="A1", passed=False, failure_reason="Broke"),
            Response(id="t2", input="Q2", output="A2", passed=False, failure_reason="Broke"),
            Response(id="t3", input="Q3", output="A3", passed=False, failure_reason="Broke"),
        ])
        
        comparator = Comparator(mock_llm)
        result = comparator.compare(baseline, candidate)
        
        assert len(result.regressions) == 3
    
    # ========================================================================
    # Unchanged Detection
    # ========================================================================
    
    def test_detect_unchanged_same_output(self, mock_llm):
        """Test detecting unchanged when output is identical."""
        baseline = EvalRun(responses=[
            Response(id="t1", input="Q1", output="Same output", passed=True),
        ])
        candidate = EvalRun(responses=[
            Response(id="t1", input="Q1", output="Same output", passed=True),
        ])
        
        comparator = Comparator(mock_llm)
        result = comparator.compare(baseline, candidate)
        
        assert len(result.unchanged) == 1
        assert "t1" in result.unchanged
    
    def test_detect_unchanged_both_failing_same(self, mock_llm):
        """Test detecting unchanged when both fail with same output."""
        baseline = EvalRun(responses=[
            Response(id="t1", input="Q1", output="Error", passed=False, failure_reason="Same error"),
        ])
        candidate = EvalRun(responses=[
            Response(id="t1", input="Q1", output="Error", passed=False, failure_reason="Same error"),
        ])
        
        comparator = Comparator(mock_llm)
        result = comparator.compare(baseline, candidate)
        
        assert len(result.unchanged) == 1
    
    # ========================================================================
    # ID Matching
    # ========================================================================
    
    def test_only_compares_common_ids(self, mock_llm):
        """Test that only common IDs are compared."""
        baseline = EvalRun(responses=[
            Response(id="t1", input="Q1", output="A1", passed=True),
            Response(id="t2", input="Q2", output="A2", passed=True),
            Response(id="baseline_only", input="Q", output="A", passed=True),
        ])
        candidate = EvalRun(responses=[
            Response(id="t1", input="Q1", output="A1", passed=True),
            Response(id="t2", input="Q2", output="A2", passed=True),
            Response(id="candidate_only", input="Q", output="A", passed=True),
        ])
        
        comparator = Comparator(mock_llm)
        result = comparator.compare(baseline, candidate)
        
        # baseline_only and candidate_only should not be in any category
        all_ids = result.improvements + result.regressions + result.unchanged
        assert "baseline_only" not in all_ids
        assert "candidate_only" not in all_ids
        assert len(all_ids) == 2  # Only t1 and t2
    
    def test_no_common_ids(self, mock_llm):
        """Test comparison with no common IDs."""
        baseline = EvalRun(responses=[
            Response(id="b1", input="Q1", output="A1", passed=True),
        ])
        candidate = EvalRun(responses=[
            Response(id="c1", input="Q1", output="A1", passed=True),
        ])
        
        comparator = Comparator(mock_llm)
        result = comparator.compare(baseline, candidate)
        
        assert len(result.improvements) == 0
        assert len(result.regressions) == 0
        assert len(result.unchanged) == 0
    
    # ========================================================================
    # Semantic Comparison
    # ========================================================================
    
    def test_semantic_diff_calls_llm_on_output_change(self, mock_llm):
        """Test that LLM is called for semantic diff when output changes."""
        baseline = EvalRun(responses=[
            Response(id="t1", input="Q1", output="Original answer", passed=True),
        ])
        candidate = EvalRun(responses=[
            Response(id="t1", input="Q1", output="Modified answer", passed=True),
        ])
        
        comparator = Comparator(mock_llm)
        comparator.compare(baseline, candidate)
        
        # LLM should have been called for semantic comparison
        assert len(mock_llm.call_history) > 0
    
    def test_semantic_diff_better_is_improvement(self):
        """Test that semantic 'better' is classified as improvement."""
        mock_llm = MockLLMProvider(responses={"compare": "BETTER"})
        
        baseline = EvalRun(responses=[
            Response(id="t1", input="Q1", output="Old answer", passed=True),
        ])
        candidate = EvalRun(responses=[
            Response(id="t1", input="Q1", output="Better answer", passed=True),
        ])
        
        comparator = Comparator(mock_llm)
        result = comparator.compare(baseline, candidate)
        
        assert "t1" in result.improvements
    
    def test_semantic_diff_worse_is_regression(self):
        """Test that semantic 'worse' is classified as regression."""
        mock_llm = MockLLMProvider(responses={"compare": "WORSE"})
        
        baseline = EvalRun(responses=[
            Response(id="t1", input="Q1", output="Good answer", passed=True),
        ])
        candidate = EvalRun(responses=[
            Response(id="t1", input="Q1", output="Worse answer", passed=True),
        ])
        
        comparator = Comparator(mock_llm)
        result = comparator.compare(baseline, candidate)
        
        assert "t1" in result.regressions
    
    def test_semantic_diff_neutral_is_unchanged(self):
        """Test that semantic 'neutral' is classified as unchanged."""
        mock_llm = MockLLMProvider(responses={"compare": "NEUTRAL"})
        
        baseline = EvalRun(responses=[
            Response(id="t1", input="Q1", output="Answer A", passed=True),
        ])
        candidate = EvalRun(responses=[
            Response(id="t1", input="Q1", output="Answer B", passed=True),
        ])
        
        comparator = Comparator(mock_llm)
        result = comparator.compare(baseline, candidate)
        
        assert "t1" in result.unchanged
    
    def test_semantic_diff_disabled(self, mock_llm):
        """Test that semantic diff can be disabled."""
        baseline = EvalRun(responses=[
            Response(id="t1", input="Q1", output="Answer A", passed=True),
        ])
        candidate = EvalRun(responses=[
            Response(id="t1", input="Q1", output="Answer B", passed=True),
        ])
        
        config = ComparisonConfig(semantic_diff=False)
        comparator = Comparator(mock_llm, config)
        result = comparator.compare(baseline, candidate)
        
        # Without semantic diff, changed outputs are marked unchanged
        assert "t1" in result.unchanged
        # LLM should not have been called
        assert len(mock_llm.call_history) == 0
    
    def test_semantic_diff_handles_llm_error(self):
        """Test that LLM errors are handled gracefully."""
        class FailingLLM:
            def complete(self, prompt, max_tokens=1000):
                raise Exception("API error")
        
        baseline = EvalRun(responses=[
            Response(id="t1", input="Q1", output="Answer A", passed=True),
        ])
        candidate = EvalRun(responses=[
            Response(id="t1", input="Q1", output="Answer B", passed=True),
        ])
        
        comparator = Comparator(FailingLLM())
        result = comparator.compare(baseline, candidate)
        
        # Should fall back to neutral on error
        assert "t1" in result.unchanged
    
    # ========================================================================
    # Edge Cases
    # ========================================================================
    
    def test_compare_with_empty_responses_in_baseline(self, mock_llm):
        """Test comparison when baseline has empty responses list."""
        baseline = EvalRun(responses=[])
        candidate = EvalRun(responses=[
            Response(id="t1", input="Q1", output="A1", passed=True),
        ])
        
        comparator = Comparator(mock_llm)
        result = comparator.compare(baseline, candidate)
        
        # No common IDs
        assert len(result.improvements) == 0
        assert len(result.regressions) == 0
        assert len(result.unchanged) == 0
    
    def test_compare_with_empty_responses_in_candidate(self, mock_llm):
        """Test comparison when candidate has empty responses list."""
        baseline = EvalRun(responses=[
            Response(id="t1", input="Q1", output="A1", passed=True),
        ])
        candidate = EvalRun(responses=[])
        
        comparator = Comparator(mock_llm)
        result = comparator.compare(baseline, candidate)
        
        # No common IDs
        assert len(result.improvements) == 0
        assert len(result.regressions) == 0
        assert len(result.unchanged) == 0
    
    def test_compare_preserves_run_references(self, mock_llm, baseline_run, candidate_run_worse):
        """Test that comparison preserves references to original runs."""
        comparator = Comparator(mock_llm)
        result = comparator.compare(baseline_run, candidate_run_worse)
        
        assert result.baseline is baseline_run
        assert result.candidate is candidate_run_worse

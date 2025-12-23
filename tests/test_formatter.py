"""Exhaustive unit tests for reco.formatters.terminal."""

import pytest
from io import StringIO
from unittest.mock import MagicMock

from rich.console import Console

from reco.core.models import (
    Response, EvalRun, Comparison, Judgment, FailureCluster, Severity
)
from reco.formatters.terminal import TerminalFormatter


class TestTerminalFormatterInit:
    """Tests for TerminalFormatter initialization."""
    
    def test_init_with_console(self):
        """Test initialization with provided console."""
        console = Console()
        formatter = TerminalFormatter(console)
        assert formatter.console is console
    
    def test_init_without_console_creates_default(self):
        """Test initialization creates default console."""
        formatter = TerminalFormatter()
        assert formatter.console is not None
        assert isinstance(formatter.console, Console)


class TestTerminalFormatterConstants:
    """Tests for TerminalFormatter constants."""
    
    def test_severity_colors_all_defined(self):
        """Test that all severity levels have colors."""
        formatter = TerminalFormatter()
        
        for severity in Severity:
            assert severity in formatter.SEVERITY_COLORS
    
    def test_recommendation_styles_all_defined(self):
        """Test that all recommendation types have styles."""
        formatter = TerminalFormatter()
        
        expected_recommendations = ["ship", "do_not_ship", "needs_review"]
        for rec in expected_recommendations:
            assert rec in formatter.RECOMMENDATION_STYLES
            color, icon = formatter.RECOMMENDATION_STYLES[rec]
            assert color is not None
            assert icon is not None


class TestRenderJudgment:
    """Tests for render_judgment method."""
    
    @pytest.fixture
    def captured_console(self):
        """Create a console that captures output."""
        output = StringIO()
        console = Console(file=output, force_terminal=True, width=80)
        return console, output
    
    @pytest.fixture
    def basic_judgment(self, baseline_run, candidate_run_worse):
        """Create a basic judgment for testing."""
        comparison = Comparison(
            baseline=baseline_run,
            candidate=candidate_run_worse,
            improvements=[],
            regressions=["t1", "t3"],
            unchanged=["t2", "t4", "t5"],
            recommendation="do_not_ship",
        )
        return Judgment(
            comparison=comparison,
            narrative="The candidate has critical regressions.",
            risk_level=Severity.HIGH,
            key_findings=["Regression in t1", "Regression in t3"],
            action_items=["Fix t1 before shipping"],
        )
    
    def test_render_judgment_no_error(self, captured_console, basic_judgment):
        """Test that render_judgment completes without error."""
        console, output = captured_console
        formatter = TerminalFormatter(console)
        
        # Should not raise
        formatter.render_judgment(basic_judgment)
    
    def test_render_judgment_includes_recommendation(self, captured_console, basic_judgment):
        """Test that output includes the recommendation."""
        console, output = captured_console
        formatter = TerminalFormatter(console)
        
        formatter.render_judgment(basic_judgment)
        
        rendered = output.getvalue()
        assert "DO NOT SHIP" in rendered or "DO_NOT_SHIP" in rendered
    
    def test_render_judgment_includes_regressions(self, captured_console, basic_judgment):
        """Test that output includes regression information."""
        console, output = captured_console
        formatter = TerminalFormatter(console)
        
        formatter.render_judgment(basic_judgment)
        
        rendered = output.getvalue()
        assert "REGRESSED" in rendered or "regress" in rendered.lower()
    
    def test_render_judgment_includes_improvements_when_present(self, captured_console, baseline_run, candidate_run_better):
        """Test that improvements are shown when present."""
        comparison = Comparison(
            baseline=baseline_run,
            candidate=candidate_run_better,
            improvements=["t4"],
            regressions=[],
            unchanged=["t1", "t2", "t3", "t5"],
            recommendation="ship",
        )
        judgment = Judgment(
            comparison=comparison,
            narrative="Fixed a bug.",
            risk_level=Severity.LOW,
        )
        
        console, output = captured_console
        formatter = TerminalFormatter(console)
        
        formatter.render_judgment(judgment)
        
        rendered = output.getvalue()
        assert "IMPROVED" in rendered or "improve" in rendered.lower()
    
    def test_render_judgment_includes_narrative(self, captured_console, basic_judgment):
        """Test that narrative is included in output."""
        console, output = captured_console
        formatter = TerminalFormatter(console)
        
        formatter.render_judgment(basic_judgment)
        
        rendered = output.getvalue()
        # The narrative should appear somewhere
        assert "regression" in rendered.lower() or "Summary" in rendered
    
    def test_render_judgment_includes_findings(self, captured_console, basic_judgment):
        """Test that key findings are included."""
        console, output = captured_console
        formatter = TerminalFormatter(console)
        
        formatter.render_judgment(basic_judgment)
        
        rendered = output.getvalue()
        assert "Finding" in rendered or "finding" in rendered.lower()
    
    def test_render_judgment_includes_actions(self, captured_console, basic_judgment):
        """Test that action items are included."""
        console, output = captured_console
        formatter = TerminalFormatter(console)
        
        formatter.render_judgment(basic_judgment)
        
        rendered = output.getvalue()
        assert "Action" in rendered or "action" in rendered.lower()
    
    def test_render_judgment_includes_risk_level(self, captured_console, basic_judgment):
        """Test that risk level is included."""
        console, output = captured_console
        formatter = TerminalFormatter(console)
        
        formatter.render_judgment(basic_judgment)
        
        rendered = output.getvalue()
        assert "HIGH" in rendered or "Risk" in rendered
    
    def test_render_judgment_ship_recommendation(self, captured_console, baseline_run, candidate_run_better):
        """Test rendering a SHIP recommendation."""
        comparison = Comparison(
            baseline=baseline_run,
            candidate=candidate_run_better,
            improvements=["t4"],
            regressions=[],
            recommendation="ship",
        )
        judgment = Judgment(
            comparison=comparison,
            risk_level=Severity.LOW,
        )
        
        console, output = captured_console
        formatter = TerminalFormatter(console)
        
        formatter.render_judgment(judgment)
        
        rendered = output.getvalue()
        assert "SHIP" in rendered
    
    def test_render_judgment_needs_review_recommendation(self, captured_console, baseline_run, candidate_run_worse):
        """Test rendering a NEEDS_REVIEW recommendation."""
        comparison = Comparison(
            baseline=baseline_run,
            candidate=candidate_run_worse,
            recommendation="needs_review",
        )
        judgment = Judgment(
            comparison=comparison,
            risk_level=Severity.MEDIUM,
        )
        
        console, output = captured_console
        formatter = TerminalFormatter(console)
        
        formatter.render_judgment(judgment)
        
        rendered = output.getvalue()
        assert "NEEDS REVIEW" in rendered or "NEEDS_REVIEW" in rendered
    
    def test_render_judgment_handles_empty_lists(self, captured_console, baseline_run, candidate_run_worse):
        """Test rendering when key_findings and action_items are empty."""
        comparison = Comparison(
            baseline=baseline_run,
            candidate=candidate_run_worse,
            recommendation="do_not_ship",
        )
        judgment = Judgment(
            comparison=comparison,
            key_findings=[],
            action_items=[],
        )
        
        console, output = captured_console
        formatter = TerminalFormatter(console)
        
        # Should not raise
        formatter.render_judgment(judgment)


class TestRenderClusters:
    """Tests for render_clusters method."""
    
    @pytest.fixture
    def captured_console(self):
        """Create a console that captures output."""
        output = StringIO()
        console = Console(file=output, force_terminal=True, width=80)
        return console, output
    
    @pytest.fixture
    def sample_clusters(self):
        """Create sample clusters for testing."""
        return [
            FailureCluster(
                label="Financial Hallucinations",
                description="Model invents financial data",
                severity=Severity.HIGH,
                response_ids=["t1", "t2", "t3"],
            ),
            FailureCluster(
                label="Format Violations",
                description="Ignores format requirements",
                severity=Severity.MEDIUM,
                response_ids=["t4", "t5"],
            ),
            FailureCluster(
                label="Uncategorized",
                description="Misc failures",
                severity=Severity.LOW,
                response_ids=["t6"],
            ),
        ]
    
    def test_render_clusters_no_error(self, captured_console, sample_clusters):
        """Test that render_clusters completes without error."""
        console, output = captured_console
        formatter = TerminalFormatter(console)
        
        # Should not raise
        formatter.render_clusters(sample_clusters, total_failures=6)
    
    def test_render_clusters_includes_header(self, captured_console, sample_clusters):
        """Test that output includes cluster header."""
        console, output = captured_console
        formatter = TerminalFormatter(console)
        
        formatter.render_clusters(sample_clusters, total_failures=6)
        
        rendered = output.getvalue()
        assert "CLUSTER" in rendered or "cluster" in rendered.lower()
    
    def test_render_clusters_includes_labels(self, captured_console, sample_clusters):
        """Test that cluster labels are included."""
        console, output = captured_console
        formatter = TerminalFormatter(console)
        
        formatter.render_clusters(sample_clusters, total_failures=6)
        
        rendered = output.getvalue()
        assert "Financial Hallucinations" in rendered
        assert "Format Violations" in rendered
    
    def test_render_clusters_includes_severity(self, captured_console, sample_clusters):
        """Test that severity is shown for clusters."""
        console, output = captured_console
        formatter = TerminalFormatter(console)
        
        formatter.render_clusters(sample_clusters, total_failures=6)
        
        rendered = output.getvalue()
        assert "HIGH" in rendered
        assert "MEDIUM" in rendered
    
    def test_render_clusters_includes_case_count(self, captured_console, sample_clusters):
        """Test that case count is shown."""
        console, output = captured_console
        formatter = TerminalFormatter(console)
        
        formatter.render_clusters(sample_clusters, total_failures=6)
        
        rendered = output.getvalue()
        # Should show counts like "(3 cases)"
        assert "3" in rendered  # First cluster has 3
        assert "2" in rendered  # Second cluster has 2
    
    def test_render_clusters_includes_case_ids(self, captured_console, sample_clusters):
        """Test that case IDs are shown."""
        console, output = captured_console
        formatter = TerminalFormatter(console)
        
        formatter.render_clusters(sample_clusters, total_failures=6)
        
        rendered = output.getvalue()
        assert "t1" in rendered
    
    def test_render_clusters_shows_total_counts_in_header(self, captured_console, sample_clusters):
        """Test that header shows total failures and pattern count."""
        console, output = captured_console
        formatter = TerminalFormatter(console)
        
        formatter.render_clusters(sample_clusters, total_failures=6)
        
        rendered = output.getvalue()
        assert "6" in rendered  # Total failures
    
    def test_render_clusters_empty_list(self, captured_console):
        """Test rendering empty cluster list."""
        console, output = captured_console
        formatter = TerminalFormatter(console)
        
        # Should not raise
        formatter.render_clusters([], total_failures=0)
    
    def test_render_clusters_single_cluster(self, captured_console):
        """Test rendering single cluster."""
        console, output = captured_console
        formatter = TerminalFormatter(console)
        
        clusters = [
            FailureCluster(
                label="Single Pattern",
                severity=Severity.MEDIUM,
                response_ids=["t1"],
            ),
        ]
        
        formatter.render_clusters(clusters, total_failures=1)
        
        rendered = output.getvalue()
        assert "Single Pattern" in rendered
    
    def test_render_clusters_truncates_many_ids(self, captured_console):
        """Test that many case IDs are truncated."""
        console, output = captured_console
        formatter = TerminalFormatter(console)
        
        # Cluster with many IDs
        many_ids = [f"test_{i}" for i in range(10)]
        clusters = [
            FailureCluster(
                label="Many Failures",
                severity=Severity.HIGH,
                response_ids=many_ids,
            ),
        ]
        
        formatter.render_clusters(clusters, total_failures=10)
        
        rendered = output.getvalue()
        # Should show "+X more" for remaining
        assert "more" in rendered.lower() or "..." in rendered


class TestRenderError:
    """Tests for render_error method."""
    
    @pytest.fixture
    def captured_console(self):
        """Create a console that captures output."""
        output = StringIO()
        console = Console(file=output, force_terminal=True, width=80)
        return console, output
    
    def test_render_error_basic(self, captured_console):
        """Test basic error rendering."""
        console, output = captured_console
        formatter = TerminalFormatter(console)
        
        formatter.render_error("Something went wrong")
        
        rendered = output.getvalue()
        assert "Error" in rendered
        assert "Something went wrong" in rendered
    
    def test_render_error_with_hint(self, captured_console):
        """Test error rendering with hint."""
        console, output = captured_console
        formatter = TerminalFormatter(console)
        
        formatter.render_error("API key missing", hint="Set ANTHROPIC_API_KEY")
        
        rendered = output.getvalue()
        assert "API key missing" in rendered
        assert "ANTHROPIC_API_KEY" in rendered
    
    def test_render_error_without_hint(self, captured_console):
        """Test error rendering without hint doesn't crash."""
        console, output = captured_console
        formatter = TerminalFormatter(console)
        
        # Should not raise
        formatter.render_error("Error message", hint=None)


class TestRenderSuccess:
    """Tests for render_success method."""
    
    @pytest.fixture
    def captured_console(self):
        """Create a console that captures output."""
        output = StringIO()
        console = Console(file=output, force_terminal=True, width=80)
        return console, output
    
    def test_render_success(self, captured_console):
        """Test success message rendering."""
        console, output = captured_console
        formatter = TerminalFormatter(console)
        
        formatter.render_success("Operation completed")
        
        rendered = output.getvalue()
        assert "Operation completed" in rendered


class TestRenderWarning:
    """Tests for render_warning method."""
    
    @pytest.fixture
    def captured_console(self):
        """Create a console that captures output."""
        output = StringIO()
        console = Console(file=output, force_terminal=True, width=80)
        return console, output
    
    def test_render_warning(self, captured_console):
        """Test warning message rendering."""
        console, output = captured_console
        formatter = TerminalFormatter(console)
        
        formatter.render_warning("This feature is experimental")
        
        rendered = output.getvalue()
        assert "This feature is experimental" in rendered

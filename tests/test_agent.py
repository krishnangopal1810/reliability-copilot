"""Tests for Phase 3 agent trace analysis."""

import pytest
from io import StringIO

from rich.console import Console

from reco.core.models import (
    AgentStep,
    AgentTrace,
    AgentIssue,
    AgentAnalysis,
    Severity,
)
from reco.core.agent_analyzer import AgentAnalyzer, load_trace_from_file
from reco.formatters.terminal import TerminalFormatter


class TestAgentModels:
    """Tests for agent data models."""
    
    def test_agent_step_creation(self):
        """Test creating an agent step."""
        step = AgentStep(
            step=1,
            action="search",
            input={"query": "test"},
            output={"results": []},
            success=True,
        )
        
        assert step.step == 1
        assert step.action == "search"
        assert step.success is True
    
    def test_agent_trace_success_rate(self):
        """Test calculating success rate."""
        trace = AgentTrace(
            id="test",
            goal="Test goal",
            outcome="partial",
            steps=[
                AgentStep(step=1, action="a", success=True),
                AgentStep(step=2, action="b", success=True),
                AgentStep(step=3, action="c", success=False),
                AgentStep(step=4, action="d", success=True),
            ],
        )
        
        assert trace.success_rate == 0.75
    
    def test_agent_trace_failed_steps(self):
        """Test getting failed steps."""
        trace = AgentTrace(
            id="test",
            goal="Test goal",
            outcome="failed",
            steps=[
                AgentStep(step=1, action="a", success=True),
                AgentStep(step=2, action="b", success=False),
                AgentStep(step=3, action="c", success=False),
            ],
        )
        
        failed = trace.failed_steps
        assert len(failed) == 2
        assert failed[0].step == 2
        assert failed[1].step == 3
    
    def test_agent_trace_tools_used(self):
        """Test getting unique tools used."""
        trace = AgentTrace(
            id="test",
            goal="Test",
            outcome="success",
            steps=[
                AgentStep(step=1, action="search"),
                AgentStep(step=2, action="search"),
                AgentStep(step=3, action="format"),
            ],
        )
        
        assert trace.tools_used == ["search", "format"]


class TestAgentAnalyzer:
    """Tests for the agent analyzer."""
    
    @pytest.fixture
    def analyzer(self):
        return AgentAnalyzer()
    
    def test_detect_tool_error(self, analyzer):
        """Test detecting tool execution errors."""
        trace = AgentTrace(
            id="test",
            goal="Test",
            outcome="failed",
            steps=[
                AgentStep(step=1, action="search", success=False, error="API error"),
            ],
        )
        
        analysis = analyzer.analyze(trace)
        
        tool_errors = [i for i in analysis.issues if i.issue_type == "Tool Execution Error"]
        assert len(tool_errors) == 1
        assert tool_errors[0].step == 1
    
    def test_detect_no_recovery(self, analyzer):
        """Test detecting when no recovery is attempted."""
        trace = AgentTrace(
            id="test",
            goal="Test",
            outcome="failed",
            steps=[
                AgentStep(step=1, action="search", success=True),
                AgentStep(step=2, action="book", success=False, error="Failed"),
            ],
        )
        
        analysis = analyzer.analyze(trace)
        
        no_recovery = [i for i in analysis.issues if i.issue_type == "No Recovery Attempted"]
        assert len(no_recovery) == 1
    
    def test_detect_excessive_retries(self, analyzer):
        """Test detecting excessive retries."""
        trace = AgentTrace(
            id="test",
            goal="Test",
            outcome="failed",
            steps=[
                AgentStep(step=1, action="book", success=False),
                AgentStep(step=2, action="book", success=False),
                AgentStep(step=3, action="book", success=False),
                AgentStep(step=4, action="book", success=False),
            ],
        )
        
        analysis = analyzer.analyze(trace)
        
        retries = [i for i in analysis.issues if i.issue_type == "Excessive Retries"]
        assert len(retries) == 1
    
    def test_generates_recommendations(self, analyzer):
        """Test that recommendations are generated."""
        trace = AgentTrace(
            id="test",
            goal="Test",
            outcome="failed",
            steps=[
                AgentStep(step=1, action="book", success=False),
            ],
        )
        
        analysis = analyzer.analyze(trace)
        
        assert len(analysis.recommendations) > 0
    
    def test_identifies_patterns(self, analyzer):
        """Test pattern identification."""
        trace = AgentTrace(
            id="test",
            goal="Test",
            outcome="failed",
            steps=[
                AgentStep(step=1, action="search", success=False),
                AgentStep(step=2, action="search", success=False),
                AgentStep(step=3, action="search", success=False),
            ],
        )
        
        analysis = analyzer.analyze(trace)
        
        assert any("tool" in p.lower() for p in analysis.patterns)


class TestAgentFormatter:
    """Tests for agent analysis rendering."""
    
    @pytest.fixture
    def captured_console(self):
        output = StringIO()
        console = Console(file=output, force_terminal=True, width=100)
        return console, output
    
    def test_render_agent_analysis(self, captured_console):
        """Test rendering agent analysis."""
        console, output = captured_console
        formatter = TerminalFormatter(console)
        
        trace = AgentTrace(
            id="test",
            goal="Test goal",
            outcome="failed",
            steps=[
                AgentStep(step=1, action="search", success=True),
                AgentStep(step=2, action="book", success=False, error="Failed"),
            ],
        )
        
        analysis = AgentAnalysis(
            trace=trace,
            issues=[
                AgentIssue(
                    issue_type="Tool Execution Error",
                    severity=Severity.HIGH,
                    step=2,
                    description="Tool 'book' failed",
                )
            ],
            patterns=["Test pattern"],
            recommendations=["Add retry logic"],
        )
        
        formatter.render_agent_analysis(analysis)
        
        result = output.getvalue()
        assert "AGENT TRACE ANALYSIS" in result
        assert "FAILED" in result
        assert "Tool Execution Error" in result


class TestLoadTraceFromFile:
    """Tests for loading traces from files."""
    
    def test_load_trace_from_file(self, tmp_path):
        """Test loading a valid trace file."""
        import json
        
        trace_data = {
            "id": "test_trace",
            "goal": "Test goal",
            "outcome": "success",
            "steps": [
                {"step": 1, "action": "search", "success": True},
                {"step": 2, "action": "format", "success": True},
            ],
            "metadata": {"key": "value"},
        }
        
        trace_file = tmp_path / "trace.json"
        trace_file.write_text(json.dumps(trace_data))
        
        trace = load_trace_from_file(str(trace_file))
        
        assert trace.id == "test_trace"
        assert trace.goal == "Test goal"
        assert trace.outcome == "success"
        assert len(trace.steps) == 2
    
    def test_load_trace_with_defaults(self, tmp_path):
        """Test loading a minimal trace file."""
        import json
        
        trace_data = {"steps": [{"action": "test"}]}
        
        trace_file = tmp_path / "minimal.json"
        trace_file.write_text(json.dumps(trace_data))
        
        trace = load_trace_from_file(str(trace_file))
        
        assert trace.goal == "Unknown goal"
        assert trace.outcome == "unknown"
        assert len(trace.steps) == 1


class TestAgentAnalyzerEdgeCases:
    """Tests for edge cases in agent analyzer."""
    
    @pytest.fixture
    def analyzer(self):
        return AgentAnalyzer()
    
    def test_goal_abandonment_detection(self, analyzer):
        """Test detecting goal abandonment."""
        trace = AgentTrace(
            id="test",
            goal="Complete the task",
            outcome="failed",
            steps=[
                AgentStep(step=1, action="search", success=True),
                AgentStep(step=2, action="process", success=True),
            ],
        )
        
        analysis = analyzer.analyze(trace)
        
        abandonment = [i for i in analysis.issues if i.issue_type == "Goal Abandonment"]
        assert len(abandonment) == 1
    
    def test_empty_trace(self, analyzer):
        """Test analyzing empty trace."""
        trace = AgentTrace(
            id="test",
            goal="Test",
            outcome="failed",
            steps=[],
        )
        
        analysis = analyzer.analyze(trace)
        
        assert trace.success_rate == 0.0
        assert analysis.issues == []
    
    def test_success_trace_no_issues(self, analyzer):
        """Test that successful traces have minimal issues."""
        trace = AgentTrace(
            id="test",
            goal="Test",
            outcome="success",
            steps=[
                AgentStep(step=1, action="a", success=True),
                AgentStep(step=2, action="b", success=True),
            ],
        )
        
        analysis = analyzer.analyze(trace)
        
        # No tool errors
        tool_errors = [i for i in analysis.issues if i.issue_type == "Tool Execution Error"]
        assert len(tool_errors) == 0
    
    def test_excessive_retries_resets_on_new_action(self, analyzer):
        """Test that excessive retries counter resets when action changes."""
        trace = AgentTrace(
            id="test",
            goal="Test",
            outcome="failed",
            steps=[
                AgentStep(step=1, action="a", success=False),
                AgentStep(step=2, action="a", success=False),
                AgentStep(step=3, action="b", success=False),  # Reset
                AgentStep(step=4, action="b", success=False),
            ],
        )
        
        analysis = analyzer.analyze(trace)
        
        # Should NOT trigger excessive retries (only 2 of each)
        retries = [i for i in analysis.issues if i.issue_type == "Excessive Retries"]
        assert len(retries) == 0

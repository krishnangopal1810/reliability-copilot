"""Agent trace analyzer for multi-step reliability analysis.

Phase 3: Analyze agent execution traces for reliability issues.
"""

from typing import Optional

from .models import (
    AgentTrace,
    AgentStep,
    AgentIssue,
    AgentAnalysis,
    Severity,
)


class AgentAnalyzer:
    """Analyzes agent traces for reliability issues."""
    
    def analyze(self, trace: AgentTrace) -> AgentAnalysis:
        """Analyze an agent trace and return detected issues.
        
        Args:
            trace: The agent trace to analyze
            
        Returns:
            AgentAnalysis with detected issues and patterns
        """
        issues = self._detect_issues(trace)
        patterns = self._identify_patterns(trace, issues)
        recommendations = self._generate_recommendations(issues, patterns)
        
        return AgentAnalysis(
            trace=trace,
            issues=issues,
            patterns=patterns,
            recommendations=recommendations,
        )
    
    def _detect_issues(self, trace: AgentTrace) -> list[AgentIssue]:
        """Detect reliability issues in the trace."""
        issues = []
        
        # Check for tool execution errors
        issues.extend(self._check_tool_errors(trace))
        
        # Check for recovery failures
        issues.extend(self._check_recovery_failures(trace))
        
        # Check for goal abandonment
        issues.extend(self._check_goal_abandonment(trace))
        
        # Check for excessive retries
        issues.extend(self._check_excessive_retries(trace))
        
        return issues
    
    def _check_tool_errors(self, trace: AgentTrace) -> list[AgentIssue]:
        """Detect tool execution failures."""
        issues = []
        
        for step in trace.steps:
            if not step.success:
                issues.append(AgentIssue(
                    issue_type="Tool Execution Error",
                    severity=Severity.HIGH,
                    step=step.step,
                    description=f"Tool '{step.action}' failed at step {step.step}",
                    details={"error": step.error or "Unknown error"},
                ))
        
        return issues
    
    def _check_recovery_failures(self, trace: AgentTrace) -> list[AgentIssue]:
        """Detect when agent fails to recover from errors."""
        issues = []
        
        for i, step in enumerate(trace.steps):
            if not step.success:
                # Check if this was the last step (no recovery attempted)
                if i == len(trace.steps) - 1:
                    issues.append(AgentIssue(
                        issue_type="No Recovery Attempted",
                        severity=Severity.MEDIUM,
                        step=step.step,
                        description="Agent stopped after failure without attempting recovery",
                        details={"failed_action": step.action},
                    ))
                else:
                    # Check if next step is same action (retry) or different (recovery)
                    next_step = trace.steps[i + 1]
                    if next_step.action == step.action and not next_step.success:
                        issues.append(AgentIssue(
                            issue_type="Failed Recovery",
                            severity=Severity.MEDIUM,
                            step=next_step.step,
                            description=f"Recovery attempt for '{step.action}' also failed",
                            details={},
                        ))
        
        return issues
    
    def _check_goal_abandonment(self, trace: AgentTrace) -> list[AgentIssue]:
        """Detect when agent abandons goal without completion."""
        issues = []
        
        if trace.outcome == "failed" and trace.steps:
            last_step = trace.steps[-1]
            if last_step.success:
                # Last step worked but outcome is failed - goal abandoned
                issues.append(AgentIssue(
                    issue_type="Goal Abandonment",
                    severity=Severity.HIGH,
                    step=None,
                    description="Agent stopped before completing goal despite successful steps",
                    details={"goal": trace.goal},
                ))
        
        return issues
    
    def _check_excessive_retries(self, trace: AgentTrace) -> list[AgentIssue]:
        """Detect when agent retries the same action too many times."""
        issues = []
        
        # Count consecutive same actions
        if not trace.steps:
            return issues
        
        current_action = trace.steps[0].action
        count = 1
        max_retries = 3
        
        for step in trace.steps[1:]:
            if step.action == current_action:
                count += 1
                if count > max_retries:
                    issues.append(AgentIssue(
                        issue_type="Excessive Retries",
                        severity=Severity.MEDIUM,
                        step=step.step,
                        description=f"Action '{current_action}' retried {count} times",
                        details={"retry_count": count},
                    ))
                    break  # Only report once per action
            else:
                current_action = step.action
                count = 1
        
        return issues
    
    def _identify_patterns(
        self, 
        trace: AgentTrace, 
        issues: list[AgentIssue]
    ) -> list[str]:
        """Identify higher-level patterns from issues."""
        patterns = []
        
        # Pattern: Tool errors cause task abandonment
        tool_errors = [i for i in issues if i.issue_type == "Tool Execution Error"]
        no_recovery = [i for i in issues if i.issue_type == "No Recovery Attempted"]
        
        if tool_errors and no_recovery:
            patterns.append("Tool errors cause task abandonment")
        
        # Pattern: Cascading failures
        if len(tool_errors) > 2:
            patterns.append("Multiple tool failures indicate systemic issue")
        
        # Pattern: Recovery attempts fail
        failed_recovery = [i for i in issues if i.issue_type == "Failed Recovery"]
        if failed_recovery:
            patterns.append("Recovery mechanisms are ineffective")
        
        return patterns
    
    def _generate_recommendations(
        self, 
        issues: list[AgentIssue], 
        patterns: list[str]
    ) -> list[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        issue_types = {i.issue_type for i in issues}
        
        if "Tool Execution Error" in issue_types:
            recommendations.append("Add retry logic with exponential backoff for tool calls")
        
        if "No Recovery Attempted" in issue_types:
            recommendations.append("Implement fallback strategies when primary tools fail")
        
        if "Goal Abandonment" in issue_types:
            recommendations.append("Add goal-completion validation before terminating")
        
        if "Excessive Retries" in issue_types:
            recommendations.append("Implement circuit breaker to prevent infinite retry loops")
        
        return recommendations


def load_trace_from_file(filepath: str) -> AgentTrace:
    """Load an agent trace from a JSON file."""
    import json
    from pathlib import Path
    
    path = Path(filepath)
    with open(path) as f:
        data = json.load(f)
    
    steps = [
        AgentStep(
            step=s.get("step", i + 1),
            action=s.get("action", "unknown"),
            input=s.get("input", {}),
            output=s.get("output"),
            success=s.get("success", True),
            error=s.get("error"),
            duration_ms=s.get("duration_ms"),
        )
        for i, s in enumerate(data.get("steps", []))
    ]
    
    return AgentTrace(
        id=data.get("id", path.stem),
        goal=data.get("goal", "Unknown goal"),
        outcome=data.get("outcome", "unknown"),
        steps=steps,
        metadata=data.get("metadata", {}),
    )

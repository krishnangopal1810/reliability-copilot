"""Judgment generator for creating narrative analysis."""

from typing import TYPE_CHECKING

from .models import Comparison, Judgment, Severity

if TYPE_CHECKING:
    from ..providers.llm.base import LLMProvider


class JudgmentGenerator:
    """Generates narrative judgment from comparison results."""
    
    def __init__(self, llm: "LLMProvider"):
        self.llm = llm
    
    def generate(self, comparison: Comparison) -> Judgment:
        """Generate a narrative judgment for the comparison.
        
        Args:
            comparison: Comparison object with improvements/regressions
            
        Returns:
            Judgment with narrative, risk level, findings, and actions
        """
        prompt = self._build_prompt(comparison)
        response = self.llm.complete(prompt, max_tokens=600)
        
        return self._parse_response(comparison, response)
    
    def _build_prompt(self, c: Comparison) -> str:
        """Build the prompt for judgment generation."""
        baseline_failures = c.baseline.failures
        candidate_failures = c.candidate.failures
        
        # Build regression details
        regression_details = self._format_regressions(c)
        improvement_details = self._format_improvements(c)
        
        return f"""You are a reliability analyst reviewing a prompt or model change for an AI system.
Your job is to give a clear, opinionated judgment on whether this change should ship.

## BASELINE RUN (before change):
- Total tests: {len(c.baseline.responses)}
- Pass rate: {c.baseline.pass_rate:.1%}
- Failures: {len(baseline_failures)}

## CANDIDATE RUN (proposed change):
- Total tests: {len(c.candidate.responses)}  
- Pass rate: {c.candidate.pass_rate:.1%}
- Failures: {len(candidate_failures)}

## CHANGES DETECTED:
- Improved: {len(c.improvements)} cases
- Regressed: {len(c.regressions)} cases
- Unchanged: {len(c.unchanged)} cases

## REGRESSION DETAILS:
{regression_details}

## IMPROVEMENT DETAILS:
{improvement_details}

---

Provide your analysis in this exact format:

RECOMMENDATION: <SHIP|DO_NOT_SHIP|NEEDS_REVIEW>
RISK_LEVEL: <LOW|MEDIUM|HIGH|CRITICAL>
SUMMARY: <2-3 sentence narrative explaining what changed and the overall impact>

KEY_FINDINGS:
- <finding 1>
- <finding 2>
- <finding 3>

ACTION_ITEMS:
- <action 1>
- <action 2>

---

Be opinionated. If there are ANY regressions, especially around safety, accuracy, or core functionality, lean toward DO_NOT_SHIP.
If improvements outweigh minor regressions, consider SHIP with caveats.
Use NEEDS_REVIEW when the trade-offs are genuinely unclear."""

    def _format_regressions(self, c: Comparison) -> str:
        """Format regression details for the prompt."""
        if not c.regressions:
            return "None detected."
        
        lines = []
        for id in c.regressions[:5]:  # Cap at 5 for prompt length
            candidate_resp = next(
                (r for r in c.candidate.responses if r.id == id), 
                None
            )
            if candidate_resp:
                reason = candidate_resp.failure_reason or "Output changed negatively"
                lines.append(f"- {id}: {reason[:150]}")
        
        if len(c.regressions) > 5:
            lines.append(f"- ... and {len(c.regressions) - 5} more regressions")
        
        return "\n".join(lines) if lines else "None detected."
    
    def _format_improvements(self, c: Comparison) -> str:
        """Format improvement details for the prompt."""
        if not c.improvements:
            return "None detected."
        
        lines = []
        for id in c.improvements[:3]:  # Cap at 3
            baseline_resp = next(
                (r for r in c.baseline.responses if r.id == id), 
                None
            )
            if baseline_resp and baseline_resp.failure_reason:
                lines.append(f"- {id}: Fixed - {baseline_resp.failure_reason[:100]}")
            else:
                lines.append(f"- {id}: Improved")
        
        if len(c.improvements) > 3:
            lines.append(f"- ... and {len(c.improvements) - 3} more improvements")
        
        return "\n".join(lines) if lines else "None detected."
    
    def _parse_response(self, comparison: Comparison, response: str) -> Judgment:
        """Parse LLM response into structured Judgment.
        
        Args:
            comparison: The original comparison
            response: Raw LLM response text
            
        Returns:
            Structured Judgment object
        """
        # Default values
        recommendation = "needs_review"
        risk_level = Severity.MEDIUM
        narrative = ""
        findings: list[str] = []
        actions: list[str] = []
        
        current_section = None
        
        for line in response.split("\n"):
            line = line.strip()
            
            if line.upper().startswith("RECOMMENDATION:"):
                rec = line.split(":", 1)[1].strip().lower().replace(" ", "_")
                if rec in ("ship", "do_not_ship", "needs_review"):
                    recommendation = rec
                    
            elif line.upper().startswith("RISK_LEVEL:"):
                rl = line.split(":", 1)[1].strip().upper()
                if rl in Severity.__members__:
                    risk_level = Severity[rl]
                    
            elif line.upper().startswith("SUMMARY:"):
                narrative = line.split(":", 1)[1].strip()
                
            elif line.upper().startswith("KEY_FINDINGS:"):
                current_section = "findings"
                
            elif line.upper().startswith("ACTION_ITEMS:"):
                current_section = "actions"
                
            elif line.startswith("- ") or line.startswith("• "):
                item = line[2:].strip()
                if item:
                    if current_section == "findings":
                        findings.append(item)
                    elif current_section == "actions":
                        actions.append(item)
        
        # Update comparison with recommendation
        comparison.recommendation = recommendation
        
        return Judgment(
            comparison=comparison,
            narrative=narrative,
            risk_level=risk_level,
            key_findings=findings[:5],  # Cap at 5
            action_items=actions[:5],   # Cap at 5
        )

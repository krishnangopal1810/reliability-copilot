"""Comparator for analyzing differences between eval runs."""

from dataclasses import dataclass

from .models import EvalRun, Comparison, Response
from ..providers.llm.base import LLMProvider


@dataclass
class ComparisonConfig:
    """Configuration for comparison behavior."""
    semantic_diff: bool = True      # Use LLM for semantic comparison
    strict_matching: bool = False   # Require exact ID matches
    ignore_unchanged: bool = True   # Skip identical responses


class Comparator:
    """Compares two eval runs to identify improvements and regressions."""
    
    def __init__(self, llm: LLMProvider, config: ComparisonConfig | None = None):
        self.llm = llm
        self.config = config or ComparisonConfig()
    
    def compare(self, baseline: EvalRun, candidate: EvalRun) -> Comparison:
        """Compare baseline and candidate runs.
        
        Args:
            baseline: The reference eval run (before changes)
            candidate: The new eval run (after changes)
            
        Returns:
            Comparison object with improvements, regressions, and unchanged cases
        """
        # Match responses by ID
        baseline_map = {r.id: r for r in baseline.responses}
        candidate_map = {r.id: r for r in candidate.responses}
        
        common_ids = set(baseline_map.keys()) & set(candidate_map.keys())
        
        improvements: list[str] = []
        regressions: list[str] = []
        unchanged: list[str] = []
        
        for id in common_ids:
            b, c = baseline_map[id], candidate_map[id]
            
            # Clear regression: was passing, now failing
            if b.passed and not c.passed:
                regressions.append(id)
            # Clear improvement: was failing, now passing
            elif not b.passed and c.passed:
                improvements.append(id)
            # Both passing or both failing, but output changed
            elif b.output != c.output:
                if self.config.semantic_diff:
                    change = self._semantic_compare(b, c)
                    if change == "better":
                        improvements.append(id)
                    elif change == "worse":
                        regressions.append(id)
                    else:
                        unchanged.append(id)
                else:
                    unchanged.append(id)
            else:
                unchanged.append(id)
        
        # Note: new_ids and removed_ids could be used in future for tracking
        # test case additions/removals across runs
        # new_ids = set(candidate_map.keys()) - set(baseline_map.keys())
        # removed_ids = set(baseline_map.keys()) - set(candidate_map.keys())
        
        return Comparison(
            baseline=baseline,
            candidate=candidate,
            improvements=improvements,
            regressions=regressions,
            unchanged=unchanged,
        )
    
    def _semantic_compare(self, baseline: Response, candidate: Response) -> str:
        """Use LLM to determine if a change is better, worse, or neutral.
        
        Args:
            baseline: The original response
            candidate: The new response
            
        Returns:
            One of: "better", "worse", "neutral"
        """
        prompt = f"""Compare these two AI responses for the same input.
Determine if the CANDIDATE response is better, worse, or about the same as the BASELINE.

INPUT: {baseline.input[:500]}

BASELINE OUTPUT:
{baseline.output[:1000]}

CANDIDATE OUTPUT:  
{candidate.output[:1000]}

Consider these factors:
- Accuracy: Is the information more or less correct?
- Helpfulness: Does it better address the user's needs?
- Safety: Any new risks or harmful content?
- Following instructions: Does it better follow the prompt?

Answer with exactly one word: BETTER, WORSE, or NEUTRAL

Your answer:"""

        try:
            result = self.llm.complete(prompt, max_tokens=10).strip().upper()
            # Extract just the first word in case of explanation
            first_word = result.split()[0] if result else "NEUTRAL"
            return {
                "BETTER": "better", 
                "WORSE": "worse",
                "NEUTRAL": "neutral"
            }.get(first_word, "neutral")
        except Exception:
            # If LLM fails, assume neutral
            return "neutral"

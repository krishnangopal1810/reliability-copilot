"""Core data models for Reliability Copilot."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4


class Severity(Enum):
    """Severity level for failures and risks."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Response:
    """A single test case response from an eval run."""
    id: str
    input: str
    output: str
    expected: Optional[str] = None
    passed: bool = True
    failure_reason: Optional[str] = None
    latency_ms: Optional[int] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalRun:
    """A collection of responses from a single evaluation run.
    
    Phase 0: Created transiently from JSON files.
    Phase 1+: Persisted with timestamps for longitudinal analysis.
    """
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    responses: list[Response] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = field(default_factory=dict)
    
    @property
    def failures(self) -> list[Response]:
        """Get all failed responses."""
        return [r for r in self.responses if not r.passed]
    
    @property
    def pass_rate(self) -> float:
        """Calculate pass rate as a fraction."""
        if not self.responses:
            return 0.0
        return sum(1 for r in self.responses if r.passed) / len(self.responses)


@dataclass
class FailureCluster:
    """A group of related failures."""
    id: UUID = field(default_factory=uuid4)
    label: str = ""
    description: str = ""
    severity: Severity = Severity.MEDIUM
    response_ids: list[str] = field(default_factory=list)
    
    # Phase 1+: Track cluster across runs
    first_seen: Optional[datetime] = None
    occurrence_count: int = 1


@dataclass
class Comparison:
    """Result of comparing two eval runs."""
    baseline: EvalRun
    candidate: EvalRun
    improvements: list[str] = field(default_factory=list)
    regressions: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    summary: str = ""
    recommendation: str = ""  # "ship" | "do_not_ship" | "needs_review"
    confidence: float = 0.0   # 0-1


@dataclass 
class Judgment:
    """Final narrative judgment for a comparison."""
    comparison: Comparison
    narrative: str = ""
    risk_level: Severity = Severity.MEDIUM
    key_findings: list[str] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)

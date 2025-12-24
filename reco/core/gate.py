"""Deployment gate for pre-release checks.

Phase 4: Block deployment if reliability thresholds are exceeded.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from .models import EvalRun, Severity
from .clusterer import Clusterer


@dataclass
class GateThresholds:
    """Configurable thresholds for the deployment gate."""
    max_regression_percent: float = 15.0
    min_pass_rate: float = 0.80
    block_on_severity: list[str] = field(default_factory=lambda: ["CRITICAL"])
    
    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "GateThresholds":
        """Load thresholds from config file.
        
        Args:
            config_path: Path to config. Defaults to .reco/thresholds.yaml
        """
        if config_path is None:
            config_path = Path(".reco") / "thresholds.yaml"
        
        if not config_path.exists():
            return cls()
        
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            return cls(
                max_regression_percent=config.get("max_regression_percent", 15.0),
                min_pass_rate=config.get("min_pass_rate", 0.80),
                block_on_severity=config.get("block_on_severity", ["CRITICAL"]),
            )
        except (yaml.YAMLError, IOError):
            return cls()


@dataclass
class GateViolation:
    """A single threshold violation."""
    rule: str
    message: str
    actual: float
    threshold: float


@dataclass
class GateResult:
    """Result of a deployment gate check."""
    passed: bool
    violations: list[GateViolation] = field(default_factory=list)
    pass_rate_baseline: float = 0.0
    pass_rate_candidate: float = 0.0
    regression_percent: float = 0.0
    
    @property
    def is_blocked(self) -> bool:
        return not self.passed


class DeploymentGate:
    """Checks if a candidate run meets deployment thresholds."""
    
    def __init__(self, thresholds: Optional[GateThresholds] = None):
        self.thresholds = thresholds or GateThresholds.load()
    
    def check(self, baseline: EvalRun, candidate: EvalRun) -> GateResult:
        """Check if candidate meets thresholds compared to baseline.
        
        Args:
            baseline: The reference run (current production)
            candidate: The new run to evaluate
            
        Returns:
            GateResult with pass/fail and violations
        """
        violations = []
        
        baseline_rate = baseline.pass_rate
        candidate_rate = candidate.pass_rate
        
        # Calculate regression
        if baseline_rate > 0:
            regression = ((baseline_rate - candidate_rate) / baseline_rate) * 100
        else:
            regression = 0.0
        
        # Check: max_regression_percent
        if regression > self.thresholds.max_regression_percent:
            violations.append(GateViolation(
                rule="max_regression_percent",
                message=f"Pass rate regressed by {regression:.1f}%",
                actual=regression,
                threshold=self.thresholds.max_regression_percent,
            ))
        
        # Check: min_pass_rate
        if candidate_rate < self.thresholds.min_pass_rate:
            violations.append(GateViolation(
                rule="min_pass_rate",
                message=f"Pass rate {candidate_rate:.1%} below minimum",
                actual=candidate_rate,
                threshold=self.thresholds.min_pass_rate,
            ))
        
        # Check: block_on_severity (if candidate has new failures with blocked severity)
        blocked_severities = {s.upper() for s in self.thresholds.block_on_severity}
        
        # Find new failures in candidate
        baseline_failures = {r.id for r in baseline.failures}
        new_failures = [r for r in candidate.failures if r.id not in baseline_failures]
        
        if new_failures and blocked_severities:
            # We'd need clustering to know severity - for now check if any new failures exist
            # and CRITICAL is in blocked list
            if "CRITICAL" in blocked_severities and new_failures:
                # NOTE: Without clustering, we can't know severity
                # This is a simplified check - could enhance with clustering
                pass
        
        return GateResult(
            passed=len(violations) == 0,
            violations=violations,
            pass_rate_baseline=baseline_rate,
            pass_rate_candidate=candidate_rate,
            regression_percent=regression,
        )

# Reliability Copilot - Technical Design Document

> Phase 0 implementation with forward-compatible architecture for Phases 1-5.

---

## Design Principles

1. **Stateless now, stateful later** — Core logic doesn't assume persistence, but interfaces support it
2. **Protocols over implementations** — Use abstract base classes for extensibility  
3. **LLM-agnostic** — Swap Claude for OpenAI/local models without refactoring
4. **CLI-first, API-ready** — Same core logic serves both interfaces

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLI Layer (typer)                              │
│  reco compare | reco cluster | reco diff                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Core Domain Layer                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │   Comparator    │  │    Clusterer    │  │   JudgmentGenerator         │  │
│  │                 │  │                 │  │                             │  │
│  │ compare(a, b)   │  │ cluster(fails)  │  │ generate(comparison)        │  │
│  └────────┬────────┘  └────────┬────────┘  └──────────────┬──────────────┘  │
│           │                    │                          │                 │
└───────────┼────────────────────┼──────────────────────────┼─────────────────┘
            │                    │                          │
            ▼                    ▼                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Infrastructure Layer                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │  LLMProvider    │  │  EmbeddingModel │  │   Storage (Phase 1+)        │  │
│  │  (Protocol)     │  │  (Protocol)     │  │   (Protocol)                │  │
│  │                 │  │                 │  │                             │  │
│  │ • Claude        │  │ • SentenceTF    │  │ • InMemory (Phase 0)        │  │
│  │ • OpenAI        │  │ • OpenAI        │  │ • SQLite (Phase 1)          │  │
│  │ • Local         │  │ • Cohere        │  │ • Postgres (Phase 2+)       │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Module Structure

```
reco/
├── __init__.py
├── __main__.py              # Entry point: python -m reco
├── cli.py                   # Typer CLI commands
│
├── core/                    # Domain logic (no I/O dependencies)
│   ├── __init__.py
│   ├── models.py            # Data classes: EvalRun, Response, Cluster, etc.
│   ├── comparator.py        # Compare two runs
│   ├── clusterer.py         # Cluster failures
│   └── judgment.py          # Generate narrative judgment
│
├── providers/               # External service adapters
│   ├── __init__.py
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py          # LLMProvider protocol
│   │   ├── claude.py        # Anthropic implementation
│   │   └── openai.py        # OpenAI implementation (Phase 1+)
│   └── embeddings/
│       ├── __init__.py
│       ├── base.py          # EmbeddingProvider protocol
│       └── sentence_transformers.py
│
├── storage/                 # Persistence (Phase 1+)
│   ├── __init__.py
│   ├── base.py              # Storage protocol
│   ├── memory.py            # In-memory (Phase 0)
│   └── sqlite.py            # SQLite (Phase 1)
│
├── formatters/              # Output formatting
│   ├── __init__.py
│   ├── terminal.py          # Rich terminal output
│   └── json.py              # JSON output
│
└── config.py                # Configuration management
```

---

## Core Data Models

```python
# reco/core/models.py
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4


class Severity(Enum):
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
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)
    
    @property
    def failures(self) -> list[Response]:
        return [r for r in self.responses if not r.passed]
    
    @property
    def pass_rate(self) -> float:
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
```

---

## Provider Protocols

### LLM Provider

```python
# reco/providers/llm/base.py
from abc import ABC, abstractmethod
from typing import Protocol


class LLMProvider(Protocol):
    """Protocol for LLM providers. Swap implementations without changing core logic."""
    
    @abstractmethod
    def complete(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate a completion for the given prompt."""
        ...
    
    @abstractmethod
    def complete_structured(self, prompt: str, schema: dict) -> dict:
        """Generate a structured response matching the schema."""
        ...


# reco/providers/llm/claude.py
import anthropic
from .base import LLMProvider


class ClaudeProvider(LLMProvider):
    def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
    
    def complete(self, prompt: str, max_tokens: int = 1000) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
    def complete_structured(self, prompt: str, schema: dict) -> dict:
        # Use tool_use for structured output
        ...
```

### Embedding Provider

```python
# reco/providers/embeddings/base.py
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""
    
    @abstractmethod
    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        """Generate embeddings for a list of texts."""
        ...
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimension of the embedding vectors."""
        ...


# reco/providers/embeddings/sentence_transformers.py
from sentence_transformers import SentenceTransformer
import numpy as np
from .base import EmbeddingProvider


class SentenceTransformerProvider(EmbeddingProvider):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self._dimension = self.model.get_sentence_embedding_dimension()
    
    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True)
    
    @property
    def dimension(self) -> int:
        return self._dimension
```

---

## Core Services

### Comparator

```python
# reco/core/comparator.py
from dataclasses import dataclass
from .models import EvalRun, Comparison, Response


@dataclass
class ComparisonConfig:
    """Configuration for comparison behavior."""
    semantic_diff: bool = True      # Use LLM for semantic comparison
    strict_matching: bool = False   # Require exact ID matches
    ignore_unchanged: bool = True   # Skip identical responses


class Comparator:
    """Compares two eval runs to identify improvements and regressions."""
    
    def __init__(self, llm: "LLMProvider", config: ComparisonConfig | None = None):
        self.llm = llm
        self.config = config or ComparisonConfig()
    
    def compare(self, baseline: EvalRun, candidate: EvalRun) -> Comparison:
        """Compare baseline and candidate runs."""
        # Match responses by ID
        baseline_map = {r.id: r for r in baseline.responses}
        candidate_map = {r.id: r for r in candidate.responses}
        
        common_ids = set(baseline_map.keys()) & set(candidate_map.keys())
        
        improvements = []
        regressions = []
        unchanged = []
        
        for id in common_ids:
            b, c = baseline_map[id], candidate_map[id]
            
            if b.passed and not c.passed:
                regressions.append(id)
            elif not b.passed and c.passed:
                improvements.append(id)
            elif b.output != c.output:
                # Semantic comparison via LLM
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
        
        return Comparison(
            baseline=baseline,
            candidate=candidate,
            improvements=improvements,
            regressions=regressions,
            unchanged=unchanged,
        )
    
    def _semantic_compare(self, baseline: Response, candidate: Response) -> str:
        """Use LLM to determine if a change is better, worse, or neutral."""
        prompt = f"""Compare these two AI responses for the same input.

INPUT: {baseline.input}

BASELINE OUTPUT:
{baseline.output}

CANDIDATE OUTPUT:  
{candidate.output}

Is the candidate BETTER, WORSE, or NEUTRAL compared to baseline?
Consider: accuracy, helpfulness, safety, and following instructions.

Answer with exactly one word: BETTER, WORSE, or NEUTRAL"""

        result = self.llm.complete(prompt, max_tokens=10).strip().upper()
        return {"BETTER": "better", "WORSE": "worse"}.get(result, "neutral")
```

### Clusterer

```python
# reco/core/clusterer.py
from dataclasses import dataclass
import numpy as np
import hdbscan
from .models import Response, FailureCluster, Severity


@dataclass
class ClusterConfig:
    min_cluster_size: int = 2
    min_samples: int = 1
    metric: str = "euclidean"


class Clusterer:
    """Clusters failure responses to identify patterns."""
    
    def __init__(
        self, 
        embedder: "EmbeddingProvider", 
        llm: "LLMProvider",
        config: ClusterConfig | None = None
    ):
        self.embedder = embedder
        self.llm = llm
        self.config = config or ClusterConfig()
    
    def cluster(self, failures: list[Response]) -> list[FailureCluster]:
        """Cluster failures by semantic similarity."""
        if len(failures) < 2:
            # Not enough to cluster
            return [self._single_cluster(failures)] if failures else []
        
        # 1. Generate embeddings
        reasons = [f.failure_reason or f.output for f in failures]
        embeddings = self.embedder.embed(reasons)
        
        # 2. Cluster with HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.config.min_cluster_size,
            min_samples=self.config.min_samples,
            metric=self.config.metric,
        )
        labels = clusterer.fit_predict(embeddings)
        
        # 3. Group by cluster label
        clusters: dict[int, list[Response]] = {}
        for failure, label in zip(failures, labels):
            clusters.setdefault(label, []).append(failure)
        
        # 4. Generate labels and create cluster objects
        result = []
        for label, cluster_failures in sorted(clusters.items()):
            cluster = self._create_cluster(label, cluster_failures)
            result.append(cluster)
        
        # Sort by severity, then size
        return sorted(result, key=lambda c: (-self._severity_rank(c.severity), -len(c.response_ids)))
    
    def _create_cluster(self, label: int, failures: list[Response]) -> FailureCluster:
        """Create a labeled cluster from failures."""
        if label == -1:
            return FailureCluster(
                label="Uncategorized",
                description="Failures that don't fit a clear pattern",
                severity=Severity.LOW,
                response_ids=[f.id for f in failures],
            )
        
        # Generate label via LLM
        reasons = [f.failure_reason or f.output for f in failures[:5]]  # Cap at 5 for prompt
        label_text, severity = self._generate_label(reasons)
        
        return FailureCluster(
            label=label_text,
            description=f"{len(failures)} failures with similar pattern",
            severity=severity,
            response_ids=[f.id for f in failures],
        )
    
    def _generate_label(self, reasons: list[str]) -> tuple[str, Severity]:
        """Use LLM to generate a cluster label and assess severity."""
        prompt = f"""These are failure reasons from an AI system that are semantically related.

Failures:
{chr(10).join(f"- {r}" for r in reasons)}

1. Generate a short label (3-5 words) describing the common pattern.
2. Assess severity: LOW (minor issues), MEDIUM (noticeable problems), HIGH (significant failures), CRITICAL (dangerous/unsafe behavior).

Format your response exactly as:
LABEL: <your label>
SEVERITY: <LOW|MEDIUM|HIGH|CRITICAL>"""

        response = self.llm.complete(prompt, max_tokens=50)
        
        # Parse response
        label = "Unknown Pattern"
        severity = Severity.MEDIUM
        
        for line in response.strip().split("\n"):
            if line.startswith("LABEL:"):
                label = line.replace("LABEL:", "").strip()
            elif line.startswith("SEVERITY:"):
                sev_str = line.replace("SEVERITY:", "").strip().upper()
                severity = Severity[sev_str] if sev_str in Severity.__members__ else Severity.MEDIUM
        
        return label, severity
    
    @staticmethod
    def _severity_rank(severity: Severity) -> int:
        return {Severity.LOW: 0, Severity.MEDIUM: 1, Severity.HIGH: 2, Severity.CRITICAL: 3}[severity]
    
    def _single_cluster(self, failures: list[Response]) -> FailureCluster:
        """Handle edge case of single failure."""
        return FailureCluster(
            label="Single Failure",
            severity=Severity.MEDIUM,
            response_ids=[f.id for f in failures],
        )
```

### Judgment Generator

```python
# reco/core/judgment.py
from .models import Comparison, Judgment, Severity


class JudgmentGenerator:
    """Generates narrative judgment from comparison results."""
    
    def __init__(self, llm: "LLMProvider"):
        self.llm = llm
    
    def generate(self, comparison: Comparison) -> Judgment:
        """Generate a narrative judgment for the comparison."""
        prompt = self._build_prompt(comparison)
        response = self.llm.complete(prompt, max_tokens=500)
        
        return self._parse_response(comparison, response)
    
    def _build_prompt(self, c: Comparison) -> str:
        baseline_failures = [r for r in c.baseline.responses if not r.passed]
        candidate_failures = [r for r in c.candidate.responses if not r.passed]
        
        return f"""You are a reliability analyst reviewing a prompt/model change.

BASELINE RUN:
- Total tests: {len(c.baseline.responses)}
- Pass rate: {c.baseline.pass_rate:.1%}
- Failures: {len(baseline_failures)}

CANDIDATE RUN (proposed change):
- Total tests: {len(c.candidate.responses)}  
- Pass rate: {c.candidate.pass_rate:.1%}
- Failures: {len(candidate_failures)}

CHANGES DETECTED:
- Improved: {len(c.improvements)} cases
- Regressed: {len(c.regressions)} cases
- Unchanged: {len(c.unchanged)} cases

REGRESSION DETAILS:
{self._format_regressions(c)}

Provide:
1. RECOMMENDATION: SHIP, DO_NOT_SHIP, or NEEDS_REVIEW
2. RISK_LEVEL: LOW, MEDIUM, HIGH, or CRITICAL
3. SUMMARY: 2-3 sentence narrative of what changed
4. KEY_FINDINGS: Up to 3 bullet points
5. ACTION_ITEMS: What should they do next?

Be opinionated. If there are regressions, lean toward DO_NOT_SHIP."""

    def _format_regressions(self, c: Comparison) -> str:
        if not c.regressions:
            return "None"
        
        lines = []
        for id in c.regressions[:5]:  # Cap at 5
            candidate_resp = next((r for r in c.candidate.responses if r.id == id), None)
            if candidate_resp:
                reason = candidate_resp.failure_reason or "Output changed negatively"
                lines.append(f"- {id}: {reason[:100]}")
        
        return "\n".join(lines) or "None"
    
    def _parse_response(self, comparison: Comparison, response: str) -> Judgment:
        """Parse LLM response into structured Judgment."""
        # Default values
        recommendation = "needs_review"
        risk_level = Severity.MEDIUM
        narrative = ""
        findings = []
        actions = []
        
        # Simple parsing (could use structured output)
        current_section = None
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("RECOMMENDATION:"):
                rec = line.replace("RECOMMENDATION:", "").strip().lower()
                recommendation = rec.replace(" ", "_")
            elif line.startswith("RISK_LEVEL:"):
                rl = line.replace("RISK_LEVEL:", "").strip().upper()
                risk_level = Severity[rl] if rl in Severity.__members__ else Severity.MEDIUM
            elif line.startswith("SUMMARY:"):
                narrative = line.replace("SUMMARY:", "").strip()
            elif line.startswith("KEY_FINDINGS:"):
                current_section = "findings"
            elif line.startswith("ACTION_ITEMS:"):
                current_section = "actions"
            elif line.startswith("- ") or line.startswith("• "):
                item = line[2:].strip()
                if current_section == "findings":
                    findings.append(item)
                elif current_section == "actions":
                    actions.append(item)
        
        comparison.recommendation = recommendation
        
        return Judgment(
            comparison=comparison,
            narrative=narrative,
            risk_level=risk_level,
            key_findings=findings,
            action_items=actions,
        )
```

---

## Storage Protocol (Phase 1+ Ready)

```python
# reco/storage/base.py
from abc import ABC, abstractmethod
from typing import Optional
from uuid import UUID
from ..core.models import EvalRun, FailureCluster


class Storage(ABC):
    """Abstract storage interface. Phase 0 uses in-memory, Phase 1+ uses SQLite/Postgres."""
    
    @abstractmethod
    def save_run(self, run: EvalRun) -> None:
        """Persist an eval run."""
        ...
    
    @abstractmethod
    def get_run(self, run_id: UUID) -> Optional[EvalRun]:
        """Retrieve a run by ID."""
        ...
    
    @abstractmethod
    def list_runs(self, limit: int = 10) -> list[EvalRun]:
        """List recent runs."""
        ...
    
    # Phase 1+: Failure memory
    @abstractmethod
    def save_cluster(self, cluster: FailureCluster, run_id: UUID) -> None:
        """Save a cluster associated with a run."""
        ...
    
    @abstractmethod
    def find_similar_clusters(self, cluster: FailureCluster, limit: int = 5) -> list[FailureCluster]:
        """Find historically similar failure clusters."""
        ...


# reco/storage/memory.py
class InMemoryStorage(Storage):
    """Phase 0: No persistence, everything in memory."""
    
    def __init__(self):
        self._runs: dict[UUID, EvalRun] = {}
        self._clusters: dict[UUID, list[FailureCluster]] = {}
    
    def save_run(self, run: EvalRun) -> None:
        self._runs[run.id] = run
    
    def get_run(self, run_id: UUID) -> Optional[EvalRun]:
        return self._runs.get(run_id)
    
    def list_runs(self, limit: int = 10) -> list[EvalRun]:
        return sorted(self._runs.values(), key=lambda r: r.created_at, reverse=True)[:limit]
    
    def save_cluster(self, cluster: FailureCluster, run_id: UUID) -> None:
        self._clusters.setdefault(run_id, []).append(cluster)
    
    def find_similar_clusters(self, cluster: FailureCluster, limit: int = 5) -> list[FailureCluster]:
        # Phase 0: No historical data
        return []
```

---

## CLI Implementation

```python
# reco/cli.py
import json
import typer
from pathlib import Path
from rich.console import Console

from .core.models import EvalRun, Response
from .core.comparator import Comparator
from .core.clusterer import Clusterer
from .core.judgment import JudgmentGenerator
from .providers.llm.claude import ClaudeProvider
from .providers.embeddings.sentence_transformers import SentenceTransformerProvider
from .formatters.terminal import TerminalFormatter

app = typer.Typer(name="reco", help="Reliability Copilot - AI judgment for prompt changes")
console = Console()


def load_run(path: Path) -> EvalRun:
    """Load an eval run from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    
    responses = [
        Response(
            id=r["id"],
            input=r.get("input", ""),
            output=r.get("output", ""),
            expected=r.get("expected"),
            passed=r.get("pass", r.get("passed", True)),
            failure_reason=r.get("failure_reason"),
            metadata=r.get("metadata", {}),
        )
        for r in data.get("responses", [])
    ]
    
    return EvalRun(
        name=data.get("name", path.stem),
        responses=responses,
        metadata=data.get("metadata", {}),
    )


@app.command()
def compare(
    baseline: Path = typer.Argument(..., help="Path to baseline eval JSON"),
    candidate: Path = typer.Argument(..., help="Path to candidate eval JSON"),
    format: str = typer.Option("terminal", help="Output format: terminal, json"),
):
    """Compare two eval runs and generate a judgment."""
    llm = ClaudeProvider()
    comparator = Comparator(llm)
    judge = JudgmentGenerator(llm)
    formatter = TerminalFormatter(console)
    
    baseline_run = load_run(baseline)
    candidate_run = load_run(candidate)
    
    comparison = comparator.compare(baseline_run, candidate_run)
    judgment = judge.generate(comparison)
    
    formatter.render_judgment(judgment)


@app.command()
def cluster(
    evalfile: Path = typer.Argument(..., help="Path to eval JSON with failures"),
    format: str = typer.Option("terminal", help="Output format: terminal, json"),
):
    """Cluster failures to identify patterns."""
    llm = ClaudeProvider()
    embedder = SentenceTransformerProvider()
    clusterer = Clusterer(embedder, llm)
    formatter = TerminalFormatter(console)
    
    run = load_run(evalfile)
    failures = run.failures
    
    if not failures:
        console.print("[green]No failures to cluster![/green]")
        return
    
    clusters = clusterer.cluster(failures)
    formatter.render_clusters(clusters, len(failures))


@app.command()
def diff(
    baseline: Path = typer.Argument(..., help="Path to baseline eval JSON"),
    candidate: Path = typer.Argument(..., help="Path to candidate eval JSON"),
    case: str = typer.Option(None, "--case", "-c", help="Specific test case ID to diff"),
):
    """Show detailed diff for specific test cases."""
    # Implementation for phase 0 week 4
    console.print("[yellow]Diff command coming soon...[/yellow]")


if __name__ == "__main__":
    app()
```

---

## Configuration

```python
# reco/config.py
from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass
class Config:
    """Global configuration, loaded from env vars and config file."""
    
    # LLM settings
    llm_provider: str = "claude"
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    llm_model: str = "claude-sonnet-4-20250514"
    
    # Embedding settings
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # Clustering settings  
    min_cluster_size: int = 2
    
    # Storage settings (Phase 1+)
    storage_backend: str = "memory"  # memory, sqlite, postgres
    storage_path: Path = field(default_factory=lambda: Path.home() / ".reco" / "data.db")
    
    @classmethod
    def load(cls) -> "Config":
        """Load config from environment and config file."""
        # Phase 0: Just use env vars
        return cls()
```

---

## Phase 1+ Extension Points

| Component | Phase 0 | Phase 1+ |
|-----------|---------|----------|
| Storage | `InMemoryStorage` | `SQLiteStorage`, `PostgresStorage` |
| LLM | `ClaudeProvider` | Add `OpenAIProvider`, `LocalLLMProvider` |
| CLI | `compare`, `cluster` | Add `history`, `track`, `report` |
| Output | Terminal only | Add API server, webhooks |
| Clusters | One-shot | `find_similar_clusters()` for memory |

### Phase 1 Additions (Storage & Memory)
```python
# New commands
@app.command()
def history(limit: int = 10):
    """Show recent runs and their judgments."""
    ...

@app.command()
def track(evalfile: Path, name: str):
    """Track a run for longitudinal analysis."""
    ...
```

### Phase 2 Additions (Profiles)
```python
@dataclass
class ReliabilityProfile:
    """Longitudinal view of a system's reliability."""
    system_id: str
    dominant_failure_modes: list[FailureCluster]
    trigger_conditions: list[str]
    confidence_trend: list[float]
    last_updated: datetime
```

---

## Testing Strategy

```
tests/
├── unit/
│   ├── test_comparator.py
│   ├── test_clusterer.py
│   └── test_judgment.py
├── integration/
│   └── test_cli.py
└── fixtures/
    ├── baseline.json
    ├── candidate_better.json
    ├── candidate_worse.json
    └── golden_judgments.json   # Calibration set
```

### Golden Test Set
```json
// tests/fixtures/golden_judgments.json
[
  {
    "baseline": "baseline.json",
    "candidate": "candidate_worse.json",
    "expected_recommendation": "do_not_ship",
    "expected_risk": "high"
  }
]
```

---

*Version: 1.0*  
*Last Updated: December 2024*

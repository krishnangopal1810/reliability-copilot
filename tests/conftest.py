"""Shared test fixtures and mock providers."""

import pytest
import numpy as np
from typing import Any
from uuid import uuid4

from reco.core.models import Response, EvalRun, FailureCluster, Comparison, Judgment, Severity


# ============================================================================
# Mock Providers
# ============================================================================

class MockLLMProvider:
    """Mock LLM provider for testing without API calls."""
    
    def __init__(self, responses: dict[str, str] | None = None):
        """
        Args:
            responses: Dict mapping prompt substrings to responses.
                      If a prompt contains a key, that response is returned.
        """
        self.responses = responses or {}
        self.call_history: list[dict[str, Any]] = []
        self.default_response = "NEUTRAL"
    
    def complete(self, prompt: str, max_tokens: int = 1000) -> str:
        """Return mock completion based on prompt content."""
        self.call_history.append({"prompt": prompt, "max_tokens": max_tokens})
        
        # Check for matching response
        for key, response in self.responses.items():
            if key.lower() in prompt.lower():
                return response
        
        return self.default_response
    
    def complete_structured(self, prompt: str, schema: dict) -> dict:
        """Return mock structured response."""
        self.call_history.append({"prompt": prompt, "schema": schema})
        return {"result": "mock"}
    
    def set_default_response(self, response: str) -> None:
        """Set the default response for prompts that don't match."""
        self.default_response = response


class MockEmbeddingProvider:
    """Mock embedding provider for testing without model loading."""
    
    def __init__(self, dimension: int = 384):
        self._dimension = dimension
        self.call_history: list[list[str]] = []
        self._embeddings_map: dict[str, np.ndarray] = {}
    
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return deterministic embeddings based on text content."""
        self.call_history.append(texts)
        
        embeddings = []
        for text in texts:
            if text in self._embeddings_map:
                embeddings.append(self._embeddings_map[text])
            else:
                # Generate deterministic embedding from text hash
                np.random.seed(hash(text) % 2**32)
                emb = np.random.randn(self._dimension).astype(np.float32)
                embeddings.append(emb)
        
        return np.array(embeddings, dtype=np.float32)
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    def set_similar_embeddings(self, texts: list[str]) -> None:
        """Make these texts return similar embeddings (for clustering tests)."""
        base = np.random.randn(self._dimension).astype(np.float32)
        for i, text in enumerate(texts):
            # Add small noise so they're similar but not identical
            noise = np.random.randn(self._dimension).astype(np.float32) * 0.01
            self._embeddings_map[text] = base + noise


# ============================================================================
# Fixtures: Responses
# ============================================================================

@pytest.fixture
def passing_response() -> Response:
    """A single passing response."""
    return Response(
        id="test_001",
        input="What is 2+2?",
        output="4",
        expected="4",
        passed=True,
    )


@pytest.fixture
def failing_response() -> Response:
    """A single failing response."""
    return Response(
        id="test_002",
        input="What is the capital of France?",
        output="London",
        expected="Paris",
        passed=False,
        failure_reason="Incorrect answer: said London instead of Paris",
    )


@pytest.fixture
def response_with_metadata() -> Response:
    """A response with all optional fields populated."""
    return Response(
        id="test_003",
        input="Complex query",
        output="Complex response",
        expected="Expected output",
        passed=True,
        failure_reason=None,
        latency_ms=150,
        metadata={"model": "gpt-4", "temperature": 0.7},
    )


# ============================================================================
# Fixtures: EvalRuns
# ============================================================================

@pytest.fixture
def empty_eval_run() -> EvalRun:
    """An eval run with no responses."""
    return EvalRun(name="empty_run", responses=[])


@pytest.fixture
def all_passing_run() -> EvalRun:
    """An eval run where all tests pass."""
    return EvalRun(
        name="all_passing",
        responses=[
            Response(id="p1", input="Q1", output="A1", passed=True),
            Response(id="p2", input="Q2", output="A2", passed=True),
            Response(id="p3", input="Q3", output="A3", passed=True),
        ],
    )


@pytest.fixture
def all_failing_run() -> EvalRun:
    """An eval run where all tests fail."""
    return EvalRun(
        name="all_failing",
        responses=[
            Response(id="f1", input="Q1", output="Wrong", passed=False, failure_reason="Incorrect"),
            Response(id="f2", input="Q2", output="Wrong", passed=False, failure_reason="Incorrect"),
            Response(id="f3", input="Q3", output="Wrong", passed=False, failure_reason="Incorrect"),
        ],
    )


@pytest.fixture
def mixed_run() -> EvalRun:
    """An eval run with mixed pass/fail."""
    return EvalRun(
        name="mixed",
        responses=[
            Response(id="m1", input="Q1", output="A1", passed=True),
            Response(id="m2", input="Q2", output="Wrong", passed=False, failure_reason="Error"),
            Response(id="m3", input="Q3", output="A3", passed=True),
            Response(id="m4", input="Q4", output="Wrong", passed=False, failure_reason="Error"),
        ],
    )


@pytest.fixture
def baseline_run() -> EvalRun:
    """Baseline run for comparison tests."""
    return EvalRun(
        name="baseline",
        responses=[
            Response(id="t1", input="Q1", output="A1", passed=True),
            Response(id="t2", input="Q2", output="A2", passed=True),
            Response(id="t3", input="Q3", output="A3", passed=True),
            Response(id="t4", input="Q4", output="A4", passed=False, failure_reason="Known issue"),
            Response(id="t5", input="Q5", output="A5", passed=True),
        ],
    )


@pytest.fixture
def candidate_run_better() -> EvalRun:
    """Candidate run that is better than baseline."""
    return EvalRun(
        name="candidate_better",
        responses=[
            Response(id="t1", input="Q1", output="A1", passed=True),
            Response(id="t2", input="Q2", output="A2", passed=True),
            Response(id="t3", input="Q3", output="A3", passed=True),
            Response(id="t4", input="Q4", output="Fixed!", passed=True),  # Fixed!
            Response(id="t5", input="Q5", output="A5", passed=True),
        ],
    )


@pytest.fixture
def candidate_run_worse() -> EvalRun:
    """Candidate run that is worse than baseline."""
    return EvalRun(
        name="candidate_worse",
        responses=[
            Response(id="t1", input="Q1", output="Broken", passed=False, failure_reason="Regression"),
            Response(id="t2", input="Q2", output="A2", passed=True),
            Response(id="t3", input="Q3", output="Broken", passed=False, failure_reason="Regression"),
            Response(id="t4", input="Q4", output="A4", passed=False, failure_reason="Known issue"),
            Response(id="t5", input="Q5", output="A5", passed=True),
        ],
    )


# ============================================================================
# Fixtures: Failures for Clustering
# ============================================================================

@pytest.fixture
def failures_for_clustering() -> list[Response]:
    """A set of failures suitable for clustering tests."""
    return [
        # Cluster 1: Financial hallucinations
        Response(id="c1_1", input="Q", output="O", passed=False, 
                 failure_reason="Hallucinated revenue figures"),
        Response(id="c1_2", input="Q", output="O", passed=False,
                 failure_reason="Made up financial data"),
        Response(id="c1_3", input="Q", output="O", passed=False,
                 failure_reason="Invented earnings numbers"),
        # Cluster 2: Format violations
        Response(id="c2_1", input="Q", output="O", passed=False,
                 failure_reason="Did not return JSON format"),
        Response(id="c2_2", input="Q", output="O", passed=False,
                 failure_reason="Ignored JSON formatting requirement"),
        # Outlier
        Response(id="outlier", input="Q", output="O", passed=False,
                 failure_reason="Random unrelated error"),
    ]


# ============================================================================
# Fixtures: Mock Providers
# ============================================================================

@pytest.fixture
def mock_llm() -> MockLLMProvider:
    """A mock LLM provider."""
    return MockLLMProvider()


@pytest.fixture
def mock_llm_with_responses() -> MockLLMProvider:
    """A mock LLM provider with preset responses."""
    return MockLLMProvider(responses={
        "better": "BETTER",
        "worse": "WORSE",
        "neutral": "NEUTRAL",
        "label": "LABEL: Test Pattern\nSEVERITY: HIGH",
        "ship": "RECOMMENDATION: SHIP\nRISK_LEVEL: LOW\nSUMMARY: All good.",
        "do_not_ship": "RECOMMENDATION: DO_NOT_SHIP\nRISK_LEVEL: HIGH\nSUMMARY: Has issues.",
    })


@pytest.fixture
def mock_embedder() -> MockEmbeddingProvider:
    """A mock embedding provider."""
    return MockEmbeddingProvider()


# ============================================================================
# Fixtures: Clusters
# ============================================================================

@pytest.fixture
def sample_cluster() -> FailureCluster:
    """A sample failure cluster."""
    return FailureCluster(
        label="Test Cluster",
        description="A test cluster",
        severity=Severity.MEDIUM,
        response_ids=["t1", "t2", "t3"],
    )


# ============================================================================
# Fixtures: Comparisons and Judgments
# ============================================================================

@pytest.fixture
def sample_comparison(baseline_run: EvalRun, candidate_run_worse: EvalRun) -> Comparison:
    """A sample comparison object."""
    return Comparison(
        baseline=baseline_run,
        candidate=candidate_run_worse,
        improvements=[],
        regressions=["t1", "t3"],
        unchanged=["t2", "t4", "t5"],
        recommendation="do_not_ship",
    )


@pytest.fixture
def sample_judgment(sample_comparison: Comparison) -> Judgment:
    """A sample judgment object."""
    return Judgment(
        comparison=sample_comparison,
        narrative="The candidate has regressions.",
        risk_level=Severity.HIGH,
        key_findings=["2 new failures detected"],
        action_items=["Fix t1 and t3 before shipping"],
    )

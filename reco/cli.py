"""CLI commands for Reliability Copilot."""

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from .config import Config
from .core.models import EvalRun, Response
from .core.comparator import Comparator, ComparisonConfig
from .core.clusterer import Clusterer, ClusterConfig
from .core.judgment import JudgmentGenerator
from .providers.llm.openrouter import OpenRouterProvider
from .providers.embeddings.sentence_transformers import SentenceTransformerProvider
from .formatters.terminal import TerminalFormatter

app = typer.Typer(
    name="reco",
    help="Reliability Copilot - AI judgment for prompt changes",
    add_completion=False,
)
console = Console()
formatter = TerminalFormatter(console)


def load_run(path: Path) -> EvalRun:
    """Load an eval run from a JSON file.
    
    Expected format:
    {
        "name": "optional run name",
        "responses": [
            {
                "id": "test_001",
                "input": "user prompt",
                "output": "model response",
                "expected": "optional expected output",
                "pass": true,
                "failure_reason": "optional reason if pass is false"
            }
        ]
    }
    """
    if not path.exists():
        raise typer.BadParameter(f"File not found: {path}")
    
    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise typer.BadParameter(f"Invalid JSON in {path}: {e}")
    
    responses = []
    for r in data.get("responses", []):
        responses.append(Response(
            id=r.get("id", f"test_{len(responses)}"),
            input=r.get("input", ""),
            output=r.get("output", ""),
            expected=r.get("expected"),
            passed=r.get("pass", r.get("passed", True)),
            failure_reason=r.get("failure_reason"),
            latency_ms=r.get("latency_ms"),
            metadata=r.get("metadata", {}),
        ))
    
    return EvalRun(
        name=data.get("name", path.stem),
        responses=responses,
        metadata=data.get("metadata", {}),
    )


def get_llm() -> OpenRouterProvider:
    """Get configured LLM provider."""
    config = Config.load()
    errors = config.validate()
    if errors:
        for error in errors:
            formatter.render_error(error)
        raise typer.Exit(1)
    
    return OpenRouterProvider(model=config.llm_model)


@app.command()
def compare(
    baseline: Path = typer.Argument(
        ..., 
        help="Path to baseline eval JSON (before changes)",
        exists=True,
    ),
    candidate: Path = typer.Argument(
        ..., 
        help="Path to candidate eval JSON (after changes)",
        exists=True,
    ),
    no_semantic: bool = typer.Option(
        False,
        "--no-semantic",
        help="Disable semantic comparison (faster, less accurate)",
    ),
):
    """Compare two eval runs and generate a judgment.
    
    Analyzes the differences between a baseline and candidate run,
    identifies improvements and regressions, and provides an
    opinionated recommendation on whether to ship.
    
    Example:
        reco compare baseline.json candidate.json
    """
    console.print("[dim]Loading eval runs...[/dim]")
    
    baseline_run = load_run(baseline)
    candidate_run = load_run(candidate)
    
    console.print(f"[dim]Baseline: {len(baseline_run.responses)} responses[/dim]")
    console.print(f"[dim]Candidate: {len(candidate_run.responses)} responses[/dim]")
    console.print()
    
    llm = get_llm()
    
    # Compare
    console.print("[dim]Analyzing differences...[/dim]")
    config = ComparisonConfig(semantic_diff=not no_semantic)
    comparator = Comparator(llm, config)
    comparison = comparator.compare(baseline_run, candidate_run)
    
    # Generate judgment
    console.print("[dim]Generating judgment...[/dim]")
    judge = JudgmentGenerator(llm)
    judgment = judge.generate(comparison)
    
    console.print()
    formatter.render_judgment(judgment)


@app.command()
def cluster(
    evalfile: Path = typer.Argument(
        ...,
        help="Path to eval JSON with failures to cluster",
        exists=True,
    ),
    min_size: int = typer.Option(
        2,
        "--min-size", "-m",
        help="Minimum failures to form a cluster",
    ),
):
    """Cluster failures to identify patterns.
    
    Groups similar failures together to help identify
    systemic issues rather than individual errors.
    
    Example:
        reco cluster eval_results.json
    """
    console.print("[dim]Loading eval run...[/dim]")
    
    run = load_run(evalfile)
    failures = run.failures
    
    if not failures:
        formatter.render_success("No failures to cluster!")
        return
    
    console.print(f"[dim]Found {len(failures)} failures[/dim]")
    console.print()
    
    llm = get_llm()
    
    console.print("[dim]Generating embeddings...[/dim]")
    embedder = SentenceTransformerProvider()
    
    console.print("[dim]Clustering failures...[/dim]")
    config = ClusterConfig(min_cluster_size=min_size)
    clusterer = Clusterer(embedder, llm, config)
    clusters = clusterer.cluster(failures)
    
    console.print()
    formatter.render_clusters(clusters, len(failures))


@app.command()
def diff(
    baseline: Path = typer.Argument(
        ...,
        help="Path to baseline eval JSON",
        exists=True,
    ),
    candidate: Path = typer.Argument(
        ...,
        help="Path to candidate eval JSON", 
        exists=True,
    ),
    case: Optional[str] = typer.Option(
        None,
        "--case", "-c",
        help="Specific test case ID to diff",
    ),
):
    """Show detailed diff for specific test cases.
    
    Coming in Phase 0 week 4.
    """
    formatter.render_warning("Diff command coming soon...")
    
    # TODO: Implement in week 4
    # - Load both runs
    # - If case specified, show side-by-side diff for that case
    # - If no case, show summary of changed cases with option to drill down


@app.command()
def version():
    """Show version information."""
    from . import __version__
    console.print(f"reco version {__version__}")


if __name__ == "__main__":
    app()

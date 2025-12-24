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
    no_history: bool = typer.Option(
        False,
        "--no-history",
        help="Disable failure memory (don't track patterns across runs)",
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
    
    # Phase 1: Initialize storage for failure memory
    storage = None
    if not no_history:
        from .storage.sqlite import SQLiteStorage
        storage = SQLiteStorage()
        storage.save_run(run)
    
    console.print("[dim]Clustering failures...[/dim]")
    config = ClusterConfig(min_cluster_size=min_size)
    clusterer = Clusterer(embedder, llm, config, storage=storage)
    clusters = clusterer.cluster(failures, run_id=run.id)
    
    if storage:
        storage.close()
    
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
def profile(
    last_n: int = typer.Option(
        10,
        "--last-n", "-n",
        help="Number of recent runs to analyze",
    ),
):
    """Show reliability profile from run history.
    
    Displays the top failure modes across your recent cluster runs,
    helping identify persistent patterns that need attention.
    
    Example:
        reco profile
        reco profile --last-n 5
    """
    from .storage.sqlite import SQLiteStorage
    
    storage = SQLiteStorage()
    
    run_count = storage.get_run_count()
    
    if run_count == 0:
        formatter.render_warning("No runs in history yet. Run 'reco cluster' first.")
        storage.close()
        return
    
    failure_stats = storage.get_failure_mode_stats(limit_runs=last_n)
    
    storage.close()
    
    formatter.render_profile(failure_stats, run_count, last_n)


@app.command("analyze-agent")
def analyze_agent(
    tracefile: Path = typer.Argument(
        ...,
        help="Path to agent trace JSON file",
    ),
):
    """Analyze an agent trace for reliability issues.
    
    Examines multi-step agent executions to detect:
    - Tool execution failures
    - Recovery failures
    - Goal abandonment
    - Excessive retries
    
    Example:
        reco analyze-agent agent_trace.json
    """
    from .core.agent_analyzer import AgentAnalyzer, load_trace_from_file
    
    if not tracefile.exists():
        formatter.render_error(f"File not found: {tracefile}")
        raise typer.Exit(1)
    
    try:
        trace = load_trace_from_file(str(tracefile))
    except Exception as e:
        formatter.render_error(f"Failed to load trace: {e}")
        raise typer.Exit(1)
    
    analyzer = AgentAnalyzer()
    analysis = analyzer.analyze(trace)
    
    formatter.render_agent_analysis(analysis)


@app.command()
def gate(
    baseline: Path = typer.Argument(
        ...,
        help="Path to baseline eval JSON (current production)",
        exists=True,
    ),
    candidate: Path = typer.Argument(
        ...,
        help="Path to candidate eval JSON (proposed release)",
        exists=True,
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="Path to thresholds config (default: .reco/thresholds.yaml)",
    ),
):
    """Check if candidate passes deployment thresholds.
    
    Compares a candidate run against a baseline and checks
    configurable thresholds. Returns exit code 1 if blocked.
    
    Ideal for CI/CD pipelines to gate deployments.
    
    Example:
        reco gate baseline.json candidate.json
    """
    from .core.gate import DeploymentGate, GateThresholds
    
    console.print("[dim]Loading eval runs...[/dim]")
    
    baseline_run = load_run(baseline)
    candidate_run = load_run(candidate)
    
    console.print(f"[dim]Baseline: {len(baseline_run.responses)} responses, {baseline_run.pass_rate:.1%} pass[/dim]")
    console.print(f"[dim]Candidate: {len(candidate_run.responses)} responses, {candidate_run.pass_rate:.1%} pass[/dim]")
    console.print()
    
    # Load thresholds
    thresholds = GateThresholds.load(config)
    
    # Run gate check
    gate_checker = DeploymentGate(thresholds)
    result = gate_checker.check(baseline_run, candidate_run)
    
    # Render result
    formatter.render_gate(result, thresholds)
    
    # Exit code for CI
    if not result.passed:
        raise typer.Exit(1)


@app.command("import-trace")
def import_trace(
    tracefile: Path = typer.Argument(
        ...,
        help="Path to trace file (LangChain JSONL or OpenAI JSON)",
    ),
    format: str = typer.Option(
        "auto",
        "--format", "-f",
        help="Trace format: langchain, openai, or auto",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output file path (default: <input>_reco.json)",
    ),
    analyze: bool = typer.Option(
        False,
        "--analyze", "-a",
        help="Immediately analyze the converted trace",
    ),
):
    """Import and convert agent traces from frameworks.
    
    Supports:
      - LangChain callback exports (JSONL)
      - OpenAI Assistants run steps (JSON)
    
    Example:
        reco import-trace langchain_trace.jsonl --format langchain
        reco import-trace openai_run.json --format openai --analyze
    """
    import json
    from .core.trace_converter import TraceConverter
    from .core.agent_analyzer import AgentAnalyzer
    
    if not tracefile.exists():
        formatter.render_error(f"File not found: {tracefile}")
        raise typer.Exit(1)
    
    console.print(f"[dim]Importing trace from {tracefile}...[/dim]")
    
    try:
        trace = TraceConverter.convert(str(tracefile), format)
    except Exception as e:
        formatter.render_error(f"Failed to convert trace: {e}")
        raise typer.Exit(1)
    
    # Determine output path
    if output is None:
        output = tracefile.parent / f"{tracefile.stem}_reco.json"
    
    # Save converted trace
    from dataclasses import asdict
    trace_dict = {
        "id": trace.id,
        "goal": trace.goal,
        "outcome": trace.outcome,
        "steps": [
            {
                "step": s.step,
                "action": s.action,
                "input": s.input,
                "output": s.output,
                "success": s.success,
                "error": s.error,
                "duration_ms": s.duration_ms,
            }
            for s in trace.steps
        ],
        "metadata": trace.metadata,
    }
    
    with open(output, "w") as f:
        json.dump(trace_dict, f, indent=2, default=str)
    
    formatter.render_success(f"Converted trace saved to: {output}")
    console.print(f"[dim]  Steps: {len(trace.steps)}, Outcome: {trace.outcome}[/dim]")
    
    # Optionally analyze immediately
    if analyze:
        console.print()
        analyzer = AgentAnalyzer()
        analysis = analyzer.analyze(trace)
        formatter.render_agent_analysis(analysis)


@app.command()
def version():
    """Show version information."""
    from . import __version__
    console.print(f"reco version {__version__}")


if __name__ == "__main__":
    app()

"""Rich terminal output formatter."""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..core.models import Judgment, FailureCluster, Severity


class TerminalFormatter:
    """Formats output for rich terminal display."""
    
    # Severity colors
    SEVERITY_COLORS = {
        Severity.LOW: "green",
        Severity.MEDIUM: "yellow",
        Severity.HIGH: "red",
        Severity.CRITICAL: "bold red",
    }
    
    # Recommendation colors and icons
    RECOMMENDATION_STYLES = {
        "ship": ("green", "✅"),
        "do_not_ship": ("red", "❌"),
        "needs_review": ("yellow", "⚠️"),
    }
    
    def __init__(self, console: Console | None = None):
        self.console = console or Console()
    
    def render_judgment(self, judgment: Judgment) -> None:
        """Render a judgment to the terminal.
        
        Args:
            judgment: The Judgment object to display
        """
        c = judgment.comparison
        rec = c.recommendation or "needs_review"
        color, icon = self.RECOMMENDATION_STYLES.get(rec, ("yellow", "⚠️"))
        
        # Build recommendation text
        rec_display = rec.upper().replace("_", " ")
        
        # Header
        header = Text()
        header.append(f"{icon} JUDGMENT: ", style="bold")
        header.append(rec_display, style=f"bold {color}")
        
        # Build content
        lines = []
        
        # Summary stats
        lines.append("")
        lines.append(f"📊 Pass rate: {c.baseline.pass_rate:.1%} → {c.candidate.pass_rate:.1%}")
        
        delta = c.candidate.pass_rate - c.baseline.pass_rate
        delta_str = f"+{delta:.1%}" if delta >= 0 else f"{delta:.1%}"
        delta_color = "green" if delta >= 0 else "red"
        lines.append(f"   Change: [{delta_color}]{delta_str}[/{delta_color}]")
        lines.append("")
        
        # Changes breakdown
        if c.improvements:
            lines.append(f"[green]✅ IMPROVED ({len(c.improvements)} cases)[/green]")
            for id in c.improvements[:3]:
                lines.append(f"   • {id}")
            if len(c.improvements) > 3:
                lines.append(f"   • ... and {len(c.improvements) - 3} more")
            lines.append("")
        
        if c.regressions:
            lines.append(f"[red]⚠️  REGRESSED ({len(c.regressions)} cases)[/red]")
            for id in c.regressions[:3]:
                # Try to get failure reason
                resp = next((r for r in c.candidate.responses if r.id == id), None)
                reason = ""
                if resp and resp.failure_reason:
                    reason = f": {resp.failure_reason[:50]}..."
                lines.append(f"   • {id}{reason}")
            if len(c.regressions) > 3:
                lines.append(f"   • ... and {len(c.regressions) - 3} more")
            lines.append("")
        
        # Narrative summary
        if judgment.narrative:
            lines.append("[bold]📝 Summary:[/bold]")
            lines.append(f"   {judgment.narrative}")
            lines.append("")
        
        # Key findings
        if judgment.key_findings:
            lines.append("[bold]🔍 Key Findings:[/bold]")
            for finding in judgment.key_findings:
                lines.append(f"   • {finding}")
            lines.append("")
        
        # Action items
        if judgment.action_items:
            lines.append("[bold]🎯 Action Items:[/bold]")
            for action in judgment.action_items:
                lines.append(f"   • {action}")
            lines.append("")
        
        # Risk level
        risk_color = self.SEVERITY_COLORS[judgment.risk_level]
        lines.append(f"Risk Level: [{risk_color}]{judgment.risk_level.value.upper()}[/{risk_color}]")
        
        content = "\n".join(lines)
        
        panel = Panel(
            content,
            title=header,
            border_style=color,
            padding=(1, 2),
        )
        
        self.console.print(panel)
    
    def render_clusters(self, clusters: list[FailureCluster], total_failures: int) -> None:
        """Render failure clusters to the terminal.
        
        Args:
            clusters: List of FailureCluster objects
            total_failures: Total number of failures before clustering
        """
        # Header
        cluster_count = len([c for c in clusters if c.label != "Uncategorized"])
        header = Text()
        header.append("🔍 FAILURE CLUSTERS ", style="bold")
        header.append(f"({total_failures} failures → {cluster_count} patterns)", style="dim")
        
        lines = []
        
        for i, cluster in enumerate(clusters, 1):
            severity_color = self.SEVERITY_COLORS[cluster.severity]
            severity_badge = f"[{severity_color}]{cluster.severity.value.upper()}[/{severity_color}]"
            
            lines.append("")
            lines.append(f"[bold]CLUSTER {i}: {cluster.label}[/bold] ({len(cluster.response_ids)} cases)")
            lines.append(f"├─ Severity: {severity_badge}")
            
            # Phase 1: Show recurring/new status
            if cluster.is_recurring:
                first_seen_str = ""
                if cluster.first_seen:
                    first_seen_str = f" (first seen: {cluster.first_seen.strftime('%b %d')})"
                lines.append(
                    f"├─ [yellow]⚠️  RECURRING: Appeared {cluster.occurrence_count}x in recent runs{first_seen_str}[/yellow]"
                )
            elif cluster.label not in ("Uncategorized", "Single Failure"):
                lines.append("├─ [green]✨ NEW: First time seeing this pattern[/green]")
            
            if cluster.description and cluster.description != f"{len(cluster.response_ids)} failures with similar pattern":
                lines.append(f"├─ Pattern: {cluster.description[:80]}")
            
            # Show case IDs
            case_ids = cluster.response_ids[:5]
            cases_str = ", ".join(case_ids)
            if len(cluster.response_ids) > 5:
                cases_str += f", ... +{len(cluster.response_ids) - 5} more"
            lines.append(f"└─ Cases: [dim]{cases_str}[/dim]")
        
        content = "\n".join(lines)
        
        panel = Panel(
            content,
            title=header,
            border_style="blue",
            padding=(1, 2),
        )
        
        self.console.print(panel)
    
    def render_error(self, message: str, hint: str | None = None) -> None:
        """Render an error message.
        
        Args:
            message: The error message
            hint: Optional hint for resolution
        """
        self.console.print(f"[red]❌ Error:[/red] {message}")
        if hint:
            self.console.print(f"[dim]   Hint: {hint}[/dim]")
    
    def render_success(self, message: str) -> None:
        """Render a success message."""
        self.console.print(f"[green]✅[/green] {message}")
    
    def render_warning(self, message: str) -> None:
        """Render a warning message."""
        self.console.print(f"[yellow]⚠️[/yellow] {message}")
    
    def render_profile(
        self, 
        failure_stats: list[tuple[str, int, int]],
        run_count: int,
        last_n: int
    ) -> None:
        """Render reliability profile.
        
        Args:
            failure_stats: List of (label, count, total) tuples
            run_count: Total runs in history
            last_n: Number of runs analyzed
        """
        runs_shown = min(run_count, last_n)
        
        header = Text()
        header.append("📊 RELIABILITY PROFILE ", style="bold")
        header.append(f"(last {runs_shown} runs)", style="dim")
        
        lines = []
        lines.append("")
        
        if not failure_stats:
            lines.append("[dim]No failure patterns recorded yet.[/dim]")
            lines.append("[dim]Run 'reco cluster' on eval files to build history.[/dim]")
        else:
            lines.append("[bold red]🔴 TOP FAILURE MODES[/bold red]")
            lines.append("")
            
            for i, (label, count, total) in enumerate(failure_stats[:5], 1):
                percentage = (count / total * 100) if total > 0 else 0
                lines.append(f"   {i}. {label} — [bold]{count}x[/bold] ({percentage:.0f}%)")
        
        lines.append("")
        
        content = "\n".join(lines)
        
        panel = Panel(
            content,
            title=header,
            border_style="blue",
            padding=(1, 2),
        )
        
        self.console.print(panel)
    
    def render_agent_analysis(self, analysis: "AgentAnalysis") -> None:
        """Render agent trace analysis.
        
        Args:
            analysis: The agent analysis to render
        """
        from ..core.models import AgentAnalysis
        
        trace = analysis.trace
        
        header = Text()
        header.append("🤖 AGENT TRACE ANALYSIS", style="bold")
        
        lines = []
        lines.append("")
        
        # Summary section
        lines.append("[bold]📋 SUMMARY[/bold]")
        total_steps = len(trace.steps)
        success_count = sum(1 for s in trace.steps if s.success)
        failed_count = total_steps - success_count
        
        lines.append(f"   ├─ Steps: {total_steps} total ({success_count} success, {failed_count} failed)")
        lines.append(f"   ├─ Tools: {', '.join(trace.tools_used)}")
        
        outcome_color = "green" if trace.outcome == "success" else "red"
        lines.append(f"   └─ Outcome: [{outcome_color}]{trace.outcome.upper()}[/{outcome_color}]")
        lines.append("")
        
        # Issues section
        if analysis.issues:
            lines.append("[bold yellow]⚠️  ISSUES DETECTED[/bold yellow]")
            for i, issue in enumerate(analysis.issues):
                prefix = "├─" if i < len(analysis.issues) - 1 else "└─"
                step_info = f" at step {issue.step}" if issue.step else ""
                lines.append(f"   {prefix} {issue.issue_type}{step_info}")
                if issue.description:
                    lines.append(f"   │   └─ {issue.description}")
            lines.append("")
        
        # Patterns section
        if analysis.patterns:
            lines.append("[bold]🎯 PATTERNS[/bold]")
            for pattern in analysis.patterns:
                lines.append(f"   └─ {pattern}")
            lines.append("")
        
        # Recommendations section
        if analysis.recommendations:
            lines.append("[bold green]💡 RECOMMENDATIONS[/bold green]")
            for rec in analysis.recommendations:
                lines.append(f"   • {rec}")
            lines.append("")
        
        content = "\n".join(lines)
        
        panel = Panel(
            content,
            title=header,
            border_style="cyan",
            padding=(1, 2),
        )
        
        self.console.print(panel)
    
    def render_gate(self, result: "GateResult", thresholds: "GateThresholds") -> None:
        """Render deployment gate result.
        
        Args:
            result: The gate check result
            thresholds: The thresholds that were used
        """
        from ..core.gate import GateResult, GateThresholds
        
        # Header
        if result.passed:
            status = "[bold green]✅ PASSED[/bold green]"
            border_color = "green"
        else:
            status = "[bold red]❌ BLOCKED[/bold red]"
            border_color = "red"
        
        header = Text()
        header.append("🚦 DEPLOYMENT GATE", style="bold")
        
        lines = [""]
        lines.append(f"[bold]RESULT:[/bold] {status}")
        lines.append("")
        
        # Pass rate comparison
        lines.append("[bold]📊 PASS RATES[/bold]")
        delta = result.pass_rate_candidate - result.pass_rate_baseline
        delta_str = f"+{delta:.1%}" if delta >= 0 else f"{delta:.1%}"
        delta_color = "green" if delta >= 0 else "red"
        
        lines.append(f"   Baseline:  {result.pass_rate_baseline:.1%}")
        lines.append(f"   Candidate: {result.pass_rate_candidate:.1%} ([{delta_color}]{delta_str}[/{delta_color}])")
        lines.append("")
        
        # Threshold checks
        lines.append("[bold]📋 THRESHOLD CHECKS[/bold]")
        
        # max_regression
        regression_ok = result.regression_percent <= thresholds.max_regression_percent
        regression_icon = "✅" if regression_ok else "❌"
        lines.append(f"   {regression_icon} max_regression: {result.regression_percent:.1f}% (limit: {thresholds.max_regression_percent}%)")
        
        # min_pass_rate
        pass_rate_ok = result.pass_rate_candidate >= thresholds.min_pass_rate
        pass_rate_icon = "✅" if pass_rate_ok else "❌"
        lines.append(f"   {pass_rate_icon} min_pass_rate: {result.pass_rate_candidate:.1%} (min: {thresholds.min_pass_rate:.1%})")
        lines.append("")
        
        # Violations
        if result.violations:
            lines.append("[bold red]⚠️  VIOLATIONS[/bold red]")
            for v in result.violations:
                lines.append(f"   • {v.message}")
            lines.append("")
        
        content = "\n".join(lines)
        
        panel = Panel(
            content,
            title=header,
            border_style=border_color,
            padding=(1, 2),
        )
        
        self.console.print(panel)

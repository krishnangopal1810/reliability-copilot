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

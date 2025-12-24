# Phase 3 TODOs

> Phase 3: Agent Reliability Analysis — **Core Complete, Pending Enhancements**

---

## ✅ Completed

### Agent Trace Analysis
- [x] `reco analyze-agent` command
- [x] Agent models: `AgentStep`, `AgentTrace`, `AgentIssue`, `AgentAnalysis`
- [x] Issue detection: tool errors, recovery failures, goal abandonment, excessive retries
- [x] Pattern identification (rule-based)
- [x] Recommendation generation (rule-based)
- [x] Terminal formatter for agent analysis output
- [x] Sample trace: `examples/agent_trace.json`
- [x] Tests: 16 agent-related tests (100% coverage)

---

## 🔲 Pending Enhancements

### High Priority

- [ ] **LLM-based pattern analysis**
  - Currently uses rule-based pattern detection
  - Should use LLM for deeper trace analysis
  - Better context-aware recommendations

- [ ] **Multi-trace aggregation**
  - Analyze patterns across multiple traces
  - Identify systemic agent issues
  - Track agent reliability over time

- [ ] **Agent trace storage**
  - Persist traces in SQLite
  - Enable longitudinal analysis
  - Track agent performance trends

### Medium Priority

- [ ] **Step-by-step trace visualization**
  - Show execution flow diagram
  - Highlight failure points
  - Mermaid diagram output

- [ ] **Tool-specific analysis**
  - Track reliability per tool/action
  - Identify problematic tools
  - Tool error rate statistics

- [ ] **Recovery effectiveness metrics**
  - Measure recovery success rate
  - Track time-to-recovery
  - Identify recovery patterns that work

### Low Priority

- [ ] **Agent profile command**
  - `reco agent-profile` for aggregated agent stats
  - Similar to `reco profile` but for agents

- [ ] **Compare agent performance**
  - `reco compare-agents trace1.json trace2.json`
  - A/B testing for agent implementations

- [ ] **Custom issue detectors**
  - Plugin architecture for domain-specific checks
  - User-defined agent failure patterns

---

## Input Format Reference

```json
{
  "id": "trace_001",
  "goal": "Book a flight from NYC to LA",
  "outcome": "failed",
  "steps": [
    {
      "step": 1,
      "action": "search_flights",
      "input": {"from": "NYC", "to": "LA"},
      "output": {"flights": [...]},
      "success": true,
      "duration_ms": 1250
    },
    {
      "step": 2,
      "action": "book_flight",
      "input": {"flight_id": "FL002"},
      "output": null,
      "success": false,
      "error": "Payment gateway timeout"
    }
  ],
  "metadata": {}
}
```

---

## Quick Verification

```bash
# Analyze a trace
reco analyze-agent examples/agent_trace.json

# Expected output shows:
# - Summary (steps, tools, outcome)
# - Detected issues
# - Identified patterns
# - Recommendations
```

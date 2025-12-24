# Reliability Copilot (reco)

> AI judgment for prompt changes. Know if your change is safe to ship.

## Quick Start

```bash
# Install
pip install -e .

# Set your OpenRouter API key
export OPENROUTER_API_KEY=<openrouter_api_key>

# Compare two eval runs
reco compare examples/baseline.json examples/candidate.json

# Cluster failures to find patterns
reco cluster examples/failures_for_clustering.json

# Check deployment readiness
reco gate examples/baseline.json examples/candidate.json

# View reliability profile
reco profile

# Analyze agent traces
reco analyze-agent examples/agent_trace.json
```

## What is this?

Reliability Copilot is a CLI tool that gives you **judgment**, not just metrics.

- **Compare** two eval runs → Get a clear recommendation: Ship or Don't Ship
- **Cluster** failures → See patterns with consistent taxonomy labels
- **Profile** your system → Track failure modes across runs
- **Gate** deployments → Block releases that exceed thresholds
- **Analyze** agents → Detect reliability issues in multi-step traces
- **Fast** → Results in seconds, not hours

## Commands

### `reco compare`

Compare a baseline and candidate eval run:

```bash
reco compare examples/baseline.json examples/candidate.json
```

Output:
```
╭─────────────────────────────────────────────────────╮
│  ❌ JUDGMENT: DO NOT SHIP                          │
├─────────────────────────────────────────────────────┤
│  📊 Pass rate: 100.0% → 70.0%                      │
│     Change: -30.0%                                  │
│                                                     │
│  ⚠️  REGRESSED (3 cases)                           │
│     • test_002: Hallucinated financial figures     │
│     • test_006: Wrong calculation method           │
│     • test_009: Factual error on capital           │
╰─────────────────────────────────────────────────────╯
```

### `reco cluster`

Group failures by pattern with taxonomy-based classification:

```bash
reco cluster examples/failures_for_clustering.json
```

Output:
```
╭─────────────────────────────────────────────────────╮
│  🔍 FAILURE CLUSTERS (12 failures → 4 patterns)    │
├─────────────────────────────────────────────────────┤
│  CLUSTER 1: Hallucination (4) [RECURRING]          │
│  ├─ Severity: HIGH                                 │
│  └─ Cases: test_101, test_102, test_103, test_401  │
│                                                     │
│  CLUSTER 2: Format Violation (3) [NEW]             │
│  ├─ Severity: MEDIUM                               │
│  └─ Cases: test_201, test_202, test_203            │
╰─────────────────────────────────────────────────────╯
```

### `reco profile`

View aggregated failure statistics across runs:

```bash
reco profile --last-n 10
```

Output:
```
╭─────────────────────────────────────────────────────╮
│  � RELIABILITY PROFILE (10 runs)                  │
├─────────────────────────────────────────────────────┤
│  TOP FAILURE MODES                                 │
│     1. Hallucination ████████ 42%                  │
│     2. Format Violation ████ 25%                   │
│     3. Reasoning Breakdown ██ 15%                  │
╰─────────────────────────────────────────────────────╯
```

### `reco gate`

Check deployment thresholds for CI/CD:

```bash
reco gate baseline.json candidate.json
echo $?  # 0 = pass, 1 = blocked
```

Output:
```
╭─────────────────────────────────────────────────────╮
│  🚦 DEPLOYMENT GATE                                │
├─────────────────────────────────────────────────────┤
│  RESULT: ❌ BLOCKED                                │
│                                                     │
│  📊 PASS RATES                                     │
│     Baseline:  100.0%                              │
│     Candidate: 70.0% (-30.0%)                      │
│                                                     │
│  📋 THRESHOLD CHECKS                               │
│     ❌ max_regression: 30.0% (limit: 15.0%)        │
│     ❌ min_pass_rate: 70.0% (min: 80.0%)           │
╰─────────────────────────────────────────────────────╯
```

### `reco analyze-agent`

Analyze multi-step agent execution traces:

```bash
reco analyze-agent examples/agent_trace.json
```

Output:
```
╭─────────────────────────────────────────────────────╮
│  🤖 AGENT TRACE ANALYSIS                           │
├─────────────────────────────────────────────────────┤
│  📋 SUMMARY                                         │
│     ├─ Steps: 5 total (3 success, 2 failed)        │
│     ├─ Tools: search, book, confirm                │
│     └─ Outcome: FAILED                             │
│                                                     │
│  ⚠️  ISSUES DETECTED                                │
│     ├─ Tool Execution Error at step 4              │
│     └─ No Recovery Attempted at step 5             │
│                                                     │
│  💡 RECOMMENDATIONS                                 │
│     • Add retry logic with exponential backoff     │
│     • Implement fallback strategies                │
╰─────────────────────────────────────────────────────╯
```

## Input Formats

### Eval Run

```json
{
  "name": "optional run name",
  "responses": [
    {
      "id": "test_001",
      "input": "The user prompt",
      "output": "The model response",
      "expected": "Optional expected output",
      "pass": true,
      "failure_reason": "Required if pass is false"
    }
  ]
}
```

### Agent Trace

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
      "output": {"flights": []},
      "success": true
    },
    {
      "step": 2,
      "action": "book_flight",
      "success": false,
      "error": "Payment timeout"
    }
  ]
}
```

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | Yes | Your OpenRouter API key |
| `RECO_LLM_MODEL` | No | Model to use (default: `anthropic/claude-3.5-sonnet`) |
| `RECO_EMBEDDING_MODEL` | No | Embedding model (default: `all-MiniLM-L6-v2`) |

### Gate Thresholds

Create `.reco/thresholds.yaml` to customize deployment gates:

```yaml
max_regression_percent: 15
min_pass_rate: 0.80
block_on_severity: [CRITICAL]
```

### Custom Taxonomy

Create `.reco/taxonomy.yaml` to add domain-specific failure categories:

```yaml
domain_categories:
  - name: "PII Leakage"
    description: "Exposes personal identifiable information"
  - name: "Compliance Violation"
    description: "Breaks regulatory requirements"
```

## Development

```bash
# Clone and install in dev mode
git clone https://github.com/your-org/reliability-copilot
cd reliability-copilot
pip install -e ".[dev]"

# Run tests (279 tests)
pytest

# Run with coverage
pytest --cov=reco
```

## Features by Phase

| Phase | Feature | Status |
|-------|---------|--------|
| 0 | Compare & Cluster | ✅ |
| 1 | Failure Memory (recurring patterns) | ✅ |
| 2 | Reliability Profiles | ✅ |
| 3 | Agent Trace Analysis | ✅ |
| 4 | Deployment Gate | ✅ |

## License

MIT
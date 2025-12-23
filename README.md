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
```

## What is this?

Reliability Copilot is a CLI tool that gives you **judgment**, not just metrics.

- **Compare** two eval runs → Get a clear recommendation: Ship or Don't Ship
- **Cluster** failures → See patterns, not individual errors
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
│                                                     │
│  🎯 Action Items:                                  │
│     • Fix financial hallucination in test_002      │
│     • Review calculation logic                     │
╰─────────────────────────────────────────────────────╯
```

### `reco cluster`

Group failures by pattern:

```bash
reco cluster examples/failures_for_clustering.json
```

Output:
```
╭─────────────────────────────────────────────────────╮
│  🔍 FAILURE CLUSTERS (12 failures → 4 patterns)    │
├─────────────────────────────────────────────────────┤
│                                                     │
│  CLUSTER 1: Financial Data Hallucinations (4)      │
│  ├─ Severity: HIGH                                 │
│  └─ Cases: test_101, test_102, test_103, test_401  │
│                                                     │
│  CLUSTER 2: Format Instruction Violations (3)      │
│  ├─ Severity: MEDIUM                               │
│  └─ Cases: test_201, test_202, test_203            │
│                                                     │
│  CLUSTER 3: Unicode Handling Failures (3)          │
│  ├─ Severity: MEDIUM                               │
│  └─ Cases: test_301, test_302, test_303            │
│                                                     │
╰─────────────────────────────────────────────────────╯
```

## Input Format

Eval runs are JSON files with this structure:

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

## Configuration

Set these environment variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | Yes | Your OpenRouter API key |
| `RECO_LLM_MODEL` | No | Model to use (default: `anthropic/claude-3.5-sonnet`) |
| `RECO_EMBEDDING_MODEL` | No | Embedding model (default: `all-MiniLM-L6-v2`) |

## Development

```bash
# Clone and install in dev mode
git clone https://github.com/your-org/reliability-copilot
cd reliability-copilot
pip install -e ".[dev]"

# Run tests
pytest
```

## License

MIT
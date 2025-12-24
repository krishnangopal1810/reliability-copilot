<p align="center">
  <h1 align="center">🛡️ Reliability Copilot</h1>
  <p align="center">
    <strong>Know if your AI change is safe to ship. In seconds.</strong>
  </p>
  <p align="center">
    <a href="#quick-start">Quick Start</a> •
    <a href="#why-reco">Why reco?</a> •
    <a href="#commands">Commands</a> •
    <a href="#ci-integration">CI Integration</a>
  </p>
</p>

---

**reco** gives you **judgment**, not just metrics. Instead of staring at pass rates wondering "is 87% good?", get a clear answer: **Ship** or **Don't Ship**.

```bash
$ reco compare baseline.json candidate.json

╭─────────────────────────────────────────────────────╮
│  ❌ JUDGMENT: DO NOT SHIP                           │
│                                                     │
│  Pass rate: 100% → 70% (-30%)                       │
│                                                     │
│  ⚠️  REGRESSIONS                                    │
│     • test_002: Hallucinated financial figures      │
│     • test_006: Wrong calculation method            │
│     • test_009: Factual error on capital            │
│                                                     │
│  💡 ACTION: Fix hallucinations before shipping      │
╰─────────────────────────────────────────────────────╯
```

## Quick Start

```bash
pip install reliability-copilot

export OPENROUTER_API_KEY=your_key  # Get one at openrouter.ai

reco compare baseline.json candidate.json
```

## Why reco?

**The problem**: You changed a prompt. Your eval suite runs. You see:

```
Pass rate: 92% → 88%
```

Now what? Is 4% bad? Should you ship? You don't know.

**The solution**: reco tells you:

| What you get | How it helps |
|--------------|--------------|
| **Ship/Don't Ship** | Clear recommendation, not just numbers |
| **Failure clusters** | "12 failures" → "3 patterns you can fix" |
| **Pattern memory** | Same issue reappears? reco remembers |
| **CI gate** | Block bad deployments automatically |

## Commands

### Compare Runs
```bash
reco compare baseline.json candidate.json
```
→ Get a judgment on whether your change is safe

### Cluster Failures
```bash
reco cluster failures.json
```
→ Turn 100 failures into 5 actionable patterns

### Gate Deployments (CI)
```bash
reco gate baseline.json candidate.json || exit 1
```
→ Block releases that exceed thresholds

### Analyze Agents
```bash
reco analyze-agent trace.json
```
→ Find reliability issues in multi-step agent traces

### Import Framework Traces
```bash
reco import-trace langchain.jsonl --format langchain --analyze
reco import-trace openai_run.json --format openai --analyze
```
→ Works with LangChain and OpenAI Assistants

## CI Integration

Add to your GitHub Actions:

```yaml
- name: Check reliability
  run: |
    pip install reliability-copilot
    reco gate evals/baseline.json evals/candidate.json
```

If pass rate drops too much → build fails → bad code doesn't ship.

## Input Format

```json
{
  "responses": [
    {
      "id": "test_001",
      "input": "What's the capital of France?",
      "output": "Paris",
      "pass": true
    },
    {
      "id": "test_002", 
      "input": "Calculate 15% tip on $50",
      "output": "$7.00",
      "pass": false,
      "failure_reason": "Should be $7.50"
    }
  ]
}
```

## Configuration

### Environment Variables

| Variable | Required | Default |
|----------|----------|---------|
| `OPENROUTER_API_KEY` | Yes | - |
| `RECO_LLM_MODEL` | No | `anthropic/claude-3.5-sonnet` |

### Gate Thresholds

Create `.reco/thresholds.yaml`:

```yaml
max_regression_percent: 15   # Block if pass rate drops >15%
min_pass_rate: 0.80          # Block if pass rate below 80%
```

### Custom Failure Categories

Create `.reco/taxonomy.yaml`:

```yaml
domain_categories:
  - name: "PII Leakage"
    description: "Model exposed personal information"
  - name: "Compliance Violation"
    description: "Response violates regulatory requirements"
```

## How It Works

1. **Compare**: LLM analyzes baseline vs candidate responses
2. **Cluster**: Embeddings + HDBSCAN group similar failures
3. **Memory**: SQLite stores patterns for consistency
4. **Gate**: Thresholds determine pass/fail

## Development

```bash
git clone https://github.com/yourusername/reliability-copilot
cd reliability-copilot
pip install -e ".[dev]"
pytest  # 308 tests
```

## License

MIT

---

<p align="center">
  <strong>Built for teams shipping AI products who need confidence, not just metrics.</strong>
</p>
<p align="center">
  <h1 align="center">🛡️ Reliability Copilot</h1>
  <p align="center">
    <strong>Know if your AI change is safe to ship. In seconds.</strong>
  </p>
  <p align="center">
    <a href="https://pypi.org/project/reliability-copilot/"><img src="https://img.shields.io/pypi/v/reliability-copilot" alt="PyPI"></a>
    <a href="https://github.com/krishnangopal1810/reliability-copilot/actions"><img src="https://img.shields.io/badge/tests-321%20passing-brightgreen" alt="Tests"></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  </p>
  <p align="center">
    <a href="#quick-start">Quick Start</a> •
    <a href="#why-reco">Why reco?</a> •
    <a href="#commands">Commands</a> •
    <a href="#ci-integration">CI Integration</a> •
    <a href="#roadmap">Roadmap</a>
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

### Already using PromptFoo?

reco is the **judgment layer** on top of your eval framework:

| PromptFoo gives you | reco adds |
|---------------------|-----------|
| "88% passed" | **"Don't ship — 3 regressions in accuracy"** |
| List of failures | **Clustered patterns you can fix** |
| Manual comparison | **Auto baseline tracking** |
| Numbers | **Decisions** |

```bash
# Before: stare at numbers
npx promptfoo eval → "88% passed" → 🤷

# After: get judgment
reco run → "❌ DO NOT SHIP — introduced hallucinations"
```

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

### PromptFoo Integration (Recommended)
```bash
# In your PromptFoo project
reco init                    # Detects promptfooconfig.yaml
reco run                     # Runs eval, saves baseline
# ... change your prompt ...
reco run                     # Compares to previous, gives judgment
```
→ Seamless workflow with [PromptFoo](https://promptfoo.dev). Requires Node.js.

## CI Integration

### With PromptFoo (Recommended)

```yaml
name: Eval

on: [push, pull_request]

jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
      
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      # Cache baseline between runs
      - uses: actions/cache@v4
        with:
          path: .reco
          key: reco-${{ github.ref }}-${{ hashFiles('promptfooconfig.yaml') }}
          restore-keys: reco-${{ github.ref }}-
      
      - run: pip install reliability-copilot
      
      - name: Run eval and check
        env:
          OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
        run: |
          reco init
          reco run  # Compares to cached baseline, fails on regressions
```

### With JSON files

```yaml
- name: Check reliability
  env:
    OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
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
| `RECO_LLM_MODEL` | No | `xiaomi/mimo-v2-flash:free` |

> **Tip**: For better judgment quality, use a paid model like `anthropic/claude-3.5-sonnet` or `openai/gpt-4o`.

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
git clone https://github.com/krishnangopal1810/reliability-copilot
cd reliability-copilot
pip install -e ".[dev]"
pytest  # 321 tests
```

## Roadmap

- [x] PromptFoo integration (`reco init`, `reco run`)
- [x] LangChain & OpenAI trace import
- [x] Failure clustering with HDBSCAN
- [x] GitHub Actions CI support
- [ ] Python API for programmatic use
- [ ] Braintrust & LangSmith importers
- [ ] reco.cloud — dashboards, trends, team features

## Contributing

Contributions welcome! Please:

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/amazing`)
3. Make your changes
4. Run tests (`pytest`)
5. Submit a PR

See [issues](https://github.com/krishnangopal1810/reliability-copilot/issues) for good first issues.

## License

MIT

---

<p align="center">
  <strong>Built for teams shipping AI products who need confidence, not just metrics.</strong>
</p>

<p align="center">
  ⭐ <strong>If this helps you ship AI with confidence, <a href="https://github.com/krishnangopal1810/reliability-copilot">star this repo</a>!</strong> ⭐
</p>
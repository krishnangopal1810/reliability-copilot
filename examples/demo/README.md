# 🚀 reco Demo

Try reco in 30 seconds.

## Prerequisites

- Python 3.9+
- Node.js 18+
- OpenRouter API key ([get one free](https://openrouter.ai))

## Quick Start

```bash
# 1. Install reco
pip install reliability-copilot

# 2. Set your API key
export OPENROUTER_API_KEY=your_key

# 3. Initialize and run baseline
reco init
reco run

# 4. Change the prompt (edit promptfooconfig.yaml)
# 5. Run again to see comparison
reco run
```

## What's in this demo?

- `promptfooconfig.yaml` — Simple 3-question eval
- Uses free OpenRouter model (no cost)
- Shows the full reco workflow

## Expected Output

**First run:**
```
✅ Saved eval run: eval_20241224_120000
No previous run to compare. This is your new baseline.

Stats:
  Pass rate: 100.0%
  Responses: 3
```

**After changing prompt:**
```
Comparing to previous run: eval_20241224_120000

╭─────────────────────────────────────────────────────╮
│  ✅ JUDGMENT: SHIP                                  │
│  Pass rate: 100% → 100%                             │
╰─────────────────────────────────────────────────────╯
```

## Next Steps

- Try the [CI Integration](../../README.md#ci-integration)
- Add your own tests to `promptfooconfig.yaml`
- Star the repo if this helped! ⭐

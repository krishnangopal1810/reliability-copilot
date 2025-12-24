# Examples

Sample data files for all reco commands.

## 🚀 Quick Start Demo

**New to reco?** Start here:

```bash
cd demo
pip install reliability-copilot
export OPENROUTER_API_KEY=your_key
reco init && reco run
```

See [demo/README.md](demo/README.md) for full instructions.

---

## Example Files

### Compare Runs (Phase 0)
```bash
reco compare baseline.json candidate.json
```
- `baseline.json` — Before your change
- `candidate.json` — After your change

### Cluster Failures (Phase 1)
```bash
reco cluster failures_for_clustering.json
```
- `failures_for_clustering.json` — Failed responses to group into patterns

### Gate Deployments (Phase 3)
```bash
reco gate baseline.json candidate.json || exit 1
```
Uses same files as Compare.

### Analyze Agents (Phase 4)
```bash
reco analyze-agent agent_trace.json
```
- `agent_trace.json` — Multi-step agent execution trace

### Import LangChain Traces
```bash
reco import-trace langchain_trace.jsonl --format langchain --analyze
```
- `langchain_trace.jsonl` — LangChain callback events (JSONL format)

### Import OpenAI Assistants
```bash
reco import-trace openai_run_steps.json --format openai --analyze
```
- `openai_run_steps.json` — OpenAI Assistants Run Steps API output

---

## File Formats

### Eval Run Format
```json
{
  "responses": [
    {
      "id": "test_001",
      "input": "user prompt",
      "output": "model response",
      "pass": true,
      "failure_reason": "optional if pass=false"
    }
  ]
}
```

### Agent Trace Format
```json
{
  "goal": "What the agent was trying to do",
  "steps": [
    {
      "step": 1,
      "action": "tool_name",
      "input": {},
      "output": {},
      "success": true
    }
  ],
  "outcome": "success"
}
```

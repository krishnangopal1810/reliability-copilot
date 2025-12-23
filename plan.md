# Reliability Copilot - Phase 0 Implementation Plan

> **Phase 0: Ship fast, learn faster.** Get judgment into 10 teams' hands in 2 weeks.

---

## Strategic Context

### Why Speed Matters
This is a **good problem in a crowded space**. Competitors (Braintrust, Langsmith, Arize) could add this as a feature. Our advantage:
- **Opinionated focus on judgment** (they do metrics)
- **Speed to market** (2 weeks, not 2 months)
- **Learn from real usage** to build the moat (Phase 1+ memory/profiles)

### The Moat We're Building Toward
Phase 0 is the wedge. The real defensibility comes from:
- **Data network effects**: Patterns learned across teams
- **Failure taxonomy**: We define what "broken" means for AI
- **Integration depth**: Hard to rip out once in the workflow

---

## Core Philosophy

**Stateless. Opinionated. Shippable in 2 weeks.**

### What We Build
✅ Compare two eval runs → Get a judgment  
✅ Cluster failures → See patterns  
✅ CLI-first → Zero setup  

### What We DON'T Build
❌ Persistence, databases, accounts  
❌ Web UI, dashboards  
❌ CI/CD integrations  
❌ Custom eval frameworks  

---

## Architecture

```
baseline.json ──┐
                ├──► reliability-copilot CLI ──► judgment (stdout)
candidate.json ─┘           │
                     ┌──────┴──────┐
                     │ Claude API  │
                     └─────────────┘
```

### Input Format
```json
{
  "responses": [
    {"id": "test_001", "input": "...", "output": "...", "pass": true},
    {"id": "test_002", "input": "...", "output": "...", "pass": false, "failure_reason": "..."}
  ]
}
```

---

## Features

### P0: Compare Command (Days 1-5)
```bash
$ reco compare baseline.json candidate.json

╭─────────────────────────────────────────────────────╮
│  📊 JUDGMENT: RISKIER — DO NOT SHIP                │
├─────────────────────────────────────────────────────┤
│  ✅ Improved: Faster responses, better formatting   │
│  ⚠️  Regressed: Financial hallucinations (3 cases) │
│  🎯 Fix test_042, test_078, test_091 before ship   │
╰─────────────────────────────────────────────────────╯
```

### P1: Cluster Command (Days 6-10)
```bash
$ reco cluster candidate.json

╭─────────────────────────────────────────────────────╮
│  🔍 12 failures → 3 patterns                       │
├─────────────────────────────────────────────────────┤
│  CLUSTER 1: Financial Hallucinations (5) — HIGH    │
│  CLUSTER 2: Format Violations (4) — MEDIUM         │
│  CLUSTER 3: Unicode Edge Cases (3) — LOW           │
╰─────────────────────────────────────────────────────╯
```

---

## Timeline: 2 Weeks

| Day | Focus | Deliverable |
|-----|-------|-------------|
| 0 | Setup | Project scaffold, API keys, sample data |
| 1-3 | Compare v1 | Basic `compare` command works |
| 4-5 | Compare polish | Rich output, edge cases |
| 6-8 | Clustering | `cluster` command with embeddings |
| 9-10 | Ship & distribute | README, install script, reach out to users |

### Parallel: User Acquisition (Days 1-14)
- [ ] Identify 10 target teams (friends, former colleagues, Twitter/X)
- [ ] DM 20 people working on LLM products
- [ ] Post in relevant Discord/Slack communities
- [ ] Goal: 10 active users by end of week 2

---

## Tech Stack

| Component | Choice |
|-----------|--------|
| Language | Python 3.11+ |
| CLI | `typer` |
| LLM | Claude 3.5 Sonnet |
| Embeddings | `sentence-transformers` (local) |
| Clustering | `hdbscan` |

---

## Success Criteria

Phase 0 is **done** when:
- [ ] 10 real teams have run `reco compare` on their data
- [ ] At least 3 say "this changed my decision" 
- [ ] We've learned what to build in Phase 1

Phase 0 is **working** if:
- Teams proactively use it before shipping
- We hear "I was going to ship, but reco said don't"

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| LLM judge inconsistency | 5 golden pairs for calibration |
| Easy to replicate | Move fast, build memory moat |
| No one uses it | Pre-commit to 20 outreach DMs |

---

## Competitive Positioning

| Them | Us |
|------|-----|
| Dashboards & metrics | Judgment & narrative |
| "Here's your accuracy" | "Don't ship, here's why" |
| Set up infrastructure | Run one command |

**One-liner**: "Reliability Copilot tells you if your prompt change is safe to ship."

---

## What Phase 1 Adds (After Validation)

Only if Phase 0 shows traction:
- **Failure memory**: "This pattern appeared 3x in last 5 runs"
- **Persistence**: Store runs for longitudinal analysis
- **API**: For programmatic access

---

*Version: 3.0 — Startup speed*  
*Ship by: Week 2*

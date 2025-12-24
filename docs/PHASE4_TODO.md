# Phase 4 TODOs

> Phase 4: Pre-Deployment Reviews — **Core Complete, Pending Enhancements**

---

## ✅ Completed

- [x] `reco gate` command with exit code 0/1
- [x] Threshold config from `.reco/thresholds.yaml`
- [x] Pass rate regression check
- [x] Minimum pass rate check
- [x] Rich terminal output
- [x] 10 tests (100% gate logic coverage)

---

## 🔲 Pending Enhancements

### High Priority

- [ ] **Severity-aware gating**
  - Block if new CRITICAL/HIGH failures appear
  - Requires clustering candidate failures
  - Compare failure modes between runs

- [ ] **Failure mode regression**
  - Track individual failure categories (Hallucination, Format, etc.)
  - Block if specific category worsens beyond threshold
  - Example: "Block if Hallucination increases by >10%"

- [ ] **JSON output for CI**
  - `--json` flag for machine-readable output
  - Structured violation details for pipeline parsing

### Medium Priority

- [ ] **Baseline from storage**
  - `reco gate candidate.json` (auto-use last run as baseline)
  - Simplifies CI integration

- [ ] **Multiple threshold profiles**
  - `--profile staging` vs `--profile production`
  - Different thresholds for different environments

- [ ] **GitHub Action**
  - Pre-built action for easy integration
  - Auto-comment on PRs with gate results

### Low Priority

- [ ] **Slack/webhook notifications**
  - Alert on gate failures
  - Post results to channel

- [ ] **Historical gate results**
  - Store gate checks in SQLite
  - Track gate pass/fail trends

---

## Config Reference

`.reco/thresholds.yaml`:
```yaml
# Current supported options
max_regression_percent: 15    # Max allowed pass rate drop
min_pass_rate: 0.80           # Minimum absolute pass rate
block_on_severity: [CRITICAL] # Block if new failures of these severities

# Future options
# mode_thresholds:
#   Hallucination: 10
#   Safety Violation: 0  # Zero tolerance
```

---

## Quick Verification

```bash
# Gate passes
$ reco gate passing_baseline.json passing_candidate.json
# Exit code: 0

# Gate fails
$ reco gate examples/baseline.json examples/candidate.json
# Exit code: 1
```

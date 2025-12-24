# Phase 2 TODOs

> Phase 2: Reliability Profiles — **Completed Core, Pending Enhancements**

---

## ✅ Completed

### Reliability Profile Command
- [x] `reco profile` command shows aggregated failure stats
- [x] Top failure modes with occurrence counts
- [x] Pass rate history display

### Failure Taxonomy System
- [x] Universal taxonomy (10 built-in categories)
- [x] Domain extensions via `.reco/taxonomy.yaml`
- [x] Classification instead of generation
- [x] Label reuse for consistency

### Storage Enhancements
- [x] `get_failure_mode_stats()` aggregation query
- [x] `get_pass_rate_history()` query
- [x] `get_run_count()` query
- [x] Severity field in ClusterMatch

---

## 🔲 Pending Enhancements

### High Priority

- [ ] **Silence tokenizer fork warning**
  - Set `TOKENIZERS_PARALLELISM=false` in CLI entry point
  - Low effort, improves UX

- [ ] **Multi-class classification (optional)**
  - Primary category + secondary tags
  - Useful for root cause analysis

- [ ] **Profile time range display**
  - Show date range covered: "Dec 20 - Dec 24"
  - Add first/last run dates to profile output

### Medium Priority

- [ ] **`reco profile --json` output**
  - Machine-readable profile for CI integration
  - JSON schema for profile data

- [ ] **Pass rate tracking**
  - Store total_responses and passed_count per run
  - Show actual pass rate trend in profile

- [ ] **Category drill-down**
  - `reco profile --category Hallucination`
  - Show specific failures in that category

### Low Priority

- [ ] **Custom severity mapping per category**
  - Allow taxonomy.yaml to specify default severity
  - E.g., Safety Violation → always CRITICAL

- [ ] **Taxonomy management commands**
  ```bash
  reco taxonomy list      # Show all categories
  reco taxonomy add       # Add custom category
  reco taxonomy merge     # Merge two categories
  ```

---

## Deferred to Phase 3+

- ❌ Trigger condition inference ("failures spike after prompt refactors")
- ❌ Trend analysis ("increasing", "decreasing", "stable")
- ❌ Cross-project comparison
- ❌ Web dashboard

---

## Quick Verification

```bash
# Clean start
rm -rf .reco/

# Run cluster twice
reco cluster examples/failures_for_clustering.json
reco cluster examples/failures_for_clustering.json

# Verify consistent labels and aggregation
reco profile

# Expected: Same labels, counts show 2x
```

# Phase 1 TODOs

> Core failure memory is complete. These are remaining items from the full Phase 1 vision.

---

## ✅ Completed (Lean Scope)

- [x] Storage layer (`reco/storage/sqlite.py`)
- [x] Cluster history tracking (`is_recurring`, `first_seen`, `occurrence_count`)
- [x] Recurring/new badges in output
- [x] CLI `--no-history` flag
- [x] Unit tests (26 tests)

---

## 🔲 Pending Items

### High Priority

- [ ] **End-to-end manual testing** with real API calls
  - Run `reco cluster` twice to verify RECURRING badge appears
  - Test `--no-history` flag skips storage
  
- [ ] **`diff` command implementation** (Phase 0 week 4, still pending)
  - Show side-by-side diff for specific test cases
  - Currently just shows "coming soon" message

### Medium Priority

- [ ] **Golden pairs for LLM calibration**
  - Create 5 baseline/candidate pairs with known correct judgments
  - Use for regression testing LLM judgment quality

- [ ] **User acquisition tasks**
  - Identify 10 target teams
  - DM 20 people working on LLM products
  - Post in relevant communities
  - Goal: 10 active users

### Optional Enhancements

- [ ] **Config file support** (`~/.reco/config.toml`)
  - Load settings from file instead of just env vars
  - Already stubbed in `config.py`

- [ ] **API mode** for programmatic access
  - Wrap CLI in API for integration

- [ ] **Semantic similarity threshold** as CLI option
  - Currently hardcoded at 0.85

---

## Deferred to Phase 2+

These were explicitly NOT built in Phase 1 lean scope:

- ❌ `reco history` command (view past runs)
- ❌ `reco vulnerabilities` command (aggregate vulnerability report)
- ❌ Database migrations
- ❌ VulnerabilityReport model
- ❌ Complex pattern matching algorithms
- ❌ Web UI / dashboard
- ❌ CI/CD integrations

---

## Quick Verification

```bash
# Test the full Phase 1 flow:
reco cluster examples/failures_for_clustering.json    # First run
reco cluster examples/failures_for_clustering.json    # Should show RECURRING
reco cluster examples/failures_for_clustering.json --no-history  # No storage
```

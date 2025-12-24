# Phase 0 - Remaining Tasks

> **Goal**: Get 10 real teams using `reco` and learn what to build next.

---

## 🔧 Technical Tasks

### 1. Implement Diff Command
**Priority**: Medium  
**Effort**: 2-3 hours

The `diff` command currently shows a placeholder. It should:
- Load both baseline and candidate runs
- Show side-by-side comparison for a specific test case (`--case test_id`)
- Highlight what changed in the output
- Show why it regressed/improved

```bash
# Target usage
reco diff baseline.json candidate.json --case test_042
```

---

### 2. End-to-End Testing with Real API
**Priority**: High  
**Effort**: 30 min

Run actual commands with OpenRouter API to verify:
```bash
export OPENROUTER_API_KEY=your-key
reco compare examples/baseline.json examples/candidate.json
reco cluster examples/failures_for_clustering.json
```

Check that:
- [ ] Compare produces sensible SHIP/DO_NOT_SHIP judgment
- [ ] Cluster groups failures into meaningful patterns
- [ ] Error messages are clear when API fails

---

### 3. Golden Pairs for Calibration (Optional)
**Priority**: Low  
**Effort**: 1-2 hours

Create 5 hand-verified test cases where we know the correct judgment:
- 2 clear "SHIP" cases (improvements only)
- 2 clear "DO NOT SHIP" cases (critical regressions)
- 1 "NEEDS REVIEW" edge case

Use these to validate LLM consistency.

---

## 📣 User Acquisition Tasks

### 4. Identify Target Teams
**Priority**: High  
**Effort**: 1 hour

Find 10 teams working on LLM products who need eval tooling:
- [ ] Friends/former colleagues at AI companies
- [ ] Twitter/X contacts building with LLMs
- [ ] Discord/Slack communities (MLOps, LLM Eng)

### 5. Outreach
**Priority**: High  
**Effort**: 2 hours

- [ ] DM 20 people with personalized message
- [ ] Post in relevant communities
- [ ] Offer to pair on their first `reco` run

### 6. Collect Feedback
**Priority**: High  
**Effort**: Ongoing

For each user:
- [ ] What problem were they solving?
- [ ] Did the judgment change their decision?
- [ ] What's missing?

---

## ✅ Success Criteria

Phase 0 is **DONE** when:
- [ ] 10 real teams have run `reco compare` on their data
- [ ] At least 3 say "this changed my decision"
- [ ] We know what to build in Phase 1

---

## 📋 Checklist Summary

- [ ] End-to-end test with real API
- [ ] Implement `diff` command (optional)
- [ ] Identify 10 target teams
- [ ] Send 20 outreach DMs
- [ ] Get 10 users running `reco`
- [ ] Collect feedback from at least 5

# FAIL-XXXX — [Failure Name]

**Date**: YYYY-MM-DD  
**Related Experiment**: [EXP-XXXX](../experiments/EXP-XXXX.md)  
**Status**: DOCUMENTED

---

## What We Tried

Clear description of the approach that failed.

---

## Why We Tried It

What was the motivation? What problem were we trying to solve?

---

## Hypothesis

What did we expect to happen?

---

## What Actually Happened

Factual description of the result.

---

## Evidence

### Metrics

| Metric | Expected | Actual | Difference |
|---|---:|---:|---:|
| Log Loss | < 0.60 | 0.68 | +0.08 |
| RPS | < 0.17 | 0.19 | +0.02 |

### Additional Evidence

- Plots, tables, or other supporting evidence
- Commands run
- Results observed

---

## Why It Failed

Analysis of root causes. What went wrong?

---

## Lessons Learned

What did we learn from this failure?

1. Lesson 1
2. Lesson 2
3. ...

---

## Correct Approach

What is the right way to solve this problem (if known)?

---

## Decision

**ABANDONED** | **WILL_RETRY** | **NEEDS_MORE_INVESTIGATION**

Explanation of next steps.

---

## Related Work

- [EXP-XXXX](../experiments/EXP-XXXX.md) - Original experiment
- [FAIL-YYYY](FAIL-YYYY.md) - Related failure
- [DEC-XXXX](../decisions/DEC-XXXX.md) - Decision based on this failure

---

## Reproducibility

### Commands to Reproduce Failure

```bash
# Exact commands that produce the failed result
```

This allows others to verify and learn from the failure.

---

**Important**: Negative results are valuable. They tell us what doesn't work and prevent others from repeating the same mistake.

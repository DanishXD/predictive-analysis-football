# EXP-XXXX — [Experiment Name]

**Date**: YYYY-MM-DD  
**Git Commit**: [commit hash]  
**Status**: PLANNED | IN_PROGRESS | COMPLETED | FAILED  
**Evidence Level**: E3 | E4 | E5

---

## Research Question

What specific question does this experiment answer?

---

## Hypothesis

What do we expect to happen?

---

## Dataset

- **Dataset Version**: 
- **Seasons**: 
- **Time Period**: 
- **Number of Fixtures**: 
- **Train/Test Split**: 

---

## Information Cutoff

- **Prediction Cutoff**: 
- **Feature Availability**: 
- **Data Through**: 

---

## Models

### Model A

- Name:
- Version:
- Hyperparameters:
- Features:

### Model B

- Name:
- Version:
- Hyperparameters:
- Features:

---

## Evaluation Protocol

- **Methodology**: (e.g., walk-forward, single split)
- **Folds**: 
- **Test Period**: 
- **No Test-Set Tuning**: YES | NO

---

## Results

### Performance Metrics

| Model | Log Loss | Brier | RPS | ECE | Accuracy |
|---|---:|---:|---:|---:|---:|
| Model A | | | | | |
| Model B | | | | | |

### Additional Results

[Tables, plots, or additional analysis]

---

## Statistical Comparison

### Observed Difference

- **Metric**: 
- **Model A**: 
- **Model B**: 
- **Difference**: 

### Bootstrap 95% CI

- **CI**: [lower, upper]
- **Interpretation**: 

### Permutation Test

- **p-value**: 
- **Interpretation**: 

---

## Calibration

[Calibration analysis if applicable]

---

## Error Analysis

[Stratified performance, failure modes]

---

## Conclusion

Summarize findings. Explain:
- Whether improvement exists
- Magnitude of difference
- Statistical significance
- Robustness
- Limitations

**Do NOT simply say "Model X wins."**

---

## Decision

**KEEP** | **REJECT** | **INVESTIGATE** | **INCONCLUSIVE**

Explanation of decision based on evidence.

---

## Limitations

1. Limitation 1
2. Limitation 2
3. ...

---

## Reproducibility

### Commands to Reproduce

```bash
# Exact commands
```

### Files Generated

- `path/to/results.csv`
- `path/to/plot.png`

---

## Related Files

- [TASK-XXXX](../JOURNAL.md#task-xxxx)
- [DEC-XXXX](../decisions/DEC-XXXX.md)

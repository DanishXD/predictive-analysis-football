# Research Evidence Repository

This directory contains the complete audit trail for the football predictive analysis research upgrade project.

## Purpose

Transform the educational prediction pipeline into a rigorous, auditable, reproducible, leakage-resistant football forecasting research system where predictions can be trusted to the extent justified by out-of-sample evidence.

## Structure

```
research/
├── README.md              # This file
├── JOURNAL.md             # Master chronological log of all tasks
├── MANIFEST.md            # Index of all evidence records
├── decisions/             # Architectural and methodological decisions
├── experiments/           # Controlled experiments with results
├── audits/                # Code audits, leakage checks, quality reviews
├── evaluations/           # Performance evaluations and comparisons
├── failures/              # Documented negative results
├── milestones/            # Phase completion reports
└── templates/             # Evidence document templates
```

## Evidence Levels

- **E0**: No evidence (not acceptable)
- **E1**: Manual observation (weak)
- **E2**: Automated test (code correctness)
- **E3**: Out-of-sample evaluation (predictive evidence)
- **E4**: Statistical evidence (includes uncertainty/comparison)
- **E5**: Reproducible independent evidence (verified across multiple periods)

## Navigation

- Start with `JOURNAL.md` for chronological history
- Use `MANIFEST.md` as an index to find specific evidence
- Each task references related evidence documents

## Golden Rules

1. No important change without evidence
2. No claim without measurement
3. No measurement without context
4. No experiment without recorded result
5. No failed experiment should disappear
6. No historical result should be silently overwritten

## Project Timeline

- **Start Date**: 2026-09-21
- **Target Completion**: 2026-11-30 (10 weeks)
- **Current Phase**: P0 - Critical Infrastructure

## Contact

This is a research-grade system. Every prediction, probability, metric, and claim must be traceable to evidence and independently auditable.

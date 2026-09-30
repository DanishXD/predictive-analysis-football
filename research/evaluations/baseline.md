# Baseline System State — Pre-Upgrade

**Date**: 2026-09-21  
**Purpose**: Capture complete system state before reliability upgrade  
**Status**: BASELINE ESTABLISHED

---

## Executive Summary

This document captures the state of the football predictive analysis system **before** the comprehensive reliability upgrade (P0-P2). All metrics, architecture, and limitations documented here serve as the baseline for measuring improvements.

---

## Dataset

- **Seasons**: 2016-2017 through 2025-2026 (10 seasons)
- **Total Fixtures**: ~3,800 matches
- **Train Period**: 2016-2025 (9 seasons)
- **Test Period**: 2025-2026 (1 season, 380 fixtures)
- **Data Source**: football-data.co.uk via penaltyblog scraper
- **Additional Data**: FBref (player stats), ClubElo (promoted team ratings)

---

## Model Architecture

### Track A: Statistical Models
- **Poisson Goal Model**: Team attack/defense parameters + home advantage
- **Dixon-Coles**: Enhanced Poisson with low-score correction + time decay
- **Elo/Pi-Rating**: Dynamic team ratings updated after each match

### Track B: Machine Learning
- **Logistic Regression**: Baseline ML classifier
- **Random Forest**: Ensemble decision trees
- **XGBoost**: Gradient boosting

### Additional Models
- **Corner Model**: Poisson for corner predictions (HC/AC)
- **Discipline Model**: Poisson for yellow/red cards (HY/AY/HR/AR)
- **Player Models**: Season stats + match-level predictions (SOT, MOTM candidates)

---

## Features

### Current Feature Set
- **Form**: Last 5 matches (W/D/L, goal difference) - rolling
- **Head-to-Head**: Historical H2H record - rolling
- **Home/Away Split**: Separate performance metrics - rolling
- **League Position**: Running table position and points
- **Rest Days**: Days between matches
- **Elo Rating**: Pre-match Elo ratings (from Track A)
- **Elo Gap**: Home Elo - Away Elo

### Temporal Integrity
- Features computed point-in-time (only prior data)
- TimeSeriesSplit used for ML models
- Feature engineering uses rolling windows

---

## Evaluation Metrics (Test Set: 2025-2026)

### Model Performance

| Model | Track | Accuracy | Log Loss | Brier | RPS | Mean ECE | Draw Pred | Draw Recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Bookmaker Odds | Benchmark | 0.495 | 1.015 | 0.610 | 0.205 | 0.040 | 0 | 0.0 |
| Random Forest | Track B | 0.474 | 1.036 | 0.623 | 0.211 | 0.050 | 1 | 0.0 |
| Logistic Regression | Track B | 0.476 | 1.047 | 0.633 | 0.215 | 0.063 | 0 | 0.0 |
| XGBoost | Track B | 0.474 | 1.056 | 0.637 | 0.216 | 0.063 | 1 | 0.0 |
| Elo | Track A | 0.484 | 1.083 | 0.644 | 0.217 | 0.080 | 0 | 0.0 |
| Dixon-Coles | Track A | 0.466 | 1.072 | 0.646 | 0.222 | 0.066 | 0 | 0.0 |
| Poisson | Track A | 0.466 | 1.072 | 0.646 | 0.222 | 0.067 | 0 | 0.0 |

### Key Observations

1. **Bookmaker odds dominate**: Lowest Log Loss, Brier, RPS, and ECE
2. **ML models competitive**: Random Forest achieves lowest RPS among models (0.211)
3. **Draw prediction failure**: All models have 0% draw recall (severe class imbalance issue)
4. **Calibration issues**: ECE ranges from 0.050 to 0.080 for models (vs 0.040 for odds)

---

## Test Coverage

### Existing Tests (9 test files, 37 tests passing)

**Test Files**:
- `test_config.py` - Configuration validation
- `test_team_mapping.py` - Team name standardization
- `test_data_collection_cleaning.py` - Data collection pipeline
- `test_feature_engineering.py` - Feature computation and leakage checks
- `test_date_based_time_splits.py` - Temporal split validation
- `test_btts_probability.py` - BTTS calculation
- `test_over_under_probability.py` - Over/under calculation
- `test_select_best_model.py` - Model selection logic
- `test_team_mapping.py` - Name standardization

**Coverage Areas**:
- ✅ Data collection and cleaning
- ✅ Feature engineering temporal integrity
- ✅ BTTS and over/under calculations
- ✅ Model selection
- ❌ **Missing**: Leakage-specific test suite
- ❌ **Missing**: Walk-forward validation tests
- ❌ **Missing**: Calibration tests
- ❌ **Missing**: Bootstrap/statistical tests

**Test Result**: 37 passed in 11.80s (as of 2026-09-21)

---

## Known Issues & Limitations

### Critical Issues

1. **Model Selection Leakage** (SEVERITY: CRITICAL)
   - Location: `src/predict_match.py` line 69-78
   - Issue: `select_best_model()` reads `evaluation_summary.csv` containing test set metrics
   - Impact: Model selection contaminated by test performance
   - Status: UNFIXED

2. **Single Train/Test Split** (SEVERITY: HIGH)
   - Current: One split (2025-26 as test)
   - Issue: Insufficient evidence of robustness across seasons
   - Impact: Cannot quantify seasonal variance or consistency
   - Status: UNFIXED

3. **No Uncertainty Quantification** (SEVERITY: HIGH)
   - All metrics reported as point estimates
   - No confidence intervals
   - No statistical significance testing
   - Status: UNFIXED

### Model Limitations

4. **Draw Prediction Failure**
   - Draw recall: 0% for all models
   - Draw is minority class (~25% base rate)
   - Class imbalance not adequately handled
   - Status: KNOWN LIMITATION

5. **Calibration Issues**
   - ECE: 0.050-0.080 (models) vs 0.040 (bookmaker)
   - Probabilities not well-calibrated
   - No calibration pipeline implemented
   - Status: KNOWN LIMITATION

6. **Cold-Start Handling**
   - Promoted teams use ClubElo fallback
   - Limited historical data for new teams
   - Uncertainty not quantified
   - Status: ACCEPTABLE WITH WARNINGS

### Infrastructure Gaps

7. **No Prediction Ledger**
   - Predictions not immutably recorded
   - No audit trail of historical predictions
   - Cannot verify pre-match predictions
   - Status: MISSING

8. **No Model Registry**
   - Model versions not tracked
   - Metadata incomplete
   - Predictions not traceable to model version
   - Status: MISSING

9. **No Data Provenance**
   - Dataset metadata incomplete
   - Retrieval timestamps not recorded
   - Reproducibility uncertain
   - Status: MISSING

---

## Architecture

### Current Pipeline Flow

```
data_collection.py
    ↓
matches_clean.csv
    ↓
feature_engineering.py
    ↓
features.csv
    ↓
├─ statistical_models.py (Track A)
│     ↓
│  Poisson/Dixon-Coles/Elo models
│     ↓
│  test_match_probabilities.csv
│
└─ train_model.py (Track B)
      ↓
   LogReg/RF/XGBoost models
      ↓
   ml_test_predictions.csv
      ↓
evaluate.py
      ↓
evaluation_summary.csv
      ↓
predict_match.py
      ↓
[USER PREDICTIONS]
```

### Model Selection (CURRENT - LEAKY)

```
evaluation_summary.csv (contains TEST metrics)
         ↓
   select_best_model()
         ↓
   Chooses model based on test RPS/log loss
         ↓
   predict_match.py uses selected model
```

**Problem**: Test set influences model selection = evaluation leakage

---

## Reproducibility

### Commands to Reproduce Baseline

```bash
# Setup environment
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Run pipeline
python src/data_collection.py
python src/feature_engineering.py
python src/statistical_models.py
python src/train_model.py
python src/corner_model.py
python src/discipline_model.py
python src/evaluate.py

# Run tests
pytest tests/ -q

# View results
# - data/processed/evaluation_summary.csv
# - data/processed/test_match_probabilities.csv
# - data/processed/ml_test_predictions.csv
```

### Environment

- **Python**: 3.11+ (tested on 3.14)
- **OS**: Windows 11
- **Key Dependencies**:
  - penaltyblog==1.11.0
  - pandas>=2.2
  - scikit-learn>=1.5
  - xgboost>=2.1
  - numpy>=1.26

---

## Baseline Metrics Summary

### Best Performing Models (by RPS)

1. **Bookmaker Odds**: 0.205 RPS (benchmark)
2. **Random Forest**: 0.211 RPS (best model)
3. **Logistic Regression**: 0.215 RPS
4. **XGBoost**: 0.216 RPS
5. **Elo**: 0.217 RPS

### Gap to Bookmaker

- **RPS Gap**: +0.006 (RF vs Bookmaker)
- **Log Loss Gap**: +0.021 (RF vs Bookmaker)
- **Calibration Gap**: +0.010 ECE (RF vs Bookmaker)

### Target for Upgrade

The upgrade should:
1. **Fix critical leakage** (model selection)
2. **Add robustness evidence** (walk-forward validation)
3. **Quantify uncertainty** (bootstrap CIs)
4. **Improve calibration** (calibration pipeline)
5. **Add auditability** (ledger, registry, provenance)

**Metric preservation**: Mean performance may stay similar or slightly degrade (due to proper validation), but uncertainty will be quantified and robustness proven.

---

## Evidence Level

**E3**: Out-of-sample evaluation (single test period)

**Note**: Will upgrade to E4 (statistical evidence) and E5 (reproducible across multiple periods) in P0-P2.

---

## Next Steps

1. Begin TASK-0001: Repository Deep Audit
2. Document all leakage risks
3. Implement walk-forward validation
4. Fix model selection leakage
5. Add uncertainty quantification

---

**Baseline Established**: 2026-09-21  
**Upgrade Target**: 2026-11-30 (10 weeks)  
**Phases**: P0 (weeks 1-3), P1 (weeks 4-6), P2 (weeks 7-10)

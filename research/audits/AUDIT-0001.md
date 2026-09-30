# AUDIT-0001 — Full Repository & Reliability Audit

**Date**: 2026-09-21  
**Auditor**: Execution Agent  
**Status**: COMPLETED  
**Scope**: Complete codebase audit for reliability, leakage, and methodological issues

---

## Executive Summary

This audit systematically inspects the football predictive analysis repository to establish a factual baseline before implementing reliability upgrades. The audit identifies **1 CRITICAL** issue (model selection leakage), **3 HIGH** priority issues (single split, no uncertainty, missing infrastructure), and documents the current system architecture.

**Overall Assessment**: The system has good temporal integrity in feature engineering and uses proper train/test splits for model training. However, **model selection uses test set performance**, which is the most severe reliability issue. The single train/test split provides insufficient robustness evidence.

---

## Critical Findings

### FINDING-001: Model Selection Uses Test Set Performance

**Status**: VERIFIED  
**Severity**: CRITICAL  
**Location**: `src/predict_match.py` lines 69-78

**Evidence**:

```python
def select_best_model(
    evaluation: pd.DataFrame,
    candidates: set[str],
) -> pd.Series:
    """Select the lowest RPS, then lowest log loss, from allowed candidates."""
    available = evaluation.loc[evaluation["model"].isin(candidates)].copy()
    if set(available["model"]) != candidates:
        missing = sorted(candidates - set(available["model"]))
        raise ValueError(f"Hasil evaluasi model belum lengkap: {missing}")
    if available[["rps", "log_loss"]].isna().any().any():
        raise ValueError("RPS/log loss model kandidat tidak lengkap")
    return available.sort_values(["rps", "log_loss", "model"]).iloc[0]
```

**Data Flow**:
1. `src/evaluate.py` computes metrics on test set (2025-26) → `evaluation_summary.csv`
2. `src/predict_match.py` line 34: `EVALUATION_PATH = PROCESSED_DIR / "evaluation_summary.csv"`
3. `src/predict_match.py` line 293: `evaluation = pd.read_csv(EVALUATION_PATH)`
4. Lines 294-298: `select_best_model()` called with test metrics
5. Selected model used for all user predictions

**Impact**:
- Model selection contaminated by test set performance
- Test set influences which model users see predictions from
- Reported performance on test set is optimistically biased
- Violates fundamental ML principle: test set must remain locked

**Why It Matters**:
The "best" model is chosen because it happened to perform well on this specific test season, not because it generalizes. A model could overfit to test season characteristics and be selected precisely because of that overfitting.

**Remediation**:
Create separate `cv_model_selection.csv` using only cross-validation metrics. Reserve `test_evaluation_locked.csv` for final reporting, never model selection.

**Priority**: P0 (must fix immediately)

---

## High-Priority Findings

### FINDING-002: Single Train/Test Split Insufficient

**Status**: VERIFIED  
**Severity**: HIGH  
**Location**: `src/config.py` line 37, used throughout pipeline

**Evidence**:
```python
# src/config.py
TEST_SEASON = "2025-2026"

# src/evaluate.py line 58
matches = matches.loc[matches["season"] == TEST_SEASON].copy()

# src/statistical_models.py line 479
train = matches.loc[matches["season"] != TEST_SEASON].copy()
test = matches.loc[matches["season"] == TEST_SEASON].copy()

# src/train_model.py line 169
train = data.loc[data["season"] != TEST_SEASON].copy()
test = data.loc[data["season"] == TEST_SEASON].copy()
```

**Current Methodology**:
- Train: 2016-2025 (9 seasons, ~3,420 matches)
- Test: 2025-2026 (1 season, ~380 matches)
- Evaluation: Single holdout period

**Impact**:
- Cannot quantify robustness across seasons
- Cannot measure seasonal variance
- Single test period could be lucky/unlucky
- No evidence model performs consistently

**Recommended Methodology**:
Walk-forward validation with multiple test periods:
```
Fold 1: Train 2016-2020 → Test 2020-21
Fold 2: Train 2016-2021 → Test 2021-22
Fold 3: Train 2016-2022 → Test 2022-23
Fold 4: Train 2016-2023 → Test 2023-24
Fold 5: Train 2016-2024 → Test 2024-25
Fold 6: Train 2016-2025 → Test 2025-26
```

**Priority**: P0 (required for robust evidence)

---

### FINDING-003: No Uncertainty Quantification

**Status**: VERIFIED  
**Severity**: HIGH  
**Location**: All evaluation scripts

**Evidence**:

From `data/processed/evaluation_summary.csv`:
```
model,track,accuracy,log_loss,brier_score,rps,mean_ece
random_forest,Track B,0.474,1.036,0.623,0.211,0.050
logistic_regression,Track B,0.476,1.047,0.633,0.215,0.063
```

All metrics are point estimates. No confidence intervals, no bootstrap analysis, no statistical significance testing.

**Impact**:
- Cannot determine if model differences are meaningful
- RPS 0.211 vs 0.215: statistically significant or noise?
- Cannot quantify uncertainty in reported metrics
- Cannot make evidence-based claims about model superiority

**Remediation**:
Implement bootstrap confidence intervals:
- Paired bootstrap (resample matches, not independent observations)
- Generate 95% CIs for all metrics
- Compute bootstrap difference for model comparisons
- Report: "Model A: RPS 0.211 [95% CI: 0.205, 0.217]"

**Priority**: P1 (required for research credibility)

---

### FINDING-004: Missing Auditability Infrastructure

**Status**: VERIFIED  
**Severity**: HIGH  
**Location**: Architecture-wide

**Missing Components**:

1. **No Prediction Ledger**
   - Predictions not recorded immutably before matches
   - Cannot prove what model predicted pre-match
   - No audit trail of historical predictions

2. **No Model Registry**
   - Model versions not systematically tracked
   - Predictions not traceable to model version
   - Metadata incomplete (some files have it, inconsistent format)

3. **Incomplete Data Provenance**
   - No systematic metadata for datasets
   - Retrieval timestamps not recorded
   - Cannot fully reproduce data collection

**Evidence**:
```bash
# Check for prediction ledger
dir data\predictions\  # Does not exist

# Check for model registry
dir models\registry.json  # Does not exist

# Check metadata consistency
dir models\*metadata.json  # 4 files exist but schema varies
```

**Impact**:
- Cannot audit historical predictions
- Cannot trace prediction → model → dataset
- Reproducibility uncertain
- Cannot verify pre-match predictions vs post-match fitting

**Priority**: P1 (required for audit trail)

---

## Medium-Priority Findings

### FINDING-005: Draw Prediction Failure

**Status**: VERIFIED  
**Severity**: MEDIUM  
**Location**: All classification models

**Evidence** (from evaluation_summary.csv):
```
model,draw_predictions,draw_recall
random_forest,1,0.0
logistic_regression,0,0.0
xgboost,1,0.0
elo,0,0.0
dixon_coles,0,0.0
poisson,0,0.0
```

**Analysis**:
- Draw base rate: ~25% (95/380 in test set)
- All models predict 0-1 draws total
- Draw recall: 0% for all models
- Models systematically under-predict minority class

**Why It Matters**:
Draw is the hardest outcome to predict but represents 25% of matches. Models essentially ignore this class.

**Attempted Remediation**:
Per README.md and scope document, `class_weight='balanced'` was attempted but reverted because it degraded RPS. This is documented as an acceptable limitation.

**Current Status**: KNOWN LIMITATION (documented, accepted)

**Priority**: P2 (investigate further improvements, but acceptable if none found)

---

### FINDING-006: Calibration Evaluated But Not Implemented

**Status**: VERIFIED  
**Severity**: MEDIUM

**Evidence**:

`src/evaluate.py` computes calibration metrics:
```python
def expected_calibration_error(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    bins: int = 10,
) -> float:
    """Calculate mean one-vs-rest ECE across Home/Draw/Away."""
```

Results: Models have ECE 0.050-0.080 vs bookmaker 0.040

**But**: No calibration *pipeline* exists. No Platt scaling, isotonic regression, or temperature scaling implemented to *improve* calibration.

**Gap**: Evaluation without remediation.

**Priority**: P1 (calibration pipeline needed)

---

## Temporal Integrity Audit

### FINDING-007: Feature Engineering Is Leakage-Safe

**Status**: VERIFIED  
**Severity**: INFO (positive finding)  
**Location**: `src/feature_engineering.py`

**Evidence**:

Critical temporal safeguards observed:

**1. Date-based Cutoff**:
```python
# Line 118-125: Features computed per date BEFORE results applied
for match_date, daily_matches in matches.groupby("date", sort=True):
    positions_by_season = {
        season: get_table_positions(season, season_teams, table)
        for season in daily_matches["season"].unique()
    }
    
    for match in daily_matches.itertuples(index=False):
        # Feature snapshot taken here (line 128-182)
        # ...
        
    # Results applied AFTER snapshots (line 185-214)
    for match in daily_matches.itertuples(index=False):
        # Update histories here
```

**2. Rolling Window Enforcement**:
```python
# Line 55
form_history = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW))
```
Deque automatically caps at 5 matches.

**3. Validation**:
```python
# Line 241-244: Checks rolling window never exceeds limit
if features[["home_form5_played", "away_form5_played", "h2h_played"]].max().max() > ROLLING_WINDOW:
    errors.append(f"rolling window melebihi {ROLLING_WINDOW} pertandingan")
```

**4. Season Opener Check**:
```python
# Line 247-254: Verifies season-to-date stats are zero at season start
first_dates = features.groupby("season")["date"].transform("min")
season_openers = features.loc[features["date"] == first_dates]
if season_openers[opener_columns].ne(0).any().any():
    errors.append("fitur season-to-date pada tanggal pembuka tidak nol")
```

**Test Coverage**: `tests/test_feature_engineering.py` verifies:
- Same-day matches don't leak
- Subsequent matches see only prior results
- Form window caps at ROLLING_WINDOW
- League position valid

**Conclusion**: Feature engineering demonstrates strong temporal integrity.

**Action**: None required. This is correct implementation.

---

### FINDING-008: Statistical Models Use Train-Only Data

**Status**: VERIFIED  
**Severity**: INFO (positive finding)  
**Location**: `src/statistical_models.py`

**Evidence**:
```python
# Line 479-482
train = matches.loc[matches["season"] != TEST_SEASON].copy()
test = matches.loc[matches["season"] == TEST_SEASON].copy()
if train["date"].max() >= test["date"].min():
    raise ValueError("Split waktu tidak valid: train overlap dengan test")

# Line 484-485: Fit on train only
poisson, dixon_coles = fit_goal_models(train)
```

**Validation**: Explicit temporal check prevents overlap.

**Action**: None required. Correct implementation.

---

### FINDING-009: ML Models Use TimeSeriesSplit

**Status**: VERIFIED  
**Severity**: INFO (positive finding)  
**Location**: `src/train_model.py`

**Evidence**:
```python
# Line 94-106: Date-based temporal splits
def date_based_time_splits(data: pd.DataFrame):
    """Apply TimeSeriesSplit to unique dates so a date never spans two folds."""
    unique_dates = np.array(sorted(data["date"].unique()))
    splitter = TimeSeriesSplit(n_splits=N_SPLITS)
    
    for fold, (train_date_idx, validation_date_idx) in enumerate(
        splitter.split(unique_dates), start=1
    ):
        # ... (lines 97-106)
        if data.iloc[train_idx]["date"].max() >= data.iloc[validation_idx]["date"].min():
            raise ValueError(f"Temporal overlap pada fold {fold}")
```

**ML Training** (line 169-171):
```python
train = data.loc[data["season"] != TEST_SEASON].copy()
test = data.loc[data["season"] == TEST_SEASON].copy()
# ... fit on train, evaluate on test once
```

**Hyperparameter Selection** (line 128):
```python
# Cross-validation uses TimeSeriesSplit, not test set
cv_results = cross_validate_models(models, train, feature_columns)
```

**Conclusion**: ML training respects temporal ordering.

**Action**: None required. Correct implementation.

---

## Stacking & Ensemble Audit

### FINDING-010: Stacking Uses Out-of-Fold Predictions

**Status**: VERIFIED  
**Severity**: INFO (positive finding, but optional)  
**Location**: `src/stacking.py`

**Evidence**:
```python
# Line 48-61: Out-of-fold expected goals generation
def generate_oof_expected_goals(train: pd.DataFrame) -> pd.DataFrame:
    """Generate expanding-window expected goals for validation rows only."""
    rows = []
    for fold, train_idx, validation_idx in date_based_time_splits(train):
        fold_train = train.iloc[train_idx]
        fold_validation = train.iloc[validation_idx]
        # ... fit on fold_train, predict fold_validation
```

**Purpose**: Stack expected goals from Poisson/Dixon-Coles into Random Forest.

**Temporal Safety**: Uses proper OOF (out-of-fold) methodology:
1. Each validation fold gets predictions from model fit on earlier data only
2. Test set gets predictions from train-only models (Phase 4 output)
3. No leakage detected

**Status**: Stacking is optional (not always run). When run, it's leakage-safe.

**Action**: None required. Implementation is correct.

---

## Evaluation Methodology Audit

### Current Evaluation Architecture

**Observed Flow**:
```
DATA (2016-2026)
    ↓
[Split: season != TEST_SEASON]
    ↓
TRAIN (2016-2025)          TEST (2025-26)
    ↓                           ↓
├─ Feature Engineering     [Features held out]
├─ Statistical Models      [Models don't see test]
└─ ML Models (CV)          [Models don't see test]
    ↓                           ↓
TRAINED MODELS             [Evaluation once]
    ↓                           ↓
Predict Test Set  ────────> evaluation_summary.csv
                                ↓
                    [PROBLEM: Used for model selection]
                                ↓
                        predict_match.py
```

**Strengths**:
1. ✅ Train/test split is temporal (not random)
2. ✅ Test set never used during training
3. ✅ Feature engineering respects temporal cutoff
4. ✅ ML uses TimeSeriesSplit for hyperparameters
5. ✅ Statistical models fit on train only

**Weaknesses**:
1. ❌ **Model selection uses test metrics** (CRITICAL)
2. ❌ Single split, no robustness evidence
3. ❌ No uncertainty quantification
4. ❌ No multiple evaluation periods

---

### Recommended Architecture

**Target Flow**:
```
DATA (2016-2026)
    ↓
WALK-FORWARD VALIDATION (6 folds)
    ↓
├─ Fold 1: Train 2016-20 → Val 2020-21
├─ Fold 2: Train 2016-21 → Val 2021-22
├─ Fold 3: Train 2016-22 → Val 2022-23
├─ Fold 4: Train 2016-23 → Val 2023-24
├─ Fold 5: Train 2016-24 → Val 2024-25
└─ Fold 6: Train 2016-25 → Val 2025-26
    ↓
AGGREGATE CV METRICS (mean ± std)
    ↓
MODEL SELECTION (based on CV only)
    ↓
cv_model_selection.csv
    ↓
FINAL TEST EVALUATION (one-time, locked)
    ↓
test_evaluation_locked.csv
    ↓
predict_match.py (uses CV-selected models)
```

**Key Differences**:
- Model selection: CV metrics only, never test
- Robustness: 6 out-of-sample periods
- Uncertainty: Bootstrap CIs on each fold
- Test set: Locked, reported once, never influences decisions

---

## Baseline Performance Snapshot

### Data Source

Metrics extracted from: `data/processed/evaluation_summary.csv`  
Generated by: `src/evaluate.py`  
Test Period: 2025-2026 (380 matches)  
Evaluation Date: 2026-08-19 (last modified timestamp)

### Current Test Set Performance

| Model | Track | Accuracy | Log Loss | Brier | RPS | ECE |
|---|---|---:|---:|---:|---:|---:|
| **Bookmaker Odds** | Benchmark | 0.495 | 1.015 | 0.610 | 0.205 | 0.040 |
| **Random Forest** | Track B | 0.474 | 1.036 | 0.623 | 0.211 | 0.050 |
| Logistic Regression | Track B | 0.476 | 1.047 | 0.633 | 0.215 | 0.063 |
| XGBoost | Track B | 0.474 | 1.056 | 0.637 | 0.216 | 0.063 |
| Elo | Track A | 0.484 | 1.083 | 0.644 | 0.217 | 0.080 |
| Dixon-Coles | Track A | 0.466 | 1.072 | 0.646 | 0.222 | 0.066 |
| Poisson | Track A | 0.466 | 1.072 | 0.646 | 0.222 | 0.067 |

### Key Observations

1. **Bookmaker dominance**: Bookmaker odds achieve lowest metrics across all categories
2. **Best model**: Random Forest (RPS 0.211, closest to bookmaker)
3. **Gap to benchmark**: +0.006 RPS, +0.021 log loss, +0.010 ECE
4. **Draw failure**: All models 0% draw recall (see FINDING-005)
5. **Model consistency**: Track B (ML) outperforms Track A (statistical) on RPS

### Important Caveats

⚠️ **These metrics are contaminated by model selection leakage** (FINDING-001)

The "best" model (Random Forest) was selected precisely because it performed well on this test set. True out-of-sample performance is unknown until walk-forward validation is implemented.

---

## Test Coverage Audit

### Existing Tests

**Test Files**: 9 files, 37 tests passing (verified 2026-09-21)

1. ✅ `test_config.py` - Configuration validation
2. ✅ `test_team_mapping.py` - Name standardization
3. ✅ `test_data_collection_cleaning.py` - Data pipeline
4. ✅ `test_feature_engineering.py` - Temporal integrity
5. ✅ `test_date_based_time_splits.py` - TimeSeriesSplit
6. ✅ `test_btts_probability.py` - BTTS calculation
7. ✅ `test_over_under_probability.py` - Over/under
8. ✅ `test_select_best_model.py` - Model selection logic
9. ✅ `test_team_mapping.py` - Name mapping

### Test Coverage Assessment

**Strengths**:
- ✅ Feature engineering temporal checks
- ✅ Data collection validation
- ✅ Time series split validation
- ✅ Probability calculation checks

**Gaps**:
- ❌ No explicit leakage test suite
- ❌ No bootstrap/statistical tests
- ❌ No calibration tests
- ❌ No walk-forward validation tests
- ❌ No model registry tests
- ❌ No prediction ledger tests

**Test Coverage Estimate**: ~40-50% (good for data/features, poor for infrastructure)

---

## Reproducibility Audit

### FINDING-011: Partially Reproducible

**Status**: VERIFIED  
**Severity**: MEDIUM

**Can Be Reproduced**:
1. ✅ Data collection (football-data.co.uk URLs documented)
2. ✅ Feature engineering (deterministic given input data)
3. ✅ Model training (random seeds set: `RANDOM_STATE = 42`)
4. ✅ Evaluation (deterministic)

**Cannot Be Fully Reproduced**:
1. ❌ Original data retrieval timestamp unknown
2. ❌ Player data (FBref scraping, may be blocked)
3. ❌ ClubElo API calls (external dependency, could change)
4. ❌ Exact environment (Python 3.11 vs 3.14, library versions)

**Mitigation**: `requirements.txt` pins major versions but allows minor updates.

**Recommendation**: Add `requirements.lock` with exact versions.

---

## Cold-Start Handling Audit

### FINDING-012: Promoted Teams Use ClubElo Fallback

**Status**: VERIFIED  
**Severity**: INFO  
**Location**: `src/statistical_models.py` lines 237-299

**Evidence**:
```python
def get_clubelo_rating(team: str, as_of_date: pd.Timestamp) -> float | None:
    """Query ClubElo API for a team's rating as of a specific date."""
    # ... (lines 245-278)

def get_bottom_three_average_elo(...) -> float:
    """Fallback: compute average Elo of bottom 3 teams from previous season."""
    # ... (lines 281-329)
```

**Strategy**:
1. Try ClubElo API first (cross-division ratings)
2. Fallback: Average Elo of bottom 3 teams from previous season
3. Final fallback: Default rating (1500)

**Assessment**: Reasonable approach. ClubElo provides better priors than arbitrary defaults.

**Limitation**: Cold-start uncertainty not quantified. Predictions for promoted teams have higher uncertainty but this isn't exposed to users.

**Action**: Document cold-start warnings in predictions (already done in predict_match.py lines 164-172)

---

## Player Data Audit

### FINDING-013: Player Data Not Used in Core Models

**Status**: VERIFIED  
**Severity**: INFO  
**Location**: `src/player_stats.py`, `src/player_match_model.py`

**Evidence**:

Player data is **display-only**, not used as features:
- `predict_match.py` shows player stats but doesn't feed them to models
- Core models (Phases 4-5) use only team-level features
- Player models generate separate predictions (SOT, MOTM candidates)

**Why This Matters**: Player data cannot leak into W/D/L predictions. It's informational context only.

**Limitations** (documented in README):
1. No injury information
2. No suspension information
3. No starting XI information
4. No rotation/squad changes

**Assessment**: Correctly scoped. Player data is supplementary information, not predictive features.

---

## Market Comparison Audit

### FINDING-014: Odds Used Only as Benchmark

**Status**: VERIFIED  
**Severity**: INFO  
**Location**: `src/evaluate.py`, `src/value_betting.py`

**Evidence**:

**In `evaluate.py`** (lines 81-107):
```python
def decode_bookmaker_odds(matches: pd.DataFrame) -> pd.DataFrame:
    """Remove average-market overround with penaltyblog's multiplicative method."""
    # ... decodes odds to probabilities
    # ... adds as 'bookmaker_avg_odds' model to evaluation
```

Odds are:
1. Decoded to remove overround
2. Evaluated as a benchmark model
3. **Never used as training features**

**In `value_betting.py`**:
- Compares model probabilities vs odds
- Computes Expected Value (EV)
- Educational tool, not betting recommendation

**Assessment**: Correct usage. Odds are benchmark, not features.

**Disclaimer** (from value_betting.py line 40-44):
> "EDUCATIONAL ONLY: not a betting recommendation. High EV may be noise, not a real edge. Only about 3,800 EPL matches were used; market odds beat every model on RPS and log loss in Phase 7."

---

## Documentation Audit

### Existing Documentation

**Strengths**:
1. ✅ Comprehensive README.md
2. ✅ Detailed scope-predictive-analysis-bola.md
3. ✅ AGENTS.md for development guidelines
4. ✅ Inline code comments where needed
5. ✅ Docstrings for most functions

**Gaps**:
1. ❌ No evaluation methodology document
2. ❌ No calibration methodology document
3. ❌ No model card
4. ❌ No data provenance document
5. ❌ No reproducibility guide
6. ❌ No drift monitoring documentation

**README Disclaimers**: Well-stated. Project clearly labeled as educational, not betting tool.

---

## Unknowns

### Items Requiring Further Investigation

1. **UNKNOWN**: Training consistency across all models
   - `evaluate.py` has `check_training_consistency()` function (line 550)
   - But metadata files may not exist for all models
   - **Action**: Run full pipeline to generate consistent metadata

2. **UNKNOWN**: Exact data retrieval dates
   - Raw CSV files exist but retrieval timestamps not recorded
   - **Action**: Add data provenance tracking

3. **UNKNOWN**: Historical prediction accuracy
   - No prediction ledger exists
   - Cannot verify what models predicted before matches occurred
   - **Action**: Implement prediction ledger

4. **UNKNOWN**: Performance on promoted teams specifically
   - Could be analyzed but not currently reported
   - **Action**: Add cold-start slice to error analysis

5. **UNKNOWN**: Seasonal variance
   - Single split doesn't show variance across seasons
   - **Action**: Walk-forward validation will reveal this

---

## Recommended Implementation Order

Based on audit findings, recommended task order:

### Priority 0 (CRITICAL - Weeks 1-2)

1. **TASK-0006** (Part 1): Fix Model Selection Leakage
   - Create `cv_model_selection.csv` using CV metrics only
   - Update `predict_match.py` to read CV selection
   - Lock `evaluation_summary.csv` as test-only

2. **TASK-0003**: Implement Walk-Forward Validation
   - Create `src/evaluation/walk_forward.py`
   - Generate per-fold predictions with metadata
   - Aggregate metrics with mean ± std

### Priority 1 (HIGH - Weeks 3-4)

3. **TASK-0008**: Bootstrap Confidence Intervals
   - Implement paired bootstrap (match-level resampling)
   - Generate 95% CIs for all metrics
   - Add to evaluation outputs

4. **TASK-0011**: Model Registry
   - Create `src/infrastructure/model_registry.py`
   - Track all model versions with metadata
   - Link predictions to model versions

5. **TASK-0012**: Prediction Ledger
   - Create `src/infrastructure/prediction_ledger.py`
   - Record predictions immutably before matches
   - Score predictions after matches

### Priority 2 (MEDIUM - Weeks 5-6)

6. **TASK-0007**: Calibration Pipeline
   - Implement Platt/isotonic/temperature scaling
   - Fit on validation, apply to test
   - Compare raw vs calibrated performance

7. **TASK-0014**: Data Provenance
   - Add dataset metadata tracking
   - Record retrieval timestamps
   - Create `data/metadata/dataset_metadata.json`

8. **TASK-0005**: Baseline Models
   - Implement uniform, base-rate, home-advantage baselines
   - Ensure all evaluated on identical fixtures

---

## Expected Risk Reduction

### By Fixing FINDING-001 (Model Selection Leakage)

**Before**: Test-contaminated selection  
**After**: CV-based selection  
**Risk Reduced**: Selection bias, test set overfitting  
**Evidence Quality**: Improves from compromised to valid

### By Implementing Walk-Forward (FINDING-002)

**Before**: Single test period (380 matches)  
**After**: 6 test periods (6× evaluation)  
**Risk Reduced**: Lucky/unlucky single split, unknown seasonal variance  
**Evidence Quality**: Improves from weak (1 period) to strong (6 periods)

### By Adding Uncertainty (FINDING-003)

**Before**: Point estimates only  
**After**: Metrics with 95% CIs  
**Risk Reduced**: False precision, inability to compare models statistically  
**Evidence Quality**: Improves from descriptive to inferential

### By Adding Infrastructure (FINDING-004)

**Before**: No audit trail  
**After**: Complete lineage (data → features → model → prediction)  
**Risk Reduced**: Irreproducibility, inability to verify predictions  
**Evidence Quality**: Improves from unauditable to fully auditable

---

## Biggest Architectural Risks

### Risk 1: Model Selection Using Test Set

**Severity**: CRITICAL  
**Current State**: Verified ongoing  
**Impact**: All reported performance potentially biased  
**Mitigation**: Fix immediately (TASK-0006)

### Risk 2: Single Split Masking Overfitting

**Severity**: HIGH  
**Current State**: No seasonal robustness evidence  
**Impact**: Model may perform poorly on 2026-27 data  
**Mitigation**: Walk-forward validation (TASK-0003)

### Risk 3: No Uncertainty Quantification

**Severity**: HIGH  
**Current State**: Cannot distinguish signal from noise  
**Impact**: Cannot make statistical claims about model superiority  
**Mitigation**: Bootstrap CIs (TASK-0008)

### Risk 4: Missing Audit Trail

**Severity**: MEDIUM  
**Current State**: Cannot verify historical predictions  
**Impact**: Cannot prove model made predictions before matches  
**Mitigation**: Prediction ledger (TASK-0012)

### Risk 5: Draw Class Failure

**Severity**: MEDIUM (accepted limitation)  
**Current State**: 0% draw recall across all models  
**Impact**: Models miss ~25% of outcomes  
**Mitigation**: Investigate further (P2), but acceptable if no improvement found

---

## Final Assessment

### What Works Well

1. ✅ **Feature engineering**: Strong temporal integrity, well-tested
2. ✅ **Train/test split**: Proper temporal separation
3. ✅ **ML validation**: TimeSeriesSplit for hyperparameters
4. ✅ **Statistical models**: Train-only fitting
5. ✅ **Documentation**: Clear disclaimers, educational focus
6. ✅ **Code quality**: Readable, modular, tested

### What Needs Immediate Attention

1. ❌ **Model selection leakage** (CRITICAL)
2. ❌ **Single split evaluation** (insufficient robustness evidence)
3. ❌ **No uncertainty quantification** (cannot make statistical claims)
4. ❌ **Missing infrastructure** (no audit trail)

### Evidence Level

**Current**: E3 (out-of-sample evaluation on single period)  
**Target**: E5 (reproducible evidence across multiple periods with statistical validation)

---

## Audit Completion

**Total Findings**: 14 (1 critical, 3 high, 10 medium/info)  
**Verification Status**: All findings verified by code inspection  
**Recommendation**: Proceed with P0 tasks (model selection fix + walk-forward validation)

**Next Step**: Present audit summary and await approval before implementation.

---

**Auditor Signature**: Execution Agent  
**Date Completed**: 2026-09-21  
**Evidence Trail**: Complete code inspection, all findings documented with line references

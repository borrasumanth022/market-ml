# Skill: model-training

Walk-forward validation standards, no-lookahead rules, and training conventions.

## The cardinal rule: no lookahead bias

**This is the single most important constraint in the project.**

Lookahead bias occurs when a model sees future data during training or feature computation.
It produces models that appear to work but fail completely on live data.

### Where it can sneak in

1. **Labels computed using `shift(-n)`** — correct, this is intentional forward-looking.
   Labels ARE forward returns. What must NOT happen is features using the same.

2. **Global scalers** — never fit a StandardScaler on the full dataset then split.
   Must fit on training fold only, apply to test fold.

3. **Rolling windows across the train/test split** — if test starts at date T, any
   feature computed on day T must only look at data through T-1. This is naturally
   satisfied by pandas `rolling()` if the data is sorted chronologically.

4. **Target leakage** — never include the return being predicted as a feature.
   `return_1d` (yesterday's return) is fine. `ret_1w` (next week's return) is a label.

5. **Feature names containing forward-looking words**: `future_`, `fwd_`, `next_`, `lead_`
   — flag immediately and investigate.

6. **Cross-validation with shuffle** — NEVER use sklearn's default KFold on time-series.
   Always use TimeSeriesSplit or our custom walk-forward implementation.

---

## Walk-forward validation (the only CV method we use)

We use **expanding-window walk-forward** with 5 folds. The initial training window is
seeded from the full history; each subsequent fold adds more training data.

```
Fold 1: Train 1995–2010 | Test 2010–2014
Fold 2: Train 1995–2014 | Test 2014–2017
Fold 3: Train 1995–2017 | Test 2017–2020
Fold 4: Train 1995–2020 | Test 2020–2022
Fold 5: Train 1995–2022 | Test 2022–2024
```

This mirrors production deployment where the model always trains on all past data and
predicts forward — unlike k-fold which trains on future data to predict the past.

**For short-history tickers** (META from 2012, MRNA from 2018): the folds shrink proportionally.
With 3,479 rows (META), 5 folds means ~700 rows per test fold — still statistically meaningful.
With 1,830 rows (MRNA), consider 3 folds instead of 5 to keep test folds above 300 rows.

### Implementation pattern

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
oos_preds = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Fit scaler on TRAIN only
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)    # NOT fit_transform

    model.fit(X_train_sc, y_train)
    preds = model.predict(X_test_sc)
    oos_preds.append((y_test, preds))
```

---

## XGBoost standard hyperparameters

These are fixed across all experiments so improvements come from features/architecture,
not from lucky tuning.

```python
XGB_PARAMS = {
    "n_estimators"    : 300,
    "max_depth"       : 4,         # conservative — prevents overfit
    "learning_rate"   : 0.05,
    "subsample"       : 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 20,        # high = requires 20 samples per leaf
    "eval_metric"     : "mlogloss",
    "random_state"    : 42,
    "n_jobs"          : -1,
    "verbosity"       : 0,
}
```

**Class balancing**: always use `class_weight="balanced"` or pass `scale_pos_weight`
when class distribution is uneven. Our 3-class labels (Bear/Sideways/Bull) are always
unbalanced (~31%/17%/52% historically for AAPL).

**Label encoding**: XGBoost requires contiguous integers from 0.
```python
LABEL_ENCODE = {-1: 0,  0: 1,  1: 2}   # Bear=0, Sideways=1, Bull=2
LABEL_DECODE = { 0:-1,  1: 0,  2: 1}
```

---

## Target variable (what we're predicting)

**Primary target**: `dir_1w` — 3-class direction over the next 5 trading days
- Bear  (-1): return < -2%
- Sideways (0): return in [-2%, +2%]
- Bull  (+1): return > +2%

**Why dir_1w?** From aapl_ml experiments: the 1-week horizon gives the best balance
of signal-to-noise. 1-month has too much noise; 1-week is actionable.

**The ±2% threshold** is deliberately wide. Anything less creates too many Sideways
labels and the model can't distinguish them from noise. Anything more creates too few
Sideways labels and the model collapses to Bear/Bull binary.

---

## Training a new ticker

Checklist before training:
1. `data/processed/{TICKER}_features.parquet` exists and has > 800 rows after warm-up
2. Feature count matches expected (57 columns including OHLCV, or 36 model features)
3. No NaN in the 36 model features (check with `df[FEATURES].isna().sum()`)
4. Labels have been computed (`03_labels.py` has been run for this ticker)
5. Run `hooks/lookahead-bias-check.sh {TICKER}` before saving any model

---

## Saving models

Convention for model filenames:
```
models/{sector}/{ticker}_{model_type}_{target}_{notes}.pkl
```

Examples:
```
models/tech/AAPL_xgb_dir1w_phase3_champion.pkl
models/tech/MSFT_xgb_dir1w_baseline.pkl
models/biotech/LLY_xgb_dir1w_baseline.pkl
```

**Always** save alongside the model:
- The feature list used (which 36 features, in order)
- OOS metrics at save time (don't rely on memory)
- The training date range

---

## What we learned from aapl_ml (apply to all tickers)

- XGBoost outperforms LightGBM and LSTM on tabular features
- LightGBM's leaf-growth strategy hurts Sideways detection
- LSTM on engineered features overfits immediately (best epoch = 1)
- Ensemble (XGB + LGBM) was worse than XGB alone — diversity wasn't complementary
- The interaction feature `rate_vol_regime = fed_rate_change_3m * hvol_63d` is very powerful
- Sideways class is always the hardest to predict — it will underperform Bear and Bull recall

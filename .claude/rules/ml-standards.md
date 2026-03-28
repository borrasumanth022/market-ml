# ML Standards — market_ml

## Walk-forward validation is mandatory — no exceptions

All model evaluation must use expanding-window walk-forward cross-validation.
**Never** use `train_test_split(shuffle=True)` or `StratifiedKFold` on time-series data.

Standard configuration:
```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5, gap=0)
# For MRNA, META (short history < 3 years labeled): n_splits=3
```

Minimum training fold size: 600 rows. Skip any fold with fewer rows.

## Lookahead bias — the #1 thing that kills ML trading models

Features must only use information available at prediction time (market close of day T).

**Prohibited patterns:**
```python
df["future_return"] = df["close"].shift(-5)    # explicit lookahead
df["next_week_high"] = df["high"].rolling(5).max().shift(-4)  # window lookahead
scaler.fit_transform(X)   # fitting on full dataset leaks test distribution
```

**Safe patterns:**
```python
df["return_5d"] = df["close"].pct_change(5)    # backward-looking only
df["hvol_21d"]  = df["close"].pct_change().rolling(21).std()  # backward window
scaler.fit(X_train).transform(X_test)          # fit on train only
```

The pre-commit hook `.claude/hooks/lookahead-bias-check.sh` scans for violations automatically.

## Always report the naive baseline

Before reporting model accuracy, compute and explicitly state the naive baseline:
- **dir_1w (3-class, 0/1/2)**: always-Bull (class 2) strategy = **37.50%** accuracy
  (based on market_ml full training set; may vary slightly per ticker)

Any model failing to beat the naive baseline is **rejected**.

## Metric reporting standard

Report in this order for every model evaluation:
1. Naive baseline accuracy (for reference)
2. OOS Accuracy (full walk-forward, not per-fold)
3. Macro F1 score ← **primary decision metric**
4. Per-class precision, recall, F1 (Bear / Sideways / Bull)
5. Confusion matrix
6. Holdout metrics (>= 2024-01-01) — separate from OOS

**Minimum acceptance thresholds** (do not save a model that fails either):
- Macro F1 ≥ 0.35
- No per-class recall < 0.20

## Label encoding — 0/1/2 only (not -1/0/1)

The market_ml pipeline uses integer encoding: **0=Bear, 1=Sideways, 2=Bull**
Sideways band: ±2% over 1 week (5 trading days).

The old aapl_ml encoding used -1/0/1 — do not mix these. The ManthIQ frontend
and backend are calibrated to 0/1/2.

## Class balancing — always required

```python
# XGBoost
model = XGBClassifier(..., class_weight="balanced")  # or compute_sample_weight

# sklearn
from sklearn.utils.class_weight import compute_sample_weight
weights = compute_sample_weight("balanced", y_train)
model.fit(X_train, y_train, sample_weight=weights)
```

Sideways is consistently the most populous class but hardest to predict without balancing.

## SHAP analysis — required before saving a new champion

```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_oos)
# Report top 10 features by mean |SHAP| overall
# Report top 5 features per class (Bear, Sideways, Bull)
```

## Model checkpoint convention

Before retraining, copy the current champion:
```bash
cp models/tech/xgb_tech_shared_v1.pkl models/tech/xgb_tech_shared_v1_backup.pkl
```
Tag the git commit: `git tag model/tech-v1-F1-0.402`

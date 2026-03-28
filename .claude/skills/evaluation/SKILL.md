# Skill: evaluation

How to report model performance and decide if a new model beats the champion.

## The metrics we use

We always report **all four** of these. Reporting just accuracy is misleading on
imbalanced classes.

| Metric | Why it matters | What to watch for |
|--------|---------------|-------------------|
| **OOS Accuracy** | Raw % correct on held-out data | Compare to naive baseline |
| **Macro F1** | F1 averaged equally across 3 classes | Primary ranking metric |
| **Per-class Recall** | Bear%, Sideways%, Bull% separately | Is any class completely ignored? |
| **Naive baseline accuracy** | Always-predict-majority accuracy | F1 > this is minimum bar |

**Primary ranking metric is Macro F1**, not accuracy. This is because:
- Our classes are imbalanced (~51% Bull historically for AAPL)
- A model that always predicts Bull gets 52% accuracy but F1 ≈ 0.23
- Macro F1 penalises ignoring any class equally

---

## Comparison table format

Always present results in this format when comparing models:

```
Model                          Acc       F1     Bear     Side     Bull
-------------------------------------------------------------------
XGB P3 champion (57 feat)   38.30%   0.375   30.6%   39.9%   42.0%  <- CHAMPION
LGBM (57 feat)              36.07%   0.353   29.6%   34.0%   42.8%
Ensemble XGB+LGBM           37.04%   0.347   18.7%   41.5%   44.6%
Naive (always Bull)         52.90%   0.230   0.0%    0.0%   100.0%
```

Include the naive baseline row. Always show it.

---

## The naive baseline

The naive baseline predicts the majority class (Bull for most tickers) for every row.

```python
from collections import Counter

counts = Counter(y_test)
majority = max(counts, key=counts.get)
naive_acc = counts[majority] / len(y_test)

print(f"Naive baseline (always {majority}): {naive_acc:.2%}")
```

A model that doesn't beat the naive baseline in F1 is not deployable, regardless of accuracy.

---

## Reporting standard (what to print before saving a model)

Every training script must print this block before calling `.to_parquet()` or `pickle.dump()`:

```
=== MODEL EVALUATION: {TICKER} {model_name} ===
Target      : {target}
Features    : {n_features}
CV folds    : {n_folds}
Train range : {train_start} to {train_end}

Walk-forward OOS results:
  Accuracy  : {acc:.2%}
  Macro F1  : {f1:.3f}
  Bear  recall : {bear:.1%}
  Side  recall : {side:.1%}
  Bull  recall : {bull:.1%}

Naive baseline (always Bull): {naive_acc:.2%}
Beats naive by: {acc - naive_acc:+.2%}  ({'+' if f1 > prev_f1 else '-'}{abs(f1-prev_f1):.3f} F1 vs champion)

Champion ({prev_model}): Acc={prev_acc:.2%}  F1={prev_f1:.3f}
Result: {'NEW CHAMPION' if f1 > prev_f1 else 'BELOW CHAMPION — not replacing'}
=====================================
```

The `Result` line must appear before the model is saved. If the model is below the
champion, don't save it as the new champion — save it as a reference artifact instead.

---

## Regression to watch for

When a new experiment improves one metric but degrades another, use this decision framework:

| Situation | Decision |
|-----------|----------|
| Macro F1 improves | New champion (F1 is primary metric) |
| F1 neutral, Bear recall improves by > 5pp | Worth noting; keep champion |
| Sideways recall drops below 25% | Reject — model has collapsed on that class |
| Accuracy beats naive but F1 is lower than champion | Keep old champion |
| Any class recall drops to < 10% | Reject — model has effectively stopped predicting that class |

**Historical context**: Sideways is our hardest class. In aapl_ml Phase 2, the baseline
XGBoost scored only 2.5% Sideways recall. The champion achieved 39.9% — this took until
Phase 3 interaction features to break through. Don't regress below 25% on Sideways.

---

## SHAP analysis (run after each new champion)

After a model becomes champion:

```python
import shap

explainer = shap.TreeExplainer(model)
shap_vals = explainer.shap_values(X_test)

# Save mean absolute SHAP per feature per class
shap_summary = pd.DataFrame({
    "feature": feature_names,
    "Bear"   : np.abs(shap_vals[0]).mean(axis=0),
    "Side"   : np.abs(shap_vals[1]).mean(axis=0),
    "Bull"   : np.abs(shap_vals[2]).mean(axis=0),
}).sort_values("Bear", ascending=False)

shap_summary.to_csv(f"models/{sector}/{ticker}_shap_summary.csv", index=False)
```

Report the top 5 SHAP features per class. This tells us whether the model is learning
expected signals (volatility for Bear, momentum for Bull) or something spurious.

---

## Cross-ticker comparison (market_ml specific)

When the same model architecture is trained on multiple tickers, compare them:

```
Ticker  Sector    Acc       F1     Bear     Side     Bull   History
AAPL    tech    38.30%   0.375   30.6%   39.9%   42.0%   7856 rows
MSFT    tech    ?.??%    ?.???   ?.?%    ?.?%    ?.?%    7856 rows
NVDA    tech    ?.??%    ?.???   ?.?%    ?.?%    ?.?%    6832 rows
...
```

If F1 is systematically lower for short-history tickers (META, MRNA), that is expected
and not a bug — less training data = weaker model.

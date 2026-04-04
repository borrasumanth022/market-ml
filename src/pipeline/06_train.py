"""
Step 6 / Phase 1c -- Multi-ticker XGBoost shared sector models
==============================================================
Trains TWO shared sector models on combined ticker data:
  Tech model:    36 technical + 14 event + 9 regime + ticker_id one-hot
  Biotech model: 36 technical + 14 event + 9 regime + 5 FDA + ticker_id one-hot

Walk-forward validation: 5 date-based folds, strictly chronological,
no cross-ticker leakage (all tickers share the same date split points).
Holdout: 2025-01-01 onwards (pushed from 2024-01-01 in Step 11).

Phase 1c expansion (2026-04-04):
  Tech:    6 original -> 12 tickers (added AMD, TSLA, CRM, ADBE, INTC, ORCL)
  Biotech: 5 original -> 10 tickers (added ABBV, BMY, GILD, AMGN, PFE)
  Saves as v2 to preserve v1 models intact.

Output:
  models/tech/xgb_tech_shared_v2.pkl
  models/biotech/xgb_biotech_shared_v2.pkl
  data/processed/{TICKER}_predictions.parquet (per ticker, all 22 tickers)

Usage:
    C:\\Users\\borra\\anaconda3\\python.exe src\\pipeline\\06_train.py
    C:\\Users\\borra\\anaconda3\\python.exe src\\pipeline\\06_train.py tech
    C:\\Users\\borra\\anaconda3\\python.exe src\\pipeline\\06_train.py biotech
"""

import sys
import pickle
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from xgboost import XGBClassifier
import shap

from config.tickers import SECTORS, TICKER_SECTOR

# ── Constants ──────────────────────────────────────────────────────────────────

PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR    = ROOT / "models"

HOLDOUT_DATE = pd.Timestamp("2025-01-01")  # Step 11: pushed from 2024-01-01 to expose 2024 regime

XGB_PARAMS = {
    "n_estimators":     300,
    "max_depth":        4,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 20,
    "objective":        "multi:softprob",
    "num_class":        3,
    "eval_metric":      "mlogloss",
    "random_state":     42,
    "n_jobs":           -1,
    "verbosity":        0,
}

# dir_1w: -1 Bear, 0 Sideways, +1 Bull -> encode to 0, 1, 2
LABEL_ENCODE = {-1: 0, 0: 1, 1: 2}
LABEL_DECODE = {0: -1, 1:  0, 2:  1}
CLASS_NAMES  = ["Bear", "Sideways", "Bull"]

# 36 validated technical features (aapl_ml Phase 2)
SELECTED_36 = [
    "atr_pct", "bb_pct", "bb_width", "candle_body", "candle_dir",
    "close_vs_sma10", "close_vs_sma100", "close_vs_sma20", "close_vs_sma200",
    "close_vs_sma50", "cross_50_200", "day_of_week", "gap_pct", "hl_range_pct",
    "hvol_10d", "hvol_21d", "hvol_63d", "is_month_end", "is_month_start",
    "is_quarter_end", "lower_shadow", "macd_hist", "macd_signal", "month",
    "price_52w_pct", "return_1d", "return_2d", "return_5d", "roc_10", "roc_21",
    "rsi_14", "rsi_7", "stoch_d", "stoch_k", "upper_shadow", "volume_zscore",
]

# 14 event features (all tickers)
EVENT_14 = [
    # earnings (5)
    "days_to_next_earnings", "days_since_last_earnings", "last_eps_surprise_pct",
    "earnings_streak", "has_earnings_data",
    # macro (6)
    "fed_rate_level", "fed_rate_change_1m", "fed_rate_change_3m",
    "cpi_yoy_change", "unemployment_level", "unemployment_change_3m",
    # regime (3)
    "inflation_regime", "macro_stress_score", "rate_environment",
]

# 5 FDA features (biotech only)
FDA_5 = [
    "days_to_next_fda_decision", "days_since_last_fda_decision",
    "last_fda_outcome", "fda_decisions_trailing_12m", "fda_approval_rate_trailing",
]

# 3 credit spread features (financials only)
# Source: BAMLH0A0HYM2 (ICE BofA US HY OAS) from FRED via 04_events.py
# Coverage: 1996-12-31+; pre-coverage rows use 0.0 sentinel
CREDIT_3 = [
    "credit_spread_level",      # daily OAS value
    "credit_spread_change_1m",  # 21-day change
    "credit_spread_zscore",     # 63-day rolling z-score
]

# 9 regime features (Step 11: VIX, yield spread, sentiment, breadth, HMM)
REGIME_9 = [
    "vix_close", "vix_change_1w", "vix_zscore_63d",
    "yield_spread", "yield_spread_change_1m",
    "sentiment_zscore",
    "put_call_ratio",
    "breadth_pct_above_200d",
    "hmm_regime",
]

TARGET = "dir_1w"
N_SPLITS = 5

# Phase 1c: model version per sector (tech/biotech = v2 retrain, financials = v1 new)
MODEL_VERSIONS = {
    "tech":       "v2",
    "biotech":    "v2",
    "financials": "v1",
}

# Baseline F1 scores to compare against
BASELINES = {
    "tech":       {"name": "AAPL-only XGBoost (aapl_ml Phase 2)", "f1": 0.375},
    "biotech":    {"name": "Random baseline",                      "f1": 0.333},
    "financials": {"name": "Random baseline",                      "f1": 0.333},
}

# Original tickers from prior training -- used to load reference metrics for comparison
PREV_TICKERS = {
    "tech":       ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META"],
    "biotech":    ["LLY", "MRNA", "BIIB", "REGN", "VRTX"],
    "financials": [],   # new sector -- no prior model to compare against
}


# ── V1 reference loader ────────────────────────────────────────────────────────

def load_v1_metrics(sector: str) -> dict:
    """Load prior OOS+holdout F1 and training row count from existing prediction files."""
    tickers = PREV_TICKERS.get(sector, [])
    oos_frames, hold_frames = [], []
    for t in tickers:
        path = PROCESSED_DIR / f"{t}_predictions.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path, engine="pyarrow")
        oos_frames.append(df[df["split"] == "oos"])
        hold_frames.append(df[df["split"] == "holdout"])

    if not oos_frames:
        return {}

    oos_all = pd.concat(oos_frames)
    oos_f1  = f1_score(oos_all["actual"], oos_all["predicted"],
                       average="weighted", zero_division=0)

    hold_f1 = float("nan")
    if hold_frames:
        hold_all = pd.concat(hold_frames)
        if len(hold_all) > 0:
            hold_f1 = f1_score(hold_all["actual"], hold_all["predicted"],
                               average="weighted", zero_division=0)

    # Count training rows from combined parquet (pre-holdout)
    n_train = 0
    for t in tickers:
        evt = PROCESSED_DIR / f"{t}_with_events.parquet"
        if evt.exists():
            idx = pd.read_parquet(evt, engine="pyarrow", columns=[]).index
            n_train += int((idx < HOLDOUT_DATE).sum())

    n_features = 65 if sector == "tech" else 69  # v1 one-hot dims
    return {
        "oos_f1":    round(oos_f1, 4),
        "holdout_f1": round(hold_f1, 4) if not np.isnan(hold_f1) else float("nan"),
        "n_tickers": len(tickers),
        "n_train":   n_train,
        "n_features": n_features,
        "one_hot_dim": 6 if sector == "tech" else 5,
    }


# ── Section printer ────────────────────────────────────────────────────────────

def section(title: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")


# ── Data loading ───────────────────────────────────────────────────────────────

def load_sector_data(sector: str) -> pd.DataFrame:
    """Load and combine all ticker data for a sector.

    - Adds 'ticker_id' column
    - Clips AMZN last_eps_surprise_pct at +/-500%
    - Encodes target labels
    - Drops rows after holdout cutoff from the combined dataset
    """
    tickers = SECTORS[sector]["tickers"]
    frames  = []

    print(f"\nLoading {sector} data...")
    for ticker in tickers:
        path = PROCESSED_DIR / f"{ticker}_with_events.parquet"
        if not path.exists():
            print(f"  WARN: {path.name} not found -- skipping {ticker}")
            continue
        df = pd.read_parquet(path)
        df["ticker_id"] = ticker
        frames.append(df)
        print(f"  {ticker}: {len(df):,} rows  "
              f"{str(df.index[0].date())} to {str(df.index[-1].date())}")

    if not frames:
        raise RuntimeError(f"No data loaded for sector '{sector}'")

    combined = pd.concat(frames).sort_index()

    # AMZN EPS outlier: 2012 near-zero-estimate quarters -> +/-3900%
    if "AMZN" in combined["ticker_id"].values:
        mask = combined["ticker_id"] == "AMZN"
        combined.loc[mask, "last_eps_surprise_pct"] = (
            combined.loc[mask, "last_eps_surprise_pct"].clip(-500.0, 500.0)
        )
        print(f"\n  AMZN last_eps_surprise_pct clipped at +/-500%")

    # Encode target
    combined["__label__"] = combined[TARGET].map(LABEL_ENCODE)
    combined = combined.dropna(subset=["__label__"])
    combined["__label__"] = combined["__label__"].astype(int)

    n_total    = len(combined)
    n_holdout  = (combined.index >= HOLDOUT_DATE).sum()
    n_pretrain = n_total - n_holdout

    print(f"\n  Combined: {n_total:,} rows total")
    print(f"  Pre-holdout (training pool): {n_pretrain:,} rows")
    print(f"  Holdout (>= {HOLDOUT_DATE.date()}): {n_holdout:,} rows")

    return combined


def build_feature_matrix(df: pd.DataFrame, sector: str) -> tuple[pd.DataFrame, list[str]]:
    """Build feature matrix with ticker_id one-hot columns."""
    base_feats = SELECTED_36 + EVENT_14 + REGIME_9
    if sector == "biotech":
        base_feats = base_feats + FDA_5
    elif sector == "financials":
        base_feats = base_feats + CREDIT_3

    # One-hot encode ticker_id
    ticker_dummies = pd.get_dummies(df["ticker_id"], prefix="ticker").astype(int)

    X = df[base_feats].copy()
    X = pd.concat([X, ticker_dummies], axis=1)

    feature_names = base_feats + list(ticker_dummies.columns)

    missing = [f for f in base_feats if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features in {sector} data: {missing}")

    return X, feature_names


# ── Walk-forward validation ────────────────────────────────────────────────────

def walk_forward_validate(
    df_pretrain: pd.DataFrame,
    X_pretrain:  pd.DataFrame,
    sector:      str,
) -> tuple[list, list, list, list, list[dict]]:
    """
    Date-based 5-fold walk-forward validation on pre-holdout data.
    Returns: all_indices, all_actual, all_predicted, all_proba, fold_results
    """
    unique_dates = np.array(sorted(df_pretrain.index.unique()))
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    y_all      = df_pretrain["__label__"].values
    X_all      = X_pretrain.values
    date_index = df_pretrain.index

    all_indices   = []
    all_actual    = []
    all_predicted = []
    all_proba     = []
    fold_results  = []

    print(f"\nWalk-forward validation ({N_SPLITS} folds):")
    print(f"  {'Fold':>4}  {'Train dates':>10}  {'Val dates':>10}  "
          f"{'Train rows':>10}  {'Val rows':>8}  {'F1':>6}")
    print("  " + "-" * 60)

    for fold, (tr_date_idx, val_date_idx) in enumerate(tscv.split(unique_dates), 1):
        train_dates = set(unique_dates[tr_date_idx])
        val_dates   = set(unique_dates[val_date_idx])

        tr_mask  = date_index.isin(train_dates)
        val_mask = date_index.isin(val_dates)

        X_train, y_train = X_all[tr_mask],  y_all[tr_mask]
        X_val,   y_val   = X_all[val_mask], y_all[val_mask]

        if len(np.unique(y_train)) < 3:
            print(f"  {fold:>4}  skipped (fewer than 3 classes in training fold)")
            continue

        weights = compute_sample_weight(class_weight="balanced", y=y_train)
        model   = XGBClassifier(**XGB_PARAMS)
        model.fit(X_train, y_train, sample_weight=weights)

        preds  = model.predict(X_val)
        probas = model.predict_proba(X_val)
        f1     = f1_score(y_val, preds, average="weighted")

        tr_start  = min(unique_dates[tr_date_idx]).strftime("%Y-%m-%d")
        val_start = min(unique_dates[val_date_idx]).strftime("%Y-%m-%d")

        print(f"  {fold:>4}  {tr_start:>10}  {val_start:>10}  "
              f"{len(y_train):>10,}  {len(y_val):>8,}  {f1:.4f}")

        # Store per-row indices into the pretrain dataframe
        row_indices = np.where(val_mask)[0]
        all_indices.extend(row_indices.tolist())
        all_actual.extend(y_val.tolist())
        all_predicted.extend(preds.tolist())
        all_proba.extend(probas.tolist())

        fold_results.append({
            "fold":       fold,
            "train_rows": len(y_train),
            "val_rows":   len(y_val),
            "f1":         f1,
            "report":     classification_report(
                              y_val, preds,
                              target_names=CLASS_NAMES,
                              output_dict=True
                          ),
        })

    return all_indices, all_actual, all_predicted, all_proba, fold_results


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_oos_report(
    actual:    list,
    predicted: list,
    fold_results: list[dict],
    sector:    str,
) -> None:
    print(f"\n--- OOS Summary: {sector.upper()} ---")

    # Aggregate fold F1
    fold_f1s = [r["f1"] for r in fold_results]
    print(f"\n  Walk-forward F1 per fold: "
          f"{['%.3f' % f for f in fold_f1s]}")
    print(f"  Mean OOS F1 (weighted): {np.mean(fold_f1s):.4f}")
    print(f"  Std  OOS F1:            {np.std(fold_f1s):.4f}")

    print(f"\n  Full OOS classification report:")
    report = classification_report(
        actual, predicted,
        target_names=CLASS_NAMES,
        digits=3,
    )
    for line in report.splitlines():
        print(f"    {line}")

    # Confusion matrix
    cm = confusion_matrix(actual, predicted)
    print(f"\n  Confusion matrix (Bear / Sideways / Bull):")
    print(f"    Predicted ->  Bear  Side  Bull")
    for i, row_name in enumerate(CLASS_NAMES):
        print(f"    {row_name:10s}    {cm[i, 0]:5d} {cm[i, 1]:5d} {cm[i, 2]:5d}")

    # vs baseline
    oos_f1    = f1_score(actual, predicted, average="weighted")
    baseline  = BASELINES[sector]
    delta     = oos_f1 - baseline["f1"]
    sign      = "+" if delta >= 0 else ""
    print(f"\n  vs. baseline ({baseline['name']}  F1={baseline['f1']:.3f}):")
    print(f"    Sector shared model OOS F1: {oos_f1:.3f}  ({sign}{delta:.3f})")


# ── SHAP analysis ─────────────────────────────────────────────────────────────

def shap_analysis(
    model:         XGBClassifier,
    X:             pd.DataFrame,
    feature_names: list[str],
    sector:        str,
) -> None:
    print(f"\n--- SHAP Feature Importance: {sector.upper()} ---")
    print("  Computing SHAP values (this may take a moment)...")

    # Sample up to 2000 rows for speed
    n_sample = min(2000, len(X))
    rng      = np.random.default_rng(42)
    idx      = rng.choice(len(X), size=n_sample, replace=False)
    X_sample = X.values[idx]

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Normalise to (n_samples, n_features, n_classes)
    if isinstance(shap_values, list):
        shap_arr = np.stack(shap_values, axis=2)
    else:
        shap_arr = np.asarray(shap_values)
        if shap_arr.ndim == 3 and shap_arr.shape[0] == 3:
            shap_arr = shap_arr.transpose(1, 2, 0)

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        mean_abs = np.abs(shap_arr[:, :, cls_idx]).mean(axis=0)
        top_idx  = np.argsort(mean_abs)[::-1][:10]
        print(f"\n  Top 10 features for {cls_name}:")
        for rank, fi in enumerate(top_idx, 1):
            print(f"    {rank:2d}. {feature_names[fi]:40s} {mean_abs[fi]:.5f}")


# ── Holdout evaluation ────────────────────────────────────────────────────────

def evaluate_holdout(
    model:         XGBClassifier,
    df_holdout:    pd.DataFrame,
    X_holdout:     pd.DataFrame,
    sector:        str,
) -> dict:
    if len(df_holdout) == 0:
        print(f"\n  No holdout data for {sector}")
        return {}

    y_holdout = df_holdout["__label__"].values
    preds     = model.predict(X_holdout.values)
    probas    = model.predict_proba(X_holdout.values)
    f1        = f1_score(y_holdout, preds, average="weighted")

    print(f"\n--- Holdout Evaluation: {sector.upper()} (>= {HOLDOUT_DATE.date()}) ---")
    print(f"\n  Holdout rows: {len(y_holdout):,}")
    print(f"  Holdout F1 (weighted): {f1:.4f}")
    report = classification_report(
        y_holdout, preds,
        target_names=CLASS_NAMES,
        digits=3,
    )
    for line in report.splitlines():
        print(f"    {line}")

    return {
        "df":        df_holdout,
        "actual":    y_holdout,
        "predicted": preds,
        "proba":     probas,
    }


# ── Save predictions ──────────────────────────────────────────────────────────

def save_predictions(
    df_sector:     pd.DataFrame,
    oos_indices:   list,
    oos_actual:    list,
    oos_predicted: list,
    oos_proba:     list,
    holdout_data:  dict,
    sector:        str,
) -> None:
    """Build per-row prediction dataframe and split by ticker."""
    tickers = SECTORS[sector]["tickers"]

    # OOS predictions (walk-forward)
    oos_df = df_sector.iloc[oos_indices][["ticker_id", TARGET]].copy()
    oos_df["actual"]     = oos_actual
    oos_df["predicted"]  = oos_predicted
    oos_df["split"]      = "oos"
    proba_arr = np.array(oos_proba)
    oos_df["proba_bear"] = proba_arr[:, 0]
    oos_df["proba_side"] = proba_arr[:, 1]
    oos_df["proba_bull"] = proba_arr[:, 2]

    # Holdout predictions
    frames = [oos_df]
    if holdout_data:
        h_df = holdout_data["df"][["ticker_id", TARGET]].copy()
        h_df["actual"]    = holdout_data["actual"]
        h_df["predicted"] = holdout_data["predicted"]
        h_df["split"]     = "holdout"
        h_proba = np.array(holdout_data["proba"])
        h_df["proba_bear"] = h_proba[:, 0]
        h_df["proba_side"] = h_proba[:, 1]
        h_df["proba_bull"] = h_proba[:, 2]
        frames.append(h_df)

    all_preds = pd.concat(frames).sort_index()

    # Save per ticker
    print(f"\nSaving predictions...")
    for ticker in tickers:
        mask       = all_preds["ticker_id"] == ticker
        ticker_df  = all_preds[mask].copy()
        out_path   = PROCESSED_DIR / f"{ticker}_predictions.parquet"
        ticker_df.to_parquet(out_path)
        size_kb    = out_path.stat().st_size / 1024
        oos_rows   = (ticker_df["split"] == "oos").sum()
        hold_rows  = (ticker_df["split"] == "holdout").sum()
        print(f"  {ticker}: {oos_rows} OOS + {hold_rows} holdout -> "
              f"{out_path.relative_to(ROOT)}  ({size_kb:.0f} KB)")


# ── Train final model ─────────────────────────────────────────────────────────

def train_final_model(
    X_pretrain: pd.DataFrame,
    y_pretrain: np.ndarray,
) -> XGBClassifier:
    """Train final model on all pre-holdout data."""
    weights = compute_sample_weight(class_weight="balanced", y=y_pretrain)
    model   = XGBClassifier(**XGB_PARAMS)
    model.fit(X_pretrain.values, y_pretrain, sample_weight=weights)
    return model


# ── Per-sector orchestrator ───────────────────────────────────────────────────

def run_sector(sector: str) -> None:
    section(f"Sector: {sector.upper()}")

    # 1. Load data
    df_combined = load_sector_data(sector)

    # 2. Build feature matrix (includes ticker_id one-hot)
    X_all, feature_names = build_feature_matrix(df_combined, sector)

    print(f"\n  Feature matrix: {X_all.shape[1]} features")
    print(f"  Features: {SELECTED_36[:3]}... + EVENT_14 "
          f"{'+ FDA_5 ' if sector == 'biotech' else ''}"
          f"+ ticker_id one-hot")

    # 3. Split pre-holdout vs holdout
    pretrain_mask = df_combined.index < HOLDOUT_DATE
    holdout_mask  = df_combined.index >= HOLDOUT_DATE

    df_pretrain = df_combined[pretrain_mask]
    df_holdout  = df_combined[holdout_mask]
    X_pretrain  = X_all[pretrain_mask]
    X_holdout   = X_all[holdout_mask]
    y_pretrain  = df_pretrain["__label__"].values

    # 4. Walk-forward validation
    oos_idx, oos_actual, oos_pred, oos_proba, fold_results = walk_forward_validate(
        df_pretrain, X_pretrain, sector
    )

    # 5. OOS report
    print_oos_report(oos_actual, oos_pred, fold_results, sector)

    # 6. Train final model on all pre-holdout data
    section(f"Training final {sector.upper()} model on all pre-holdout data")
    print(f"  Rows: {len(y_pretrain):,}  |  Features: {X_pretrain.shape[1]}")
    final_model = train_final_model(X_pretrain, y_pretrain)
    print("  Final model trained.")

    # 7. SHAP analysis
    shap_analysis(final_model, X_pretrain, feature_names, sector)

    # 8. Holdout evaluation
    holdout_data = evaluate_holdout(final_model, df_holdout, X_holdout, sector)

    # 9. Save model (version per MODEL_VERSIONS dict)
    model_version = MODEL_VERSIONS[sector]
    model_path = MODELS_DIR / sector / f"xgb_{sector}_shared_{model_version}.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump({
            "model":         final_model,
            "feature_names": feature_names,
            "sector":        sector,
            "tickers":       SECTORS[sector]["tickers"],
            "holdout_date":  HOLDOUT_DATE,
            "xgb_params":    XGB_PARAMS,
        }, f)
    size_kb = model_path.stat().st_size / 1024
    print(f"\n  Model saved -> {model_path.relative_to(ROOT)}  ({size_kb:.0f} KB)")

    # 10. Save predictions
    save_predictions(
        df_combined, oos_idx, oos_actual, oos_pred, oos_proba,
        holdout_data, sector,
    )

    # 11. Before/after comparison (skipped for new sectors with no prior model)
    new_oos_f1  = f1_score(oos_actual, oos_pred, average="weighted")
    new_hold_f1 = (f1_score(holdout_data["actual"], holdout_data["predicted"],
                            average="weighted") if holdout_data else float("nan"))
    new_n_tickers = len(SECTORS[sector]["tickers"])
    new_n_train   = len(y_pretrain)
    new_features  = X_pretrain.shape[1]

    prev = load_v1_metrics(sector)
    model_version = MODEL_VERSIONS[sector]

    section(f"Model Comparison: {sector.upper()} (version: {model_version})")

    if not prev:
        print(f"\n  New sector -- no prior model to compare against.")
        print(f"  Tickers: {new_n_tickers}  |  Train rows: {new_n_train:,}  "
              f"|  Features: {new_features}")
        print(f"  OOS weighted F1:     {new_oos_f1:.4f}")
        print(f"  Holdout weighted F1: {new_hold_f1:.4f}")
    else:
        new_one_hot = new_n_tickers  # one-hot dim matches ticker count
        print(f"\n  {'Metric':<32}  {'Prior':>14}  {'New ({model_version})':>14}  {'Delta':>8}")
        print(f"  {'-'*76}")

        def _row(label, v_prev, v_new, fmt=".3f"):
            if isinstance(v_prev, float) and np.isnan(v_prev):
                prev_s = "  N/A"
                d_s    = "   N/A"
            else:
                prev_s = format(v_prev, fmt)
                delta  = v_new - v_prev
                sign   = "+" if delta >= 0 else ""
                d_s    = f"{sign}{delta:{fmt}}"
            new_s = format(v_new, fmt)
            print(f"  {label:<32}  {prev_s:>14}  {new_s:>14}  {d_s:>8}")

        _row("Tickers in sector", float(prev.get("n_tickers", 0)), float(new_n_tickers), fmt=".0f")
        _row("Pre-holdout train rows", float(prev.get("n_train", 0)), float(new_n_train), fmt=".0f")
        _row("Feature dimensions", float(prev.get("n_features", 0)), float(new_features), fmt=".0f")
        _row("OOS weighted F1", prev.get("oos_f1", float("nan")), new_oos_f1)
        _row("Holdout weighted F1", prev.get("holdout_f1", float("nan")), new_hold_f1)

    print(f"\n  OOS per-class recall ({model_version}):")
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        mask_cls = [a == cls_idx for a in oos_actual]
        if sum(mask_cls) == 0:
            continue
        correct  = sum(p == cls_idx for a, p in zip(oos_actual, oos_pred) if a == cls_idx)
        total    = sum(mask_cls)
        recall   = correct / total if total > 0 else 0.0
        print(f"    {cls_name:8s}: recall={recall:.3f}  (n={total:,})")

    print(f"\n  {sector.upper()} sector complete.")


# ── Entry points ──────────────────────────────────────────────────────────────

def main() -> None:
    section(f"market_ml -- XGBoost sector models (27 tickers, regime-aware)")
    print(f"  Target:        {TARGET}")
    print(f"  Holdout:       >= {HOLDOUT_DATE.date()}")
    print(f"  WF splits:     {N_SPLITS}")
    print(f"  Tech:          36+14+9+12 one-hot = 71 features (12 tickers, v2)")
    print(f"  Biotech:       36+14+9+5 FDA+10 one-hot = 74 features (10 tickers, v2)")
    print(f"  Financials:    36+14+9+3 credit+5 one-hot = 67 features (5 tickers, v1 new)")
    print(f"  Tech baseline:       F1={BASELINES['tech']['f1']}")
    print(f"  Biotech baseline:    F1={BASELINES['biotech']['f1']}")
    print(f"  Financials baseline: F1={BASELINES['financials']['f1']}")

    sectors_to_run = sys.argv[1:] or ["tech", "biotech"]
    for sector in sectors_to_run:
        sector = sector.lower()
        if sector not in SECTORS:
            print(f"ERROR: Unknown sector '{sector}'. Choose: {list(SECTORS.keys())}")
            sys.exit(1)
        run_sector(sector)

    section("All sectors complete")
    for sec, ver in MODEL_VERSIONS.items():
        if sec in sectors_to_run:
            print(f"  Model: models/{sec}/xgb_{sec}_shared_{ver}.pkl")
    print("  Predictions: data/processed/{TICKER}_predictions.parquet")
    print("\n  Next step: src/pipeline/07_evaluate.py\n")


if __name__ == "__main__":
    main()

"""
Step 9 -- Per-ticker fine tuning from shared sector models
==========================================================
Warm-starts from the Step 6 shared sector XGBoost model for each ticker
and continues training on that ticker's data only (+100 estimators).

Warm-start constraint: XGBoost requires the same feature count as the
base model (56 tech / 60 biotech). The ticker one-hot columns are kept
but fixed to the specific ticker (ticker_AAPL=1, all others=0 for AAPL),
so the per-ticker model effectively conditions on a single ticker identity.

Walk-forward evaluation: per-ticker TimeSeriesSplit (3 folds if < 2000
pre-holdout rows, else 5 folds). Compared against Step 7 shared-model
per-ticker OOS F1 from {TICKER}_eval_report.json.

Save logic:
  - Fine-tuned OOS F1 must improve by >= MIN_IMPROVEMENT (0.005) vs shared
  - Skips if models/{sector}/{TICKER}_finetuned_v1.pkl already exists

Output:
  models/{sector}/{TICKER}_finetuned_v1.pkl   (saved tickers only)
  data/processed/{TICKER}_predictions.parquet  (updated for saved tickers)
  docs/finetuning_report.md

Usage:
    python src/pipeline/09_finetune.py
    python src/pipeline/09_finetune.py NVDA       (single ticker)
    python src/pipeline/09_finetune.py tech        (full sector)
    python src/pipeline/09_finetune.py NVDA MRNA   (multiple tickers)
"""

import sys
import json
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
from sklearn.metrics import f1_score, classification_report
from xgboost import XGBClassifier

from config.tickers import SECTORS, TICKER_SECTOR

# ── Constants ──────────────────────────────────────────────────────────────────

PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR    = ROOT / "models"
DOCS_DIR      = ROOT / "docs"

HOLDOUT_DATE    = pd.Timestamp("2024-01-01")
MIN_IMPROVEMENT = 0.005   # minimum OOS F1 gain to save fine-tuned model
MIN_FOLD_ROWS   = 100     # skip folds with fewer training rows
FT_ESTIMATORS   = 100     # additional trees added on top of shared model

CLASS_NAMES  = ["Bear", "Sideways", "Bull"]
CLASS_PROBA  = ["proba_bear", "proba_side", "proba_bull"]
LABEL_ENCODE = {-1: 0, 0: 1, 1: 2}

# Feature sets (mirrors 06_train.py exactly)
SELECTED_36 = [
    "atr_pct", "bb_pct", "bb_width", "candle_body", "candle_dir",
    "close_vs_sma10", "close_vs_sma100", "close_vs_sma20", "close_vs_sma200",
    "close_vs_sma50", "cross_50_200", "day_of_week", "gap_pct", "hl_range_pct",
    "hvol_10d", "hvol_21d", "hvol_63d", "is_month_end", "is_month_start",
    "is_quarter_end", "lower_shadow", "macd_hist", "macd_signal", "month",
    "price_52w_pct", "return_1d", "return_2d", "return_5d", "roc_10", "roc_21",
    "rsi_14", "rsi_7", "stoch_d", "stoch_k", "upper_shadow", "volume_zscore",
]

EVENT_14 = [
    "days_to_next_earnings", "days_since_last_earnings", "last_eps_surprise_pct",
    "earnings_streak", "has_earnings_data",
    "fed_rate_level", "fed_rate_change_1m", "fed_rate_change_3m",
    "cpi_yoy_change", "unemployment_level", "unemployment_change_3m",
    "inflation_regime", "macro_stress_score", "rate_environment",
]

FDA_5 = [
    "days_to_next_fda_decision", "days_since_last_fda_decision",
    "last_fda_outcome", "fda_decisions_trailing_12m", "fda_approval_rate_trailing",
]

XGB_FT_PARAMS = {
    "n_estimators":     FT_ESTIMATORS,
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


# ── Printer ────────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    bar = "=" * 64
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_shared_model(sector: str) -> tuple:
    """Load shared sector model pkl. Returns (XGBClassifier, feature_names list)."""
    path = MODELS_DIR / sector / f"xgb_{sector}_shared_v1.pkl"
    if not path.exists():
        raise FileNotFoundError(f"[FAIL] Shared model not found: {path}")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["model"], obj["feature_names"]


def load_baseline_f1(ticker: str) -> float:
    """Load the shared model's per-ticker OOS F1 from Step 7 eval report."""
    path = PROCESSED_DIR / f"{ticker}_eval_report.json"
    if not path.exists():
        print(f"  [WARN] No eval report for {ticker}, using 0.0 as baseline")
        return 0.0
    with open(path) as f:
        r = json.load(f)
    return float(r.get("oos_f1_weighted", 0.0))


def build_ticker_features(
    ticker:        str,
    sector:        str,
    feature_names: list,
) -> tuple:
    """
    Build the full feature matrix for a single ticker, matching the shared
    model's feature_names exactly.

    The ticker one-hot columns are set to fixed values:
      ticker_{THIS_TICKER} = 1,  all other ticker_* = 0
    This preserves feature count for warm-start while conditioning on identity.

    Returns:
      df           : raw DataFrame with __label__ column added
      X_all        : DataFrame, columns = feature_names (all rows, sorted by date)
      y_all        : np.ndarray of encoded labels
    """
    path = PROCESSED_DIR / f"{ticker}_with_events.parquet"
    if not path.exists():
        raise FileNotFoundError(f"[FAIL] Features not found: {path.name}")

    df = pd.read_parquet(path, engine="pyarrow")
    df = df.sort_index()

    # AMZN EPS clipping
    if ticker == "AMZN" and "last_eps_surprise_pct" in df.columns:
        df["last_eps_surprise_pct"] = df["last_eps_surprise_pct"].clip(-500.0, 500.0)

    df["__label__"] = df["dir_1w"].map(LABEL_ENCODE)
    df = df.dropna(subset=["__label__"])
    df["__label__"] = df["__label__"].astype(int)

    # Base features (36 tech + 14 event, +5 FDA for biotech)
    base_feats = SELECTED_36 + EVENT_14 + (FDA_5 if sector == "biotech" else [])

    # Fixed ticker one-hot values
    ticker_cols = [c for c in feature_names if c.startswith("ticker_")]
    X = df[base_feats].copy()
    for col in ticker_cols:
        X[col] = 1 if col == f"ticker_{ticker}" else 0

    X_all = X[feature_names]   # enforce exact column order
    y_all = df["__label__"].values

    return df, X_all, y_all


# ── Walk-forward fine-tuning evaluation ───────────────────────────────────────

def walk_forward_finetune(
    ticker:       str,
    X_oos:        np.ndarray,
    y_oos:        np.ndarray,
    oos_dates:    pd.DatetimeIndex,
    shared_model: XGBClassifier,
    n_splits:     int,
) -> tuple:
    """
    Per-ticker walk-forward fine-tuning evaluation.
    Each fold: warm-start from shared model, add FT_ESTIMATORS trees on fold train data.
    Returns (oos_indices, oos_actual, oos_predicted, oos_proba, fold_f1s).
    """
    unique_dates = np.array(sorted(np.unique(oos_dates)))
    tscv         = TimeSeriesSplit(n_splits=n_splits)
    base_booster = shared_model.get_booster()

    all_indices   = []
    all_actual    = []
    all_predicted = []
    all_proba     = []
    fold_f1s      = []

    print(f"    Walk-forward ({n_splits} folds):")

    for fold, (tr_date_idx, val_date_idx) in enumerate(tscv.split(unique_dates), 1):
        train_dates = set(unique_dates[tr_date_idx])
        val_dates   = set(unique_dates[val_date_idx])

        tr_mask  = np.array([d in train_dates for d in oos_dates])
        val_mask = np.array([d in val_dates   for d in oos_dates])

        X_tr, y_tr = X_oos[tr_mask], y_oos[tr_mask]
        X_val, y_val = X_oos[val_mask], y_oos[val_mask]

        if len(X_tr) < MIN_FOLD_ROWS or len(np.unique(y_tr)) < 3:
            print(f"      Fold {fold}: skipped (too small or missing classes)")
            continue

        weights  = compute_sample_weight("balanced", y_tr)
        ft_model = XGBClassifier(**XGB_FT_PARAMS)
        ft_model.fit(X_tr, y_tr, sample_weight=weights, xgb_model=base_booster)

        preds  = ft_model.predict(X_val)
        probas = ft_model.predict_proba(X_val)
        f1     = f1_score(y_val, preds, average="weighted", zero_division=0)
        fold_f1s.append(f1)

        val_indices = np.where(val_mask)[0]
        all_indices.extend(val_indices.tolist())
        all_actual.extend(y_val.tolist())
        all_predicted.extend(preds.tolist())
        all_proba.extend(probas.tolist())

        print(f"      Fold {fold}: train={len(y_tr):,} val={len(y_val):,}  F1={f1:.4f}")

    return all_indices, all_actual, all_predicted, all_proba, fold_f1s


# ── Final model training ──────────────────────────────────────────────────────

def train_final_finetune(
    X_oos:        np.ndarray,
    y_oos:        np.ndarray,
    shared_model: XGBClassifier,
) -> XGBClassifier:
    """Train final fine-tuned model on ALL pre-holdout rows for this ticker."""
    weights  = compute_sample_weight("balanced", y_oos)
    ft_model = XGBClassifier(**XGB_FT_PARAMS)
    ft_model.fit(X_oos, y_oos, sample_weight=weights, xgb_model=shared_model.get_booster())
    return ft_model


# ── Predictions regeneration ─────────────────────────────────────────────────

def regenerate_predictions(
    ticker:     str,
    sector:     str,
    ft_model:   XGBClassifier,
    df:         pd.DataFrame,
    X_all:      pd.DataFrame,
) -> None:
    """
    Replace proba_* columns in predictions parquet using the fine-tuned model.
    predicted and actual are recalculated from the new probabilities to stay consistent.
    """
    pred_path = PROCESSED_DIR / f"{ticker}_predictions.parquet"
    if not pred_path.exists():
        print(f"  [WARN] {ticker}: predictions parquet not found, skipping regeneration")
        return

    orig     = pd.read_parquet(pred_path, engine="pyarrow")
    X_arr    = X_all.loc[X_all.index.isin(orig.index)].reindex(orig.index)

    # Drop rows with no matching feature data
    valid = X_arr.notna().all(axis=1)
    if not valid.all():
        n_miss = (~valid).sum()
        print(f"  [WARN] {ticker}: {n_miss} prediction rows have no feature data")

    X_valid = X_arr[valid].values
    new_proba = ft_model.predict_proba(X_valid)

    orig.loc[valid, "proba_bear"] = new_proba[:, 0]
    orig.loc[valid, "proba_side"] = new_proba[:, 1]
    orig.loc[valid, "proba_bull"] = new_proba[:, 2]
    # Recalculate predicted from updated probabilities
    orig.loc[valid, "predicted"]  = new_proba.argmax(axis=1).astype(int)

    orig.to_parquet(pred_path, engine="pyarrow", index=True)
    oos_rows  = (orig["split"] == "oos").sum()
    hold_rows = (orig["split"] == "holdout").sum()
    print(f"  [OK]  {ticker}: {oos_rows} OOS + {hold_rows} holdout predictions updated")


# ── Per-ticker orchestration ──────────────────────────────────────────────────

def run_ticker(
    ticker:        str,
    shared_model:  XGBClassifier,
    feature_names: list,
) -> dict:
    """Run full fine-tuning pipeline for one ticker. Returns result dict."""
    sector      = TICKER_SECTOR[ticker]
    model_path  = MODELS_DIR / sector / f"{ticker}_finetuned_v1.pkl"
    baseline_f1 = load_baseline_f1(ticker)

    # ── Skip if already fine-tuned ────────────────────────────────────────────
    if model_path.exists():
        print(f"  [SKIP] {ticker}: {model_path.relative_to(ROOT)} already exists")
        # Load and evaluate to include in final table
        with open(model_path, "rb") as f:
            obj = pickle.load(f)
        return {
            "ticker":       ticker,
            "sector":       sector,
            "baseline_f1":  baseline_f1,
            "ft_oos_f1":    obj.get("oos_f1", float("nan")),
            "ft_hold_f1":   obj.get("holdout_f1", float("nan")),
            "delta":        obj.get("oos_f1", float("nan")) - baseline_f1,
            "saved":        True,
            "reason":       "pre-existing",
            "fold_f1s":     obj.get("fold_f1s", []),
        }

    # ── Build feature matrix ───────────────────────────────────────────────────
    print(f"  Loading features...")
    df, X_all, y_all = build_ticker_features(ticker, sector, feature_names)

    oos_mask     = df.index < HOLDOUT_DATE
    holdout_mask = df.index >= HOLDOUT_DATE
    X_oos     = X_all[oos_mask].values
    y_oos     = y_all[oos_mask]
    X_holdout = X_all[holdout_mask].values
    y_holdout = y_all[holdout_mask]
    oos_dates = df.index[oos_mask]

    print(f"  OOS rows: {len(y_oos):,}  Holdout rows: {len(y_holdout):,}")

    if len(y_oos) < MIN_FOLD_ROWS * 2:
        print(f"  [FAIL] {ticker}: too few OOS rows ({len(y_oos)}) to fine-tune")
        return {
            "ticker": ticker, "sector": sector,
            "baseline_f1": baseline_f1, "ft_oos_f1": float("nan"),
            "ft_hold_f1": float("nan"), "delta": float("nan"),
            "saved": False, "reason": "insufficient data", "fold_f1s": [],
        }

    # Choose folds based on data size
    n_splits = 3 if len(y_oos) < 2000 else 5
    print(f"  Walk-forward folds: {n_splits}")

    # ── Walk-forward fine-tuning evaluation ───────────────────────────────────
    oos_idx, oos_actual, oos_pred, oos_proba, fold_f1s = walk_forward_finetune(
        ticker, X_oos, y_oos, oos_dates, shared_model, n_splits
    )

    if not fold_f1s:
        print(f"  [FAIL] {ticker}: no valid folds produced")
        return {
            "ticker": ticker, "sector": sector,
            "baseline_f1": baseline_f1, "ft_oos_f1": float("nan"),
            "ft_hold_f1": float("nan"), "delta": float("nan"),
            "saved": False, "reason": "no valid folds", "fold_f1s": [],
        }

    ft_oos_f1  = round(f1_score(oos_actual, oos_pred, average="weighted", zero_division=0), 4)
    mean_fold  = round(float(np.mean(fold_f1s)), 4)
    delta      = round(ft_oos_f1 - baseline_f1, 4)

    print(f"  Shared baseline OOS F1: {baseline_f1:.4f}")
    print(f"  Fine-tuned OOS F1:      {ft_oos_f1:.4f}  (mean folds={mean_fold:.4f})")
    print(f"  Delta:                  {delta:+.4f}  "
          f"({'IMPROVEMENT' if delta >= MIN_IMPROVEMENT else 'no gain'})")

    # ── Holdout evaluation (final model trained on all OOS data) ──────────────
    print(f"  Training final model on all pre-holdout data...")
    final_model = train_final_finetune(X_oos, y_oos, shared_model)

    ft_hold_f1 = 0.0
    if len(y_holdout) > 0:
        h_pred    = final_model.predict(X_holdout)
        ft_hold_f1 = round(f1_score(y_holdout, h_pred, average="weighted", zero_division=0), 4)
        print(f"  Holdout F1 (final model): {ft_hold_f1:.4f}")

    # ── Save decision ─────────────────────────────────────────────────────────
    if delta < MIN_IMPROVEMENT:
        print(f"  [SKIP SAVE] {ticker}: delta {delta:+.4f} < threshold {MIN_IMPROVEMENT}")
        return {
            "ticker": ticker, "sector": sector,
            "baseline_f1": baseline_f1, "ft_oos_f1": ft_oos_f1,
            "ft_hold_f1": ft_hold_f1, "delta": delta,
            "saved": False, "reason": f"delta {delta:+.4f} < {MIN_IMPROVEMENT}",
            "fold_f1s": fold_f1s,
        }

    # Save fine-tuned model
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump({
            "model":         final_model,
            "feature_names": feature_names,
            "ticker":        ticker,
            "sector":        sector,
            "holdout_date":  HOLDOUT_DATE,
            "shared_model_f1": baseline_f1,
            "oos_f1":        ft_oos_f1,
            "holdout_f1":    ft_hold_f1,
            "fold_f1s":      fold_f1s,
            "ft_estimators": FT_ESTIMATORS,
        }, f)
    size_kb = model_path.stat().st_size / 1024
    print(f"  [OK] Saved -> {model_path.relative_to(ROOT)}  ({size_kb:.0f} KB)")

    # Regenerate predictions parquet
    regenerate_predictions(ticker, sector, final_model, df, X_all)

    return {
        "ticker": ticker, "sector": sector,
        "baseline_f1": baseline_f1, "ft_oos_f1": ft_oos_f1,
        "ft_hold_f1": ft_hold_f1, "delta": delta,
        "saved": True, "reason": "saved", "fold_f1s": fold_f1s,
    }


# ── Output helpers ────────────────────────────────────────────────────────────

def print_comparison_table(all_results: list) -> None:
    section("FINAL COMPARISON TABLE -- Shared vs Fine-Tuned OOS F1")

    # Sort by delta descending (most improved first)
    ranked = sorted(all_results, key=lambda r: r["delta"] if not np.isnan(r["delta"]) else -99, reverse=True)

    print(f"\n  {'Ticker':6}  {'Sector':7}  {'Shared F1':>10}  "
          f"{'FT F1':>8}  {'Delta':>8}  {'Hold F1':>8}  Status")
    print("  " + "-" * 72)

    for r in ranked:
        ft_f1   = f"{r['ft_oos_f1']:.4f}" if not np.isnan(r.get("ft_oos_f1", float("nan"))) else "   N/A"
        delta_s = f"{r['delta']:+.4f}"     if not np.isnan(r.get("delta", float("nan")))     else "   N/A"
        hold_s  = f"{r.get('ft_hold_f1', float('nan')):.4f}" if not np.isnan(r.get("ft_hold_f1", float("nan"))) else "   N/A"
        status  = "SAVED" if r["saved"] else f"SKIPPED ({r['reason']})"
        print(f"  {r['ticker']:6}  {r['sector']:7}  {r['baseline_f1']:>10.4f}  "
              f"{ft_f1:>8}  {delta_s:>8}  {hold_s:>8}  {status}")

    saved_count   = sum(1 for r in all_results if r["saved"])
    skipped_count = sum(1 for r in all_results if not r["saved"])
    print(f"\n  Saved: {saved_count}   Skipped (no improvement): {skipped_count}")
    print(f"  Improvement threshold: >= {MIN_IMPROVEMENT}")


def save_finetuning_report(all_results: list) -> None:
    DOCS_DIR.mkdir(exist_ok=True)
    out    = DOCS_DIR / "finetuning_report.md"
    today  = pd.Timestamp.now().strftime("%Y-%m-%d")
    ranked = sorted(
        all_results,
        key=lambda r: r["delta"] if not np.isnan(r.get("delta", float("nan"))) else -99,
        reverse=True,
    )

    lines = [
        "# market_ml Step 9 -- Per-Ticker Fine Tuning Report",
        "",
        f"Generated: {today}  ",
        f"Method: XGBoost warm-start (+{FT_ESTIMATORS} estimators from shared sector model)  ",
        f"Save threshold: OOS F1 improvement >= {MIN_IMPROVEMENT}  ",
        f"Walk-forward: 5 folds (3 for tickers with < 2000 pre-holdout rows)",
        "",
        "## Comparison Table",
        "",
        "| Ticker | Sector | Shared F1 | FT OOS F1 | Delta | Holdout F1 | Verdict |",
        "|--------|--------|-----------|-----------|-------|------------|---------|",
    ]

    for r in ranked:
        ft_f1  = f"{r['ft_oos_f1']:.4f}"              if not np.isnan(r.get("ft_oos_f1", float("nan"))) else "N/A"
        delta  = f"{r['delta']:+.4f}"                  if not np.isnan(r.get("delta", float("nan")))     else "N/A"
        hold   = f"{r.get('ft_hold_f1', float('nan')):.4f}" if not np.isnan(r.get("ft_hold_f1", float("nan"))) else "N/A"
        status = "**SAVED**" if r["saved"] else f"skipped ({r['reason']})"
        lines.append(
            f"| {r['ticker']} | {r['sector']} | {r['baseline_f1']:.4f} | "
            f"{ft_f1} | {delta} | {hold} | {status} |"
        )

    saved   = [r for r in all_results if r["saved"]]
    skipped = [r for r in all_results if not r["saved"]]

    lines += [
        "",
        "## Summary",
        "",
        f"- **{len(saved)} tickers saved** (F1 improved by >= {MIN_IMPROVEMENT})",
        f"- **{len(skipped)} tickers skipped** (shared model already optimal or insufficient gain)",
        "",
    ]

    if saved:
        lines.append("### Tickers where fine-tuning helped:")
        lines.append("")
        for r in sorted(saved, key=lambda x: x["delta"], reverse=True):
            folds_str = ", ".join(f"{f:.3f}" for f in r.get("fold_f1s", []))
            lines.append(
                f"- **{r['ticker']}**: shared F1={r['baseline_f1']:.4f} -> "
                f"fine-tuned F1={r['ft_oos_f1']:.4f} (delta={r['delta']:+.4f})  "
                f"Folds: [{folds_str}]"
            )
        lines.append("")

    if skipped:
        lines.append("### Tickers where fine-tuning did not improve:")
        lines.append("")
        for r in sorted(skipped, key=lambda x: x.get("delta", -99)):
            delta_s = f"{r['delta']:+.4f}" if not np.isnan(r.get("delta", float("nan"))) else "N/A"
            lines.append(
                f"- **{r['ticker']}**: delta={delta_s}  ({r['reason']})"
            )
        lines.append("")

    lines += [
        "## Verdict",
        "",
    ]

    if not saved:
        lines.append(
            "Fine-tuning did not improve any ticker above the threshold. "
            "The shared sector model already generalises well across all tickers. "
            "Consider: more aggressive learning rate, more estimators, or richer "
            "ticker-specific features."
        )
    elif len(saved) >= len(all_results) // 2:
        lines.append(
            f"Fine-tuning improved {len(saved)}/{len(all_results)} tickers. "
            "Per-ticker models are now available for those tickers. "
            "The shared model remains the fallback for non-improved tickers."
        )
    else:
        lines.append(
            f"Fine-tuning improved {len(saved)}/{len(all_results)} tickers. "
            "The shared model remains preferred for most tickers. "
            "Saved per-ticker models provide marginal improvement where applicable."
        )

    lines += [
        "",
        "---",
        "",
        "*Generated by src/pipeline/09_finetune.py*",
        "",
    ]

    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  [OK] Saved {out.relative_to(ROOT)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    section("market_ml -- Step 9: Per-Ticker Fine Tuning")
    print(f"  Warm-start: shared sector model + {FT_ESTIMATORS} additional estimators")
    print(f"  Save threshold: OOS F1 improvement >= {MIN_IMPROVEMENT}")
    print(f"  Holdout:        >= {HOLDOUT_DATE.date()} (eval only, never used for training)")

    # Resolve target tickers from CLI args
    args = sys.argv[1:]
    if not args:
        target_tickers = [t for v in SECTORS.values() for t in v["tickers"]]
    else:
        target_tickers = []
        for a in args:
            if a.lower() in SECTORS:
                target_tickers.extend(SECTORS[a.lower()]["tickers"])
            elif a.upper() in TICKER_SECTOR:
                target_tickers.append(a.upper())
            else:
                print(f"  [WARN] Unknown argument '{a}' -- skipping")

    if not target_tickers:
        print("[FAIL] No valid tickers to process")
        sys.exit(1)

    print(f"  Target tickers ({len(target_tickers)}): {target_tickers}")

    # Load shared models (cache to avoid re-loading for each ticker)
    shared_models: dict = {}
    for sector in set(TICKER_SECTOR[t] for t in target_tickers):
        model, feature_names = load_shared_model(sector)
        shared_models[sector] = (model, feature_names)
        n_ft = [c for c in feature_names if c.startswith("ticker_")]
        print(f"  [OK]  Loaded {sector} model: {len(feature_names)} features, "
              f"ticker one-hots: {n_ft}")

    # Process each ticker
    all_results = []
    for ticker in target_tickers:
        sector = TICKER_SECTOR[ticker]
        shared_model, feature_names = shared_models[sector]
        section(f"Fine-tuning: {ticker} ({sector.upper()})")
        try:
            result = run_ticker(ticker, shared_model, feature_names)
            all_results.append(result)
        except Exception as exc:
            print(f"  [FAIL] {ticker}: {exc}")
            all_results.append({
                "ticker": ticker, "sector": sector,
                "baseline_f1": 0.0, "ft_oos_f1": float("nan"),
                "ft_hold_f1": float("nan"), "delta": float("nan"),
                "saved": False, "reason": str(exc), "fold_f1s": [],
            })

    # Summary output
    print_comparison_table(all_results)

    section("Saving Report")
    save_finetuning_report(all_results)

    section("Step 9 Complete")
    saved = [r["ticker"] for r in all_results if r["saved"]]
    skipped = [r["ticker"] for r in all_results if not r["saved"]]
    if saved:
        print(f"  Improved and saved:   {saved}")
    if skipped:
        print(f"  No improvement:       {skipped}")
    print("  Report: docs/finetuning_report.md")
    print("\n  Next step: Update ManthIQ to serve per-ticker models where available\n")


if __name__ == "__main__":
    main()

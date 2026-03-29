"""
refresh_holdout_predictions.py
==============================
Extends existing {TICKER}_predictions.parquet files with new holdout rows
by running inference using the SAVED models (no retraining).

Handles:
  - Tech tickers:    models/tech/xgb_tech_shared_v1.pkl
  - Biotech tickers: models/biotech/xgb_biotech_shared_v1.pkl
  - VRTX:            models/biotech/VRTX_finetuned_v1.pkl  (overrides shared)

Run AFTER re-running steps 1-5 with updated data. Does NOT touch OOS rows.

Usage:
    C:\\Users\\borra\\anaconda3\\python.exe src\\pipeline\\refresh_holdout_predictions.py
"""

import sys
import pickle
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config.tickers import SECTORS, TICKER_SECTOR

# ── Constants ──────────────────────────────────────────────────────────────────
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR    = ROOT / "models"
HOLDOUT_DATE  = pd.Timestamp("2024-01-01")

LABEL_ENCODE = {-1: 0, 0: 1, 1: 2}    # dir_1w raw -> encoded
TARGET       = "dir_1w"
AMZN_EPS_CLIP = 500.0


# ── Load models ────────────────────────────────────────────────────────────────
def load_model(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Build feature matrix for one ticker ───────────────────────────────────────
def build_feature_matrix(
    df: pd.DataFrame,
    ticker: str,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Build the exact feature matrix expected by the saved model.
    Sets ticker_{TICKER}=1, all other ticker one-hot columns=0.
    """
    # Identify base features vs ticker one-hot columns from feature_names
    ticker_cols = [c for c in feature_names if c.startswith("ticker_")]
    base_cols   = [c for c in feature_names if not c.startswith("ticker_")]

    missing = [c for c in base_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{ticker}] Missing base features: {missing}")

    X = df[base_cols].copy()

    # Add ticker one-hot columns: 1 for this ticker, 0 for all others
    for col in ticker_cols:
        expected_ticker = col.replace("ticker_", "")
        X[col] = int(expected_ticker == ticker)

    # Reindex to exact feature_names order
    X = X[feature_names]

    return X


# ── Per-ticker refresh ─────────────────────────────────────────────────────────
def refresh_ticker(
    ticker:        str,
    sector:        str,
    model_payload: dict,
    before_counts: dict,
) -> int:
    """
    Extend {TICKER}_predictions.parquet with new holdout rows.
    Returns number of new rows appended.
    """
    pred_path    = PROCESSED_DIR / f"{ticker}_predictions.parquet"
    events_path  = PROCESSED_DIR / f"{ticker}_with_events.parquet"

    if not pred_path.exists():
        print(f"  [FAIL] {ticker}: predictions parquet not found -- skipping")
        return 0
    if not events_path.exists():
        print(f"  [FAIL] {ticker}: _with_events parquet not found -- run step 5 first")
        return 0

    model         = model_payload["model"]
    feature_names = model_payload["feature_names"]

    # Load existing predictions to find which dates are already covered
    existing = pd.read_parquet(pred_path)
    before_counts[ticker] = int((existing["split"] == "holdout").sum())
    existing_dates = set(existing.index)

    # Load full feature+event data
    df_all = pd.read_parquet(events_path)

    # AMZN EPS clipping (matches step 6)
    if ticker == "AMZN" and "last_eps_surprise_pct" in df_all.columns:
        df_all["last_eps_surprise_pct"] = df_all["last_eps_surprise_pct"].clip(
            -AMZN_EPS_CLIP, AMZN_EPS_CLIP
        )

    # New holdout rows = >= HOLDOUT_DATE AND not already predicted
    df_new = df_all[
        (df_all.index >= HOLDOUT_DATE) &
        (~df_all.index.isin(existing_dates))
    ].copy()

    if len(df_new) == 0:
        print(f"  [SKIP] {ticker}: no new holdout rows to add")
        return 0

    # Build feature matrix
    X_new = build_feature_matrix(df_new, ticker, feature_names)

    # Encode actual labels
    y_raw     = df_new[TARGET]
    y_encoded = y_raw.map(LABEL_ENCODE)

    missing_labels = y_encoded.isna().sum()
    if missing_labels > 0:
        print(f"  [WARN] {ticker}: {missing_labels} rows with unmappable dir_1w -- dropping")
        valid = y_encoded.notna()
        df_new    = df_new[valid]
        X_new     = X_new[valid]
        y_encoded = y_encoded[valid]

    if len(df_new) == 0:
        print(f"  [SKIP] {ticker}: all new rows had invalid labels")
        return 0

    # Predict
    preds  = model.predict(X_new.values)
    probas = model.predict_proba(X_new.values)

    # Build new rows dataframe
    new_df = pd.DataFrame(index=df_new.index)
    new_df["ticker_id"]  = ticker
    new_df[TARGET]       = y_raw[df_new.index].values
    new_df["actual"]     = y_encoded.values.astype(int)
    new_df["predicted"]  = preds.astype(int)
    new_df["split"]      = "holdout"
    new_df["proba_bear"] = probas[:, 0]
    new_df["proba_side"] = probas[:, 1]
    new_df["proba_bull"] = probas[:, 2]

    # Append and save
    combined = pd.concat([existing, new_df]).sort_index()
    combined.to_parquet(pred_path, engine="pyarrow", index=True)

    return len(new_df)


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    print("\n" + "=" * 60)
    print("  refresh_holdout_predictions.py")
    print("  Extending holdout rows using SAVED models (no retraining)")
    print("=" * 60)
    print(f"  Holdout cutoff: {HOLDOUT_DATE.date()}")

    # Load models
    tech_payload    = load_model(MODELS_DIR / "tech"    / "xgb_tech_shared_v1.pkl")
    biotech_payload = load_model(MODELS_DIR / "biotech" / "xgb_biotech_shared_v1.pkl")
    vrtx_payload    = load_model(MODELS_DIR / "biotech" / "VRTX_finetuned_v1.pkl")
    print("  [OK]  Loaded tech shared model")
    print("  [OK]  Loaded biotech shared model")
    print("  [OK]  Loaded VRTX fine-tuned model")

    sector_models = {
        "tech":    tech_payload,
        "biotech": biotech_payload,
    }

    before_counts: dict = {}
    total_new = 0

    print(f"\n{'Ticker':<8}  {'Sector':<8}  {'Model':<16}  {'Before':>6}  {'Added':>6}  {'After':>6}")
    print("-" * 60)

    for sector_name, sector_cfg in SECTORS.items():
        for ticker in sector_cfg["tickers"]:
            # VRTX uses fine-tuned model; all others use sector shared model
            if ticker == "VRTX":
                payload = vrtx_payload
                model_label = "VRTX_finetuned"
            else:
                payload = sector_models[sector_name]
                model_label = f"{sector_name}_shared"

            before: dict = {}
            n_added = refresh_ticker(ticker, sector_name, payload, before)
            b = before.get(ticker, 0)
            a = b + n_added
            total_new += n_added
            print(f"  {ticker:<6}  {sector_name:<8}  {model_label:<16}  {b:>6}  {n_added:>6}  {a:>6}")

    print("-" * 60)
    print(f"\n  Total new holdout rows added: {total_new:,}")

    # Print final holdout counts
    print("\n  Final holdout row counts:")
    for sector_name, sector_cfg in SECTORS.items():
        for ticker in sector_cfg["tickers"]:
            p = PROCESSED_DIR / f"{ticker}_predictions.parquet"
            df = pd.read_parquet(p)
            hold = df[df["split"] == "holdout"]
            if len(hold):
                print(f"    {ticker}: {len(hold)} holdout rows  "
                      f"{hold.index[0].date()} to {hold.index[-1].date()}")
            else:
                print(f"    {ticker}: 0 holdout rows")

    print("\n  Next step: src/pipeline/10_backtest.py\n")


if __name__ == "__main__":
    main()

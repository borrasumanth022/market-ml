"""
Step 2 — Feature engineering (ticker-agnostic)
===============================================
Takes raw OHLCV for any ticker and computes technical indicators + price-derived features.
Reproduces the 57-column feature matrix from aapl_ml, from which the 36 model features
are a validated subset (see .claude/skills/feature-engineering/SKILL.md).

Feature groups:
  A. Price & volume transforms   -- returns, volume z-score, 52w range position
  B. Trend indicators            -- SMA, EMA, MACD, cross flags
  C. Momentum indicators         -- RSI, Stochastic, Rate of Change
  D. Volatility indicators       -- Bollinger Bands, ATR, historical vol
  E. Market microstructure       -- candle body, shadows, gap, HL range
  F. Calendar features           -- day-of-week, month, quarter-end flags

Output: data/processed/{TICKER}_features.parquet

Usage:
    # All tickers (skips already-processed)
    C:\\Users\\borra\\anaconda3\\python.exe src\\pipeline\\02_features.py

    # Single ticker
    C:\\Users\\borra\\anaconda3\\python.exe src\\pipeline\\02_features.py AAPL
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

from config.tickers import SECTORS, TICKER_SECTOR

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DIR       = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# ── The 36 selected model features (validated in aapl_ml Phase 2) ────────────
# Full feature matrix has 57 cols; these are the subset passed to ML models.
# Everything else is stored in the parquet but excluded at training time.
SELECTED_36 = [
    "atr_pct", "bb_pct", "bb_width", "candle_body", "candle_dir",
    "close_vs_sma10", "close_vs_sma100", "close_vs_sma20", "close_vs_sma200",
    "close_vs_sma50", "cross_50_200", "day_of_week", "gap_pct", "hl_range_pct",
    "hvol_10d", "hvol_21d", "hvol_63d", "is_month_end", "is_month_start",
    "is_quarter_end", "lower_shadow", "macd_hist", "macd_signal", "month",
    "price_52w_pct", "return_1d", "return_2d", "return_5d", "roc_10", "roc_21",
    "rsi_14", "rsi_7", "stoch_d", "stoch_k", "upper_shadow", "volume_zscore",
]


# ══════════════════════════════════════════════════════════════════════════════
# Feature group functions
# ══════════════════════════════════════════════════════════════════════════════

def add_price_transforms(df: pd.DataFrame) -> pd.DataFrame:
    df["log_close"]     = np.log(df["close"])
    df["return_1d"]     = df["close"].pct_change(1)
    df["return_2d"]     = df["close"].pct_change(2)
    df["return_5d"]     = df["close"].pct_change(5)
    df["log_return_1d"] = np.log(df["close"] / df["close"].shift(1))

    vol_mean            = df["volume"].rolling(20).mean()
    vol_std             = df["volume"].rolling(20).std()
    df["volume_zscore"] = (df["volume"] - vol_mean) / (vol_std + 1e-9)

    hi52                = df["high"].rolling(252).max()
    lo52                = df["low"].rolling(252).min()
    df["price_52w_pct"] = (df["close"] - lo52) / (hi52 - lo52 + 1e-9)

    return df


def add_trend(df: pd.DataFrame) -> pd.DataFrame:
    for w in [10, 20, 50, 100, 200]:
        df[f"sma_{w}"]         = df["close"].rolling(w).mean()
        df[f"close_vs_sma{w}"] = df["close"] / df[f"sma_{w}"] - 1

    for w in [12, 26]:
        df[f"ema_{w}"] = df["close"].ewm(span=w, adjust=False).mean()

    df["macd"]        = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]
    df["cross_50_200"] = (df["sma_50"] > df["sma_200"]).astype(int)

    return df


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - 100 / (1 + rs)


def add_momentum(df: pd.DataFrame) -> pd.DataFrame:
    df["rsi_14"] = _rsi(df["close"], 14)
    df["rsi_7"]  = _rsi(df["close"],  7)

    for w in [5, 10, 21]:
        df[f"roc_{w}"] = df["close"].pct_change(w) * 100

    lo14          = df["low"].rolling(14).min()
    hi14          = df["high"].rolling(14).max()
    df["stoch_k"] = 100 * (df["close"] - lo14) / (hi14 - lo14 + 1e-9)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()
    df["williams_r"] = -100 * (hi14 - df["close"]) / (hi14 - lo14 + 1e-9)

    return df


def add_volatility(df: pd.DataFrame) -> pd.DataFrame:
    sma20          = df["close"].rolling(20).mean()
    std20          = df["close"].rolling(20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma20
    df["bb_pct"]   = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)

    prev_close   = df["close"].shift(1)
    tr           = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr_14"] / df["close"]

    log_ret = np.log(df["close"] / df["close"].shift(1))
    for w in [10, 21, 63]:
        df[f"hvol_{w}d"] = log_ret.rolling(w).std() * np.sqrt(252)

    return df


def add_microstructure(df: pd.DataFrame) -> pd.DataFrame:
    candle_range       = (df["high"] - df["low"]).replace(0, np.nan)
    df["candle_body"]  = (df["close"] - df["open"]).abs() / candle_range
    df["upper_shadow"] = (df["high"] - df[["close", "open"]].max(axis=1)) / candle_range
    df["lower_shadow"] = (df[["close", "open"]].min(axis=1) - df["low"]) / candle_range
    df["candle_dir"]   = np.sign(df["close"] - df["open"])
    df["gap_pct"]      = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)
    df["hl_range_pct"] = (df["high"] - df["low"]) / df["close"]
    return df


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    idx                   = df.index
    df["day_of_week"]     = idx.dayofweek
    df["month"]           = idx.month
    df["quarter"]         = idx.quarter
    df["is_month_end"]    = idx.is_month_end.astype(int)
    df["is_month_start"]  = idx.is_month_start.astype(int)
    df["is_quarter_end"]  = idx.is_quarter_end.astype(int)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def build_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Compute all feature groups for a single ticker DataFrame."""
    df = df.copy().sort_index()

    df = add_price_transforms(df)
    df = add_trend(df)
    df = add_momentum(df)
    df = add_volatility(df)
    df = add_microstructure(df)
    df = add_calendar(df)

    before = len(df)
    df = df.dropna(subset=["sma_200"])
    dropped = before - len(df)
    print(f"  {ticker}: dropped {dropped} warm-up rows, {len(df)} remain")

    return df


def process_ticker(ticker: str) -> bool:
    """
    Load raw data, compute features, save. Returns True on success.
    Skips if output already exists.
    """
    raw_path = RAW_DIR / f"{ticker}_daily_raw.parquet"
    out_path = PROCESSED_DIR / f"{ticker}_features.parquet"

    if out_path.exists():
        size_kb = out_path.stat().st_size / 1024
        print(f"  {ticker}: already exists ({size_kb:.0f} KB) -- skipping")
        return True

    if not raw_path.exists():
        print(f"  {ticker}: raw data not found at {raw_path.relative_to(ROOT)}")
        print(f"           Run 01_fetch_data.py first.")
        return False

    raw = pd.read_parquet(raw_path)
    print(f"  {ticker}: {len(raw)} raw rows loaded")

    features = build_features(raw, ticker)

    # Sanity check: verify 36 selected features are all present
    missing = [f for f in SELECTED_36 if f not in features.columns]
    if missing:
        raise ValueError(f"[{ticker}] Missing expected model features: {missing}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_path)
    size_kb = out_path.stat().st_size / 1024
    print(f"  {ticker}: saved {features.shape[0]} rows x {features.shape[1]} cols "
          f"-> {out_path.relative_to(ROOT)} ({size_kb:.0f} KB)")

    return True


# ══════════════════════════════════════════════════════════════════════════════
# Entry points
# ══════════════════════════════════════════════════════════════════════════════

def run_all() -> None:
    all_tickers = [
        (sector, ticker)
        for sector, cfg in SECTORS.items()
        for ticker in cfg["tickers"]
    ]

    print(f"\nmarket_ml -- feature engineering")
    print(f"Tickers  : {[t for _, t in all_tickers]}")
    print(f"Output   : {PROCESSED_DIR.relative_to(ROOT)}/\n")

    results = {"ok": [], "skip": [], "err": []}

    for sector, ticker in all_tickers:
        out_path = PROCESSED_DIR / f"{ticker}_features.parquet"
        print(f"[{sector.upper()}] {ticker}")

        if out_path.exists():
            size_kb = out_path.stat().st_size / 1024
            print(f"  Already exists ({size_kb:.0f} KB) -- skipping\n")
            results["skip"].append(ticker)
            continue

        try:
            success = process_ticker(ticker)
            if success:
                results["ok"].append(ticker)
        except Exception as exc:
            print(f"  ERROR: {exc}\n")
            results["err"].append((ticker, str(exc)))
        print()

    print("=" * 60)
    print(f"Done.  Computed: {len(results['ok'])}  "
          f"Skipped: {len(results['skip'])}  "
          f"Errors: {len(results['err'])}")
    if results["err"]:
        print("\nFailed tickers:")
        for t, msg in results["err"]:
            print(f"  {t}: {msg}")
    print("=" * 60)
    print("\nNext step: src/pipeline/03_labels.py\n")


def run_single(ticker: str) -> None:
    ticker = ticker.upper()
    if ticker not in TICKER_SECTOR:
        print(f"ERROR: '{ticker}' not in config/tickers.py")
        print(f"Known: {sorted(TICKER_SECTOR.keys())}")
        sys.exit(1)

    sector = TICKER_SECTOR[ticker]
    print(f"\nFeature engineering: {ticker} ({sector})\n")
    success = process_ticker(ticker)
    if not success:
        sys.exit(1)
    print("\nDone.\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_single(sys.argv[1])
    else:
        run_all()

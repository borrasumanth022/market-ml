"""
Step 3 — Build prediction labels (ticker-agnostic)
===================================================
Computes forward returns at five horizons and converts them to classification
and regression targets. Runs over all tickers in config/tickers.py.

Label design (inherited from aapl_ml Phase 1):
  ret_{h}     — raw % forward return (regression)
  dir_{h}     — 3-class direction: +1 Bull / 0 Sideways / -1 Bear  (classification)
  bin_{h}     — binary up/down: 1 / 0  (simpler classification)
  adj_ret_{h} — vol-adjusted return = ret / hvol_21d  (regime-aware regression)

Horizons: 1w (5d), 1m (21d), 3m (63d), 6m (126d), 1y (252d)
Direction threshold: ±2%  (returns within the band are labelled Sideways)

Output: data/processed/{TICKER}_labeled.parquet  (features + labels in one file)

Usage:
    C:\\Users\\borra\\anaconda3\\python.exe src\\pipeline\\03_labels.py        # all tickers
    C:\\Users\\borra\\anaconda3\\python.exe src\\pipeline\\03_labels.py AAPL   # single ticker
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

from config.tickers import SECTORS, TICKER_SECTOR

# ── Config ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR       = ROOT / "data" / "processed"
DIRECTION_THRESHOLD = 0.02          # ±2% sideways band
HORIZONS            = {             # label suffix -> trading days forward
    "1w":  5,
    "1m":  21,
    "3m":  63,
    "6m":  126,
    "1y":  252,
}
MAX_HORIZON         = max(HORIZONS.values())   # 252 — rows dropped from tail


# ── Label builders ─────────────────────────────────────────────────────────────

def forward_return(close: pd.Series, n: int) -> pd.Series:
    """% return n trading days ahead. Uses shift(-n) — intentional lookahead for labels only."""
    return close.shift(-n) / close - 1


def direction_label(fwd_ret: pd.Series, threshold: float) -> pd.Series:
    """
    +1  Bull     — return > +threshold
     0  Sideways — abs(return) <= threshold
    -1  Bear     — return < -threshold
    """
    labels = pd.Series(0, index=fwd_ret.index, dtype=int)
    labels[fwd_ret >  threshold] =  1
    labels[fwd_ret < -threshold] = -1
    return labels


def binary_direction(fwd_ret: pd.Series) -> pd.Series:
    """Simple up / down (1 / 0). No sideways band."""
    return (fwd_ret > 0).astype(int)


# ── Main label builder ─────────────────────────────────────────────────────────

def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    df   = df.copy()
    close = df["close"]

    for name, n_days in HORIZONS.items():
        fwd = forward_return(close, n_days)

        df[f"ret_{name}"]     = fwd
        df[f"dir_{name}"]     = direction_label(fwd, DIRECTION_THRESHOLD)
        df[f"bin_{name}"]     = binary_direction(fwd)

        if "hvol_21d" in df.columns:
            df[f"adj_ret_{name}"] = fwd / (df["hvol_21d"] + 1e-9)

    # Drop only the rows where dir_1w cannot be computed (last 5 rows).
    # Longer-horizon labels (1m..1y) will be 0 (Sideways) for the most recent
    # rows but those horizons are not used in steps 6-10 which target dir_1w only.
    df = df.iloc[:-HORIZONS["1w"]]
    return df


# ── Summary printer ────────────────────────────────────────────────────────────

def label_summary(df: pd.DataFrame, ticker: str) -> None:
    total = len(df)
    header = f"  {ticker} label distribution ({total:,} rows)"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name in HORIZONS:
        col = f"dir_{name}"
        if col not in df.columns:
            continue
        counts   = df[col].value_counts().sort_index()
        up_pct   = counts.get( 1, 0) / total * 100
        side_pct = counts.get( 0, 0) / total * 100
        dn_pct   = counts.get(-1, 0) / total * 100
        print(f"    {name:>3}  Bull {up_pct:5.1f}%  Side {side_pct:5.1f}%  Bear {dn_pct:5.1f}%")
    print()


# ── Per-ticker processing ──────────────────────────────────────────────────────

def process_ticker(ticker: str) -> bool:
    feat_path = PROCESSED_DIR / f"{ticker}_features.parquet"
    out_path  = PROCESSED_DIR / f"{ticker}_labeled.parquet"

    if out_path.exists():
        size_kb = out_path.stat().st_size / 1024
        print(f"  {ticker}: already exists ({size_kb:.0f} KB) -- skipping\n")
        return True

    if not feat_path.exists():
        print(f"  {ticker}: features file missing -- run 02_features.py first\n")
        return False

    df = pd.read_parquet(feat_path)
    print(f"  {ticker}: {len(df)} feature rows loaded")

    labeled = build_labels(df)

    label_cols = [c for c in labeled.columns
                  if c.startswith(("ret_", "dir_", "bin_", "adj_ret_"))]
    print(f"  {ticker}: {len(labeled)} rows after trimming tail  "
          f"({len(label_cols)} label columns)")

    label_summary(labeled, ticker)

    labeled.to_parquet(out_path)
    size_kb = out_path.stat().st_size / 1024
    print(f"  {ticker}: saved -> {out_path.relative_to(ROOT)}  ({size_kb:.0f} KB)\n")
    return True


# ── Entry points ───────────────────────────────────────────────────────────────

def run_all() -> None:
    all_tickers = [
        (sector, ticker)
        for sector, cfg in SECTORS.items()
        for ticker in cfg["tickers"]
    ]

    print(f"\nmarket_ml -- label engineering")
    print(f"Tickers   : {[t for _, t in all_tickers]}")
    print(f"Horizons  : {list(HORIZONS.keys())}")
    print(f"Threshold : +/-{DIRECTION_THRESHOLD*100:.0f}% sideways band")
    print(f"Output    : {PROCESSED_DIR.relative_to(ROOT)}/\n")

    results = {"ok": [], "skip": [], "err": []}

    for sector, ticker in all_tickers:
        out_path = PROCESSED_DIR / f"{ticker}_labeled.parquet"
        print(f"[{sector.upper()}] {ticker}")

        if out_path.exists():
            size_kb = out_path.stat().st_size / 1024
            print(f"  Already exists ({size_kb:.0f} KB) -- skipping\n")
            results["skip"].append(ticker)
            continue

        try:
            success = process_ticker(ticker)
            results["ok"].append(ticker) if success else results["err"].append((ticker, "processing failed"))
        except Exception as exc:
            print(f"  ERROR: {exc}\n")
            results["err"].append((ticker, str(exc)))

    print("=" * 60)
    print(f"Done.  Labeled: {len(results['ok'])}  "
          f"Skipped: {len(results['skip'])}  "
          f"Errors: {len(results['err'])}")
    if results["err"]:
        print("\nFailed tickers:")
        for t, msg in results["err"]:
            print(f"  {t}: {msg}")
    print("=" * 60)
    print("\nNext step: src/pipeline/04_events.py\n")


def run_single(ticker: str) -> None:
    ticker = ticker.upper()
    if ticker not in TICKER_SECTOR:
        print(f"ERROR: '{ticker}' not in config/tickers.py")
        print(f"Known: {sorted(TICKER_SECTOR.keys())}")
        sys.exit(1)
    sector = TICKER_SECTOR[ticker]
    print(f"\nLabel engineering: {ticker} ({sector})\n")
    success = process_ticker(ticker)
    if not success:
        sys.exit(1)
    print("Done.\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_single(sys.argv[1])
    else:
        run_all()

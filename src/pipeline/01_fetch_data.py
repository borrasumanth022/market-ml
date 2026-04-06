"""
Step 1 — Fetch raw OHLCV data for all tickers in all sectors.

Behaviour:
  No existing file  : full download from 1995-01-01 to today.
  File exists, stale: incremental update — fetch only new rows, append, dedup.
  File exists, current: skip entirely (saves ~2 min on Monday runs).

"Current" = last date in parquet is within 4 calendar days of today.
This handles Monday runs (last date = Friday), holiday weekends, etc.

Saves each ticker to data/raw/{TICKER}_daily_raw.parquet.

Usage:
    # Fetch / update all tickers (default)
    C:\\Users\\borra\\anaconda3\\python.exe src/pipeline/01_fetch_data.py

    # Fetch / update a single ticker
    C:\\Users\\borra\\anaconda3\\python.exe src/pipeline/01_fetch_data.py NVDA
"""

import sys
from pathlib import Path

# Allow imports from project root
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import yfinance as yf
import pandas as pd
from datetime import datetime

from config.tickers import SECTORS, TICKER_SECTOR

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_START = "1995-01-01"
END_DATE      = datetime.today().strftime("%Y-%m-%d")
RAW_DIR       = ROOT / "data" / "raw"


# ── Core fetch ────────────────────────────────────────────────────────────────
def fetch_ticker(ticker: str, start: str = DEFAULT_START, end: str = END_DATE,
                 min_rows: int = 100) -> pd.DataFrame:
    """
    Download daily OHLCV for a single ticker. Returns a clean DataFrame.
    min_rows=1 for incremental fetches (may return just a few days of data).
    """
    print(f"  Downloading {ticker}  {start} to {end} ...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for {ticker} — check the ticker symbol.")

    # Flatten MultiIndex columns: ('Close', 'AAPL') -> 'close'
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]

    df.index.name = "date"
    df.index = pd.to_datetime(df.index)

    assert "close" in df.columns, f"[{ticker}] Missing 'close' column"
    df = df.dropna(subset=["close"])

    if len(df) < min_rows:
        raise ValueError(
            f"[{ticker}] Only {len(df)} rows returned — possible bad ticker or very recent IPO."
        )

    print(f"  {ticker}: {len(df)} trading days  "
          f"({df.index[0].date()} to {df.index[-1].date()})")
    return df


# ── Save ──────────────────────────────────────────────────────────────────────
def save_ticker(df: pd.DataFrame, ticker: str) -> Path:
    """Save DataFrame to data/raw/{TICKER}_daily_raw.parquet."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / f"{ticker}_daily_raw.parquet"
    df.to_parquet(out)
    size_kb = out.stat().st_size / 1024
    print(f"  Saved  -> {out.relative_to(ROOT)}  ({size_kb:.1f} KB)")
    return out


# ── Run ───────────────────────────────────────────────────────────────────────
def _is_current(last_date: pd.Timestamp) -> bool:
    """
    Return True if last_date is recent enough to skip an update.
    Threshold: 4 calendar days covers Mon run (last=Fri), holiday weekends, etc.
    """
    today = pd.Timestamp.today().normalize()
    return (today - last_date) <= pd.Timedelta(days=4)


def fetch_all() -> None:
    """
    Fetch / incrementally update every ticker defined in config/tickers.py.

    Three cases per ticker:
      - No file      : full download from DEFAULT_START.
      - Stale file   : incremental fetch from last_date+1, append, dedup.
      - Current file : skip (last date within 4 calendar days of today).
    """
    all_tickers = [
        (sector, ticker)
        for sector, cfg in SECTORS.items()
        for ticker in cfg["tickers"]
    ]

    print(f"\nmarket_ml -- fetch / update all tickers")
    print(f"Sectors  : {list(SECTORS.keys())}")
    print(f"Tickers  : {[t for _, t in all_tickers]}")
    print(f"End date : {END_DATE}")
    print(f"Output   : {RAW_DIR.relative_to(ROOT)}/\n")

    results = {"full": [], "incr": [], "skip": [], "err": []}

    for sector, ticker in all_tickers:
        out_path = RAW_DIR / f"{ticker}_daily_raw.parquet"
        print(f"[{sector.upper()}] {ticker}")

        if out_path.exists():
            existing = pd.read_parquet(out_path, engine="pyarrow")
            existing.index = pd.to_datetime(existing.index)
            last_date = existing.index.max()

            if _is_current(last_date):
                print(f"  [OK]   already current ({last_date.date()}) -- skipping\n")
                results["skip"].append(ticker)
                continue

            # Incremental: fetch only new rows
            fetch_start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            print(f"  [INCR] last date {last_date.date()} -- fetching {fetch_start} to {END_DATE}")
            try:
                new_df = fetch_ticker(ticker, start=fetch_start, end=END_DATE, min_rows=1)
                combined = pd.concat([existing, new_df])
                combined = combined[~combined.index.duplicated(keep="last")].sort_index()
                new_rows = len(combined) - len(existing)
                save_ticker(combined, ticker)
                print(f"  [OK]   +{new_rows} new rows  "
                      f"({last_date.date()} -> {combined.index.max().date()})\n")
                results["incr"].append(ticker)
            except Exception as exc:
                print(f"  [ERR]  {exc}\n")
                results["err"].append((ticker, str(exc)))
        else:
            # Full download
            try:
                df = fetch_ticker(ticker)
                save_ticker(df, ticker)
                results["full"].append(ticker)
            except Exception as exc:
                print(f"  [ERR]  {exc}\n")
                results["err"].append((ticker, str(exc)))
            print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"Done.  Full: {len(results['full'])}  "
          f"Incremental: {len(results['incr'])}  "
          f"Current: {len(results['skip'])}  "
          f"Errors: {len(results['err'])}")
    if results["err"]:
        print("\nFailed tickers:")
        for t, msg in results["err"]:
            print(f"  {t}: {msg}")
    print("=" * 60)
    print("\nNext step: src/pipeline/02_features.py\n")


def fetch_single(ticker: str) -> None:
    """Fetch / incrementally update one ticker by name (CLI usage)."""
    ticker = ticker.upper()
    if ticker not in TICKER_SECTOR:
        print(f"ERROR: '{ticker}' not found in config/tickers.py")
        print(f"Known tickers: {sorted(TICKER_SECTOR.keys())}")
        sys.exit(1)

    sector   = TICKER_SECTOR[ticker]
    out_path = RAW_DIR / f"{ticker}_daily_raw.parquet"
    print(f"\nFetching / updating {ticker} ({sector})")

    if out_path.exists():
        existing  = pd.read_parquet(out_path, engine="pyarrow")
        existing.index = pd.to_datetime(existing.index)
        last_date = existing.index.max()
        if _is_current(last_date):
            print(f"  [OK] already current ({last_date.date()}) -- nothing to do\n")
            return
        fetch_start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"  [INCR] fetching {fetch_start} to {END_DATE}")
        new_df   = fetch_ticker(ticker, start=fetch_start, end=END_DATE, min_rows=1)
        combined = pd.concat([existing, new_df])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        save_ticker(combined, ticker)
    else:
        df = fetch_ticker(ticker)
        save_ticker(df, ticker)

    print(f"\nDone.\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        fetch_single(sys.argv[1])
    else:
        fetch_all()

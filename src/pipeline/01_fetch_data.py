"""
Step 1 — Fetch raw OHLCV data for all tickers in all sectors.

Downloads full history from 1995-01-01 (or IPO date if later) to today.
Saves each ticker to data/raw/{TICKER}_daily_raw.parquet.

Usage:
    # Fetch all tickers (default)
    C:\\Users\\borra\\anaconda3\\python.exe src/pipeline/01_fetch_data.py

    # Fetch a single ticker
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
def fetch_ticker(ticker: str, start: str = DEFAULT_START, end: str = END_DATE) -> pd.DataFrame:
    """Download daily OHLCV for a single ticker. Returns a clean DataFrame."""
    print(f"  Downloading {ticker}  {start} to {end} ...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for {ticker} — check the ticker symbol.")

    # Flatten MultiIndex columns: ('Close', 'AAPL') → 'close'
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]

    df.index.name = "date"
    df.index = pd.to_datetime(df.index)

    assert "close" in df.columns, f"[{ticker}] Missing 'close' column"
    df = df.dropna(subset=["close"])

    if len(df) < 100:
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
def fetch_all() -> None:
    """Fetch every ticker defined in config/tickers.py."""
    all_tickers = [
        (sector, ticker)
        for sector, cfg in SECTORS.items()
        for ticker in cfg["tickers"]
    ]

    print(f"\nmarket_ml — fetch all tickers")
    print(f"Sectors  : {list(SECTORS.keys())}")
    print(f"Tickers  : {[t for _, t in all_tickers]}")
    print(f"End date : {END_DATE}")
    print(f"Output   : {RAW_DIR.relative_to(ROOT)}/\n")

    results = {"ok": [], "skip": [], "err": []}

    for sector, ticker in all_tickers:
        out_path = RAW_DIR / f"{ticker}_daily_raw.parquet"
        print(f"[{sector.upper()}] {ticker}")

        if out_path.exists():
            size_kb = out_path.stat().st_size / 1024
            print(f"  Already exists ({size_kb:.1f} KB) — skipping. "
                  f"Delete the file to re-fetch.\n")
            results["skip"].append(ticker)
            continue

        try:
            df = fetch_ticker(ticker)
            save_ticker(df, ticker)
            results["ok"].append(ticker)
        except Exception as exc:
            print(f"  ERROR: {exc}\n")
            results["err"].append((ticker, str(exc)))
        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"Done.  Fetched: {len(results['ok'])}  "
          f"Skipped: {len(results['skip'])}  "
          f"Errors: {len(results['err'])}")
    if results["err"]:
        print("\nFailed tickers:")
        for t, msg in results["err"]:
            print(f"  {t}: {msg}")
    print("=" * 60)
    print("\nNext step: src/pipeline/02_features.py\n")


def fetch_single(ticker: str) -> None:
    """Fetch one ticker by name (CLI usage)."""
    ticker = ticker.upper()
    if ticker not in TICKER_SECTOR:
        print(f"ERROR: '{ticker}' not found in config/tickers.py")
        print(f"Known tickers: {sorted(TICKER_SECTOR.keys())}")
        sys.exit(1)

    sector = TICKER_SECTOR[ticker]
    print(f"\nFetching {ticker} ({sector})")
    df = fetch_ticker(ticker)
    save_ticker(df, ticker)
    print(f"\nDone.\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        fetch_single(sys.argv[1])
    else:
        fetch_all()

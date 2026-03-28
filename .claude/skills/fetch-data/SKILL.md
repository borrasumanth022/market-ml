# Skill: fetch-data

How we fetch and validate ticker data in market_ml.

## What this skill covers
- Running `src/pipeline/01_fetch_data.py` correctly
- Adding new tickers to the registry
- Validating downloads before moving on
- Handling edge cases (short history, encoding issues)

---

## The fetch script

Located at `src/pipeline/01_fetch_data.py`. It reads `config/tickers.py` as its sole source of truth — no hardcoding of ticker lists anywhere else.

```bash
# Fetch all tickers (skips already-downloaded files)
C:\Users\borra\anaconda3\python.exe src\pipeline\01_fetch_data.py

# Fetch one specific ticker
C:\Users\borra\anaconda3\python.exe src\pipeline\01_fetch_data.py NVDA

# Force re-fetch a ticker (delete the file first)
del data\raw\NVDA_daily_raw.parquet
C:\Users\borra\anaconda3\python.exe src\pipeline\01_fetch_data.py NVDA
```

## Output
- `data/raw/{TICKER}_daily_raw.parquet` — one file per ticker, all lowercase OHLCV columns
- Columns: `open, high, low, close, volume` (all auto-adjusted for splits and dividends)
- Index: DatetimeIndex named `date`

## Skip-if-exists behaviour
By design, the script skips tickers that already have a file in `data/raw/`. This means:
- Re-running the script is safe — it won't re-download or overwrite
- To update a ticker's data, delete the file and re-run
- To update ALL tickers: `del data\raw\*_daily_raw.parquet` then re-run

## Adding a new ticker
1. Add the ticker to `config/tickers.py` under the appropriate sector
2. Run the fetch script — it picks up the new ticker automatically
3. Validate (see below)

## Validation checklist
After any fetch run, verify:

```python
import pandas as pd
from pathlib import Path

for f in sorted(Path("data/raw").glob("*_daily_raw.parquet")):
    df = pd.read_parquet(f)
    ticker = f.stem.split("_")[0]
    assert "close" in df.columns, f"{ticker}: missing close"
    assert len(df) > 100, f"{ticker}: suspiciously few rows ({len(df)})"
    assert df.index.name == "date", f"{ticker}: wrong index name"
    assert not df["close"].isna().any(), f"{ticker}: NaN in close"
    print(f"{ticker:6s}  {len(df):5d} rows  {df.index[0].date()} to {df.index[-1].date()}")
```

## Known edge cases

**Short history tickers** — META (2012), MRNA (2018) have far fewer rows than 1995 tickers.
Walk-forward folds will have fewer periods. Account for this in training scripts by checking
`len(df) >= 5 * min_fold_size` before training.

**Windows encoding** — print statements must not use Unicode arrows (`→`). Use `->` or `to`
instead. The Windows cp1252 terminal will crash otherwise. This is already fixed in the script;
keep it in mind when adding new print statements.

**yfinance MultiIndex columns** — yfinance returns `('Close', 'AAPL')` style columns for single
tickers. The fetch script already flattens these to plain `close`. Don't change this.

**Auto-adjust=True** — we always pass `auto_adjust=True` to yfinance. This adjusts for dividends
and splits. Never mix adjusted and unadjusted data in the same model.

## What "full history from 1995" means
We request `start="1995-01-01"` for all tickers. yfinance silently clips to the IPO date if the
ticker didn't exist yet. So NVDA starts 1999 even though we asked for 1995. This is correct
behaviour — do not treat it as an error.

# Data Conventions — market_ml

## Ticker registry — single source of truth: config/tickers.py

```python
SECTORS = {
    "tech":    ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META"],
    "biotech": ["LLY",  "MRNA", "BIIB", "REGN",  "VRTX"],
}
ALL_TICKERS = [t for tickers in SECTORS.values() for t in tickers]
```

All 11 tickers must be uppercase. Pipeline scripts import from here — never hardcode.

## Parquet file naming — always `{TICKER}_{type}.parquet`

| File | Location | Description |
|------|----------|-------------|
| `{TICKER}_daily_raw.parquet` | `data/raw/` | Raw OHLCV from Yahoo Finance |
| `{TICKER}_features.parquet` | `data/processed/` | 36 technical features |
| `{TICKER}_labeled.parquet` | `data/processed/` | Features + 20 label columns |
| `{TICKER}_with_events.parquet` | `data/processed/` | Features + labels + event features |
| `{TICKER}_predictions.parquet` | `data/processed/` | OOS model predictions |

All tickers use uppercase in filenames: `AAPL_features.parquet`, never `aapl_features.parquet`.

## Parquet schema requirements

Every processed parquet must:
- Have a `DatetimeIndex` named `date` (no timezone)
- Use lowercase_snake_case column names
- Contain no `inf` or `-inf` values (replace with `NaN` before saving)
- Be sorted ascending by date
- Use `engine="pyarrow"` for read/write

## Event file structure

```
data/events/
  universal/
    macro_events.parquet        ← FRED: FEDFUNDS, CPI, UNRATE, GDP (all tickers)
  biotech/
    {TICKER}_fda_events.parquet ← openFDA drugsfda (LLY, MRNA, BIIB, REGN, VRTX)
  stocks/
    {TICKER}_earnings.parquet   ← earnings dates + EPS surprise (all tickers)
```

## Feature naming conventions

| Category | Convention | Examples |
|----------|-----------|---------|
| Returns | `return_{N}d` | `return_1d`, `return_5d`, `return_21d` |
| Indicators | `{ind}_{param}` | `rsi_14`, `macd_hist`, `bb_pct` |
| Volatility | `hvol_{N}d` | `hvol_10d`, `hvol_21d`, `hvol_63d` |
| Moving avg | `close_vs_sma{N}` | `close_vs_sma50`, `close_vs_sma200` |
| Events | `days_to_{event}`, `days_since_{event}` | `days_to_next_earnings` |
| Interactions | `{f1}_{f2}_interaction` | `rate_vol_regime` |

**Prohibited column name prefixes**: `future_`, `fwd_`, `_forward`, `next_`, `tomorrow`, `lead_`

## Expected column counts (as of Step 6)

| File type | Tech | Biotech | Notes |
|-----------|------|---------|-------|
| `_features` | 36 | 36 | Same feature set for all tickers |
| `_labeled` | 57 | 57 | + 20 label cols + 1 ticker_id |
| `_with_events` | 91 | 97 | + 14 event + 6 one-hot (tech) or + 14 + 5 FDA + 5 one-hot (biotech) |
| `_predictions` | 8 | 8 | ticker_id, dir_1w, actual, predicted, split, proba_bear, proba_side, proba_bull |

## Predictions parquet schema (required columns)

```
ticker_id    : str   — ticker symbol e.g. "AAPL"
dir_1w       : int   — true label (0=Bear, 1=Sideways, 2=Bull)
actual       : int   — same as dir_1w (kept for ManthIQ compatibility)
predicted    : int   — model predicted class
split        : int   — walk-forward fold number (0-4)
proba_bear   : float — P(class=0)
proba_side   : float — P(class=1)
proba_bull   : float — P(class=2)
```

Index: `DatetimeIndex` named `date`.

## Model file convention

```
models/
  tech/
    xgb_tech_shared_v1.pkl          ← current champion
    xgb_tech_shared_v1_backup.pkl   ← pre-retrain checkpoint
  biotech/
    xgb_biotech_shared_v1.pkl
    xgb_biotech_shared_v1_backup.pkl
```

Models are **gitignored** — never commit pkl files. Reproduce via `06_train.py`.

## Known data quirks

- **AMZN EPS**: `last_eps_surprise_pct` clipped at ±500% before training (raw: -777% to +3900%)
- **MRNA**: IPO ~2018 — only ~1,600 rows (use 3 walk-forward folds, not 5)
- **META**: IPO 2012 — shorter history than other Tech tickers (use 3 folds)
- **GOOGL**: pre-split prices (~$2800+) before 2022-07-18 — Yahoo Finance auto-adjusts

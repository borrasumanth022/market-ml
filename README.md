# market_ml

Multi-sector, multi-ticker ML pipeline for stock market prediction.

Extends the AAPL-only pipeline (aapl_ml) to cover 11 tickers across tech and biotech sectors.

## Sectors & Tickers

| Sector  | Tickers                             |
|---------|-------------------------------------|
| tech    | AAPL, MSFT, NVDA, GOOGL, AMZN, META |
| biotech | LLY, MRNA, BIIB, REGN, VRTX         |

## Quick start

```bash
# Fetch all tickers
C:\Users\borra\anaconda3\python.exe src\pipeline\01_fetch_data.py

# Fetch one ticker
C:\Users\borra\anaconda3\python.exe src\pipeline\01_fetch_data.py NVDA
```

## Pipeline

| Script | Description | Status |
|--------|-------------|--------|
| `src/pipeline/01_fetch_data.py` | Download OHLCV for all tickers | COMPLETE |
| `src/pipeline/02_features.py` | Technical feature engineering | TODO |
| `src/pipeline/03_labels.py` | Forward-return labels | TODO |
| `src/pipeline/04_train.py` | XGBoost baseline per ticker | TODO |
| `src/pipeline/05_events.py` | Event data collection | TODO |

## Config

All tickers and sectors are defined in `config/tickers.py`. Add new tickers there and the pipeline picks them up automatically.

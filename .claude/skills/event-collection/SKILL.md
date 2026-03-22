# Skill: event-collection

How we collect, store, and structure events in market_ml.

## What events are for
Events link market behaviour to real-world catalysts. The goal is not to predict events
but to characterise the market environment at any given date:
- How many days until/since the last Fed rate decision?
- Is CPI rising or falling?
- Did this company just beat earnings?
- Is there an FDA PDUFA date coming up?

Each of these becomes a numeric feature fed to the model alongside technical features.

---

## Event storage layout

```
data/events/
├── universal/      <- Applies to ALL tickers (all sectors share this)
├── tech/           <- Tech sector only
├── biotech/        <- Biotech sector only
└── stocks/         <- Ticker-specific (one file per ticker)
```

**Why this hierarchy?** When building event features for a ticker, merge:
`universal events` + `sector events` (from SECTORS config) + `stock-specific events`

This avoids storing duplicate macro data per ticker and makes it easy to add a new
sector without touching universal data.

---

## Universal events (`data/events/universal/`)

These come from FRED (Federal Reserve Economic Data). No API key needed — use the
`fredgraph.csv` endpoint (already proven in aapl_ml/src/08_events.py).

| Event type | FRED series | File | Frequency |
|-----------|-------------|------|-----------|
| `fed_rate` | FEDFUNDS | `fed_rate.parquet` | Monthly |
| `cpi` | CPIAUCSL | `cpi.parquet` | Monthly |
| `gdp` | GDP | `gdp.parquet` | Quarterly |
| `unemployment` | UNRATE | `unemployment.parquet` | Monthly |

FRED URL pattern (no key required):
```
https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS
```

Schema for each universal event file:
```
date            DatetimeIndex
value           float64    (raw FRED value)
change_1m       float64    (month-over-month change)
change_3m       float64    (3-month change)
```

---

## Tech sector events (`data/events/tech/`)

These require manual curation or news scraping. Start with:

| Event type | File | Source | Notes |
|-----------|------|--------|-------|
| `ai_narrative` | `ai_narrative.parquet` | Manual curation | Key dates: ChatGPT launch (2022-11-30), GPT-4 (2023-03-14), etc. |
| `semiconductor_cycle` | `semiconductor_cycle.parquet` | SEMI industry data | Chip shipment up/down cycles |
| `antitrust` | `antitrust.parquet` | News/SEC filings | DOJ/FTC actions against FAANG |
| `cloud_growth` | — | Derived from earnings | AWS/Azure/GCP revenue growth in earnings |

For now, the AI narrative events are the most impactful given NVDA and META's dependency.
Build `ai_narrative.parquet` first.

---

## Biotech sector events (`data/events/biotech/`)

| Event type | File | Source | Notes |
|-----------|------|--------|-------|
| `fda_pdufa` | `fda_pdufa.parquet` | FDA PDUFA calendar | Published quarterly by FDA |
| `clinical_trials` | `clinical_trials.parquet` | ClinicalTrials.gov | Phase 3 readouts for key drugs |
| `drug_approvals` | `drug_approvals.parquet` | FDA press releases | Approval/rejection dates |

PDUFA dates are the most predictable biotech event. The FDA publishes a calendar at
https://www.fda.gov/patients/fast-track-breakthrough-therapy-accelerated-approval-priority-review/pdufa-calendar
These are high-impact: stocks often move 20–50% on PDUFA date.

---

## Stock-specific events (`data/events/stocks/`)

One file per ticker: `data/events/stocks/{TICKER}_events.parquet`

Schema (consistent across all stock event files):
```
date            DatetimeIndex
event_type      str    (e.g., "earnings", "product_launch", "split")
event_subtype   str    (e.g., "beat", "miss", "iphone", "2-for-1")
magnitude       float  (e.g., EPS surprise %)
direction       int    (1=positive, -1=negative, 0=neutral)
source          str    ("yfinance", "manual", "fred")
description     str    (human-readable summary)
```

For AAPL this already exists (migrated from aapl_ml). For other tickers, build from:
- `yfinance.Ticker.get_earnings_dates()` — earnings history (capped at 100 rows)
- `yfinance.Ticker.get_earnings_history()` — EPS surprise %
- Manually: splits, major product launches

---

## Event feature engineering (when building model features)

When generating event features for a ticker, load and merge:
1. `data/events/universal/*.parquet` — always included
2. `data/events/{sector}/*.parquet` — based on `TICKER_SECTOR[ticker]` from config
3. `data/events/stocks/{TICKER}_events.parquet` — stock-specific

Then for each event type, compute:
- `days_to_next_{event}` — forward-looking within each fold's training window (not lookahead — computed from events calendar, not from future prices)
- `days_since_last_{event}` — backward-looking, always safe
- `{event}_magnitude` — the surprise or magnitude at the last event
- Regime flags — `rate_environment (-1/0/1)`, `inflation_regime (-1/0/1)`

**Critical**: `days_to_next_*` is NOT lookahead. The event dates are known in advance
(earnings calendar, PDUFA calendar). We are not using future price data.

---

## Inherited event features (from aapl_ml, applicable to tech tickers)

From aapl_ml Phase 3, these 16 features proved most useful:

```python
EVENT_FEATURES = [
    "days_to_next_earnings", "days_since_last_earnings", "has_earnings_data",
    "last_eps_surprise_pct", "earnings_streak",
    "fed_rate_level", "fed_rate_change_1m", "fed_rate_change_3m",
    "cpi_yoy_change", "unemployment_level", "unemployment_change_3m",
    "days_to_next_product_event", "days_since_last_product_event",
    "is_iphone_cycle",   # AAPL-specific — skip for other tickers
    "rate_environment", "inflation_regime",
]
```

Top performers by SHAP (from aapl_ml):
1. `fed_rate_change_3m` — strongest for Bear AND Bull
2. `cpi_yoy_change` — strongest for Bear
3. `last_eps_surprise_pct` — strongest for Bull
4. `rate_vol_regime` (interaction) — rate change × hvol_63d, rank #3 overall

---

## NaN sentinel values
Pre-history rows (before sufficient event data exists) get sentinel values, not NaN:
- `days_since_last_earnings = 90` (before first yfinance earnings record)
- `days_since_last_product_event = 180` (before first tracked product event)
- `last_eps_surprise_pct = 0.0` (neutral, not missing)

This avoids dropping pre-2005 rows and keeps the model's training window as large as possible.

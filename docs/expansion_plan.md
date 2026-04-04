# market_ml Expansion Plan

## Current state (as of 2026-04-03)

| Sector  | Tickers | Model features | Status |
|---------|---------|---------------|--------|
| Tech    | AAPL, MSFT, NVDA, GOOGL, AMZN, META | 65 (36+14+9+6 one-hot) | Live |
| Biotech | LLY, MRNA, BIIB, REGN, VRTX        | 69 (36+14+9+5FDA+5 one-hot) | Live |

11 tickers total. Weekly signal generator live at `src/pipeline/12_signal_generator.py`.
Paper trading threshold 0.55 (holdout win rate 69%, 42 trades -- needs ~200 to confirm).

---

## Phase 1 -- Expand existing sectors

### Tech additions (6 tickers)
AMD, TSLA, CRM, ADBE, INTC, ORCL

### Biotech additions (5 tickers)
ABBV, BMY, GILD, AMGN, PFE

### Steps required
1. Add tickers to `config/tickers.py` (sector lists only -- no other config changes)
2. Run Steps 1-5 for new tickers only (existing tickers skip via exists-check):
   - `01_fetch_data.py`
   - `02_features.py`
   - `03_labels.py`
   - `04_events.py`
   - `05_event_features.py`
3. Retrain both sector shared models via `06_train.py` (combines all tickers in sector)
4. Re-run `07_evaluate.py` and `10_backtest.py` to update reports

### Expected outcome after Phase 1
- Tech: 12 tickers, larger shared model training pool
- Biotech: 10 tickers, FDA pipeline already handles all standard biotech
- Total: 22 tickers, roughly 2x weekly signal opportunities

### Notes
- TSLA: high short interest, earnings volatility -- may widen iron condor wings
- INTC: declining market cap, restructuring -- check data continuity pre-2020
- ABBV, PFE: large cap pharma with heavy FDA approval history -- FDA events pre-loaded
- One-hot encoding dimensions increase with more tickers (6 -> 12 for tech, 5 -> 10 for biotech)

---

## Phase 2 -- Financials sector

### Tickers
JPM, GS, BAC, MS, WFC

### Event sources
| Source | Series | Frequency | Coverage | Notes |
|--------|--------|-----------|----------|-------|
| Fed funds rate | FEDFUNDS (FRED) | Monthly | 1954+ | Already in EVENT_14 |
| Yield curve (T10Y2Y) | T10Y2Y (FRED) | Daily | 1976+ | Already in REGIME_9 |
| HY credit spread | BAMLH0A0HYM2 (FRED) | Daily | 1996+ | New: add to financials event features |

### Credit spread feature engineering
```
credit_spread_level      BAMLH0A0HYM2 daily value (OAS, basis points)
credit_spread_change_1m  21-day change in credit spread
credit_spread_zscore     63-day rolling z-score (sentinel 0.0 pre-1996)
```

FRED fetch: `https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAMLH0A0HYM2`
History: 1996-12-31 to present. Pre-1996 rows: fill with sentinel 0.0.

### Steps required
1. Add `financials` sector to `config/tickers.py` with event tags `["fed_rate", "yield_curve", "credit_spreads"]`
2. Fetch BAMLH0A0HYM2 in `04_events.py` (save to `data/events/universal/credit_spreads.parquet`)
3. Add `add_credit_spread_features()` function to `05_event_features.py`
4. Add `CREDIT_3` feature group to `06_train.py` feature set for financials sector
5. Run Steps 1-6 for financials
6. Update `12_signal_generator.py` to load the new financials model

### Feature count (financials)
36 tech + 14 event + 9 regime + 3 credit spread + 5 one-hot = 67 features

---

## Phase 3 -- Energy sector

### Tickers
XOM, CVX, COP, SLB, EOG

### Event sources
| Source | Series / URL | Frequency | Coverage | Notes |
|--------|-------------|-----------|----------|-------|
| WTI crude oil | DCOILWTICO (FRED) | Daily | 1986+ | Cushing spot price |
| Natural gas | DHHNGSP (FRED) | Weekly | 1997+ | Henry Hub spot |
| Baker Hughes rig count | bakerhughes.com/rig-count | Weekly | 1987+ | Free CSV download |

### Energy feature engineering
```
wti_close               WTI crude spot price ($/barrel)
wti_change_1m           21-day change in WTI price
wti_zscore_63d          63-day rolling z-score (sentinel 0.0 pre-1986)
natgas_close            Henry Hub natural gas spot ($/MMBtu)
natgas_change_1m        21-day change
rig_count_us            Total US rig count (weekly, forward-filled daily)
rig_count_change_4w     4-week change in rig count
```

FRED fetches:
- WTI:    `https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILWTICO`
- NatGas: `https://fred.stlouisfed.org/graph/fredgraph.csv?id=DHHNGSP`

Baker Hughes rig count:
- URL: `https://rigcount.bakerhughes.com/static-files/...` (weekly Excel/CSV)
- Fallback: scrape the public CSV linked from bakerhughes.com/rig-count
- Frequency: weekly (Friday); forward-fill to daily
- Sentinel: 0 for pre-1987 rows (or use the mean as neutral)

### Steps required
1. Add `energy` sector to `config/tickers.py`
2. Add `fetch_wti()`, `fetch_natgas()`, `fetch_rig_count()` to `04_events.py`
   (save to `data/events/energy/`)
3. Add `add_energy_features()` to `05_event_features.py`
4. Add `ENERGY_7` feature group to `06_train.py` for energy sector
5. Run Steps 1-6 for energy
6. Update `12_signal_generator.py` for energy model

### Feature count (energy)
36 tech + 14 event + 9 regime + 7 energy commodity + 5 one-hot = 71 features

### Notes
- SLB (Schlumberger): oilfield services, high rig count correlation -- key sector signal
- WTI has occasional gaps (holidays, trading halts); forward-fill is safe
- Natural gas is highly seasonal; consider adding month as interaction term

---

## Full target state

| Sector     | Tickers | Count | Model features |
|------------|---------|-------|---------------|
| Tech       | AAPL, MSFT, NVDA, GOOGL, AMZN, META, AMD, TSLA, CRM, ADBE, INTC, ORCL | 12 | 36+14+9+12 one-hot = 71 |
| Biotech    | LLY, MRNA, BIIB, REGN, VRTX, ABBV, BMY, GILD, AMGN, PFE | 10 | 36+14+9+5FDA+10 one-hot = 74 |
| Financials | JPM, GS, BAC, MS, WFC | 5 | 36+14+9+3 credit+5 one-hot = 67 |
| Energy     | XOM, CVX, COP, SLB, EOG | 5 | 36+14+9+7 commodity+5 one-hot = 71 |
| **Total**  | **32 tickers** | | |

Expected weekly signals at threshold 0.55: ~3-5 per week (up from 0-2).
At ~4 signals/week, target 200 holdout trades reached in ~50 weeks (by early 2027).

---

## Implementation order

```
Phase 1a: Config + data for new tech tickers (AMD, TSLA, CRM, ADBE, INTC, ORCL)
Phase 1b: Config + data for new biotech tickers (ABBV, BMY, GILD, AMGN, PFE)
Phase 1c: Retrain tech + biotech shared models, update backtest and signal generator
Phase 2:  Financials sector (JPM, GS, BAC, MS, WFC + BAMLH0A0HYM2)
Phase 3:  Energy sector (XOM, CVX, COP, SLB, EOG + WTI/NatGas/RigCount)
```

Each phase is independent and can be validated before starting the next.
The signal generator (`12_signal_generator.py`) loads models by sector label --
adding a new sector only requires adding its model path and load logic there.

---

## config/tickers.py target state

```python
SECTORS = {
    "tech": {
        "tickers": [
            "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META",
            "AMD", "TSLA", "CRM", "ADBE", "INTC", "ORCL",
        ],
        ...
    },
    "biotech": {
        "tickers": [
            "LLY", "MRNA", "BIIB", "REGN", "VRTX",
            "ABBV", "BMY", "GILD", "AMGN", "PFE",
        ],
        ...
    },
    "financials": {
        "tickers": ["JPM", "GS", "BAC", "MS", "WFC"],
        "description": "Large cap US banks and investment banks",
        "universal_events": ["fed_rate", "yield_curve", "credit_spreads"],
        "sector_events": [],
    },
    "energy": {
        "tickers": ["XOM", "CVX", "COP", "SLB", "EOG"],
        "description": "Large cap US oil and gas",
        "universal_events": ["fed_rate"],
        "sector_events": ["wti_crude", "natural_gas", "rig_count"],
    },
}
```

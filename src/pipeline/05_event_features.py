"""
Step 5 -- Event feature engineering (ticker-agnostic)
=====================================================
Joins macro + earnings + FDA + energy events onto each ticker's labeled feature matrix.

Features built for ALL 32 tickers:
  A. Earnings    -- proximity, EPS surprise, beat/miss streak, has_data flag
  B. Macro       -- fed rate level/changes, CPI YoY, unemployment level/change
  C. Regime      -- rate_environment, inflation_regime, macro_stress_score

Features built for BIOTECH tickers only (LLY, MRNA, BIIB, REGN, VRTX, ABBV, BMY, GILD, AMGN, PFE):
  D. FDA actions -- proximity (days to/since), outcome, trailing decision count,
                   is_pdufa_month flag, rolling approval rate

Features built for FINANCIALS tickers only (JPM, GS, BAC, MS, WFC):
  D2. Credit spread -- BAMLH0A0HYM2 OAS level, 21d change, 63d z-score

Features built for ENERGY tickers only (XOM, CVX, COP, SLB, EOG):
  D3. Energy commodities -- WTI price/changes/zscore, natgas price/changes,
                            rig count and 4-week change (ENERGY_9)

Data sources:
  - Earnings      : yfinance get_earnings_dates() per ticker
  - Macro         : FRED re-fetched as daily levels (FEDFUNDS, CPIAUCSL, UNRATE)
  - FDA           : data/events/biotech/fda_events.parquet (from 04_events.py)
  - Credit spread : data/events/financials/credit_spreads.parquet
  - Energy        : data/events/energy/energy_events.parquet

Lookahead policy:
  - All features are strictly backward-looking on each trading day
  - FRED monthly data forward-filled daily (each day sees last published obs)
  - days_to_next_earnings uses known earnings CALENDAR dates (not future prices)
  - is_pdufa_month uses known FDA event month (PDUFA dates are pre-announced)
  - macro_stress_score uses per-ticker global z-score (no cross-date leakage)
  - shift(-n) is ONLY used in 03_labels.py for label construction

NaN sentinels (pre-history rows):
  - days_since_last_earnings  -> CAP_EARNINGS (90)   when no history yet
  - days_to_next_earnings     -> CAP_EARNINGS (90)   when no future date known
  - last_eps_surprise_pct     -> 0.0
  - earnings_streak           -> 0
  - FDA proximity             -> 365  (neutral large value, not NaN)
  - energy natgas pre-1997    -> 0.0
  - energy rig count gaps     -> 0.0

Output: data/processed/{TICKER}_with_events.parquet for all 32 tickers

Usage:
    C:\\Users\\borra\\anaconda3\\python.exe src\\pipeline\\05_event_features.py
    C:\\Users\\borra\\anaconda3\\python.exe src\\pipeline\\05_event_features.py AAPL
"""

import io
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from config.tickers import SECTORS, TICKER_SECTOR

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR        = ROOT / "data" / "processed"
FDA_EVENTS           = ROOT / "data" / "events" / "biotech"          / "fda_events.parquet"
CREDIT_SPREADS       = ROOT / "data" / "events" / "financials"       / "credit_spreads.parquet"
ENERGY_EVENTS        = ROOT / "data" / "events" / "energy"           / "energy_events.parquet"
CONSUMER_STAPLES_EVENTS = ROOT / "data" / "events" / "consumer_staples" / "consumer_staples_events.parquet"

# ── Constants (inherited from aapl_ml settings) ───────────────────────────────
CAP_EARNINGS     = 90      # sentinel days when no earnings history exists
CAP_FDA          = 365     # sentinel days when no FDA history exists
INFLATION_HIGH   = 4.0    # CPI YoY % -> high regime
INFLATION_LOW    = 1.5    # CPI YoY % -> low regime
RATE_RISING_BPS  =  10    # 3m fed change bps -> rising
RATE_FALLING_BPS = -10    # 3m fed change bps -> falling
FDA_WINDOW_DAYS  = 365    # trailing days for FDA activity counts
FDA_APPROVAL_WIN = 1095   # trailing days for approval rate (3 years)
FRED_BASE        = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"


# ── Low-level utilities ───────────────────────────────────────────────────────

def _to_day_ints(dates) -> np.ndarray:
    """Convert any date-like input to integer days since 1970-01-01 (datetime64[D])."""
    return pd.DatetimeIndex(dates).normalize().values.astype("datetime64[D]").astype("int64")


def days_to_next(trading_dates: pd.DatetimeIndex, event_dates) -> np.ndarray:
    """Calendar days from each trading date to the next event (strictly after)."""
    trade_d = _to_day_ints(trading_dates)
    event_d = np.sort(_to_day_ints(event_dates))
    idx     = np.searchsorted(event_d, trade_d, side="right")
    result  = np.full(len(trade_d), np.nan)
    mask    = idx < len(event_d)
    result[mask] = (event_d[idx[mask]] - trade_d[mask]).astype(float)
    return result


def days_since_last(trading_dates: pd.DatetimeIndex, event_dates) -> np.ndarray:
    """Calendar days from most recent past event to each trading date."""
    trade_d = _to_day_ints(trading_dates)
    event_d = np.sort(_to_day_ints(event_dates))
    idx     = np.searchsorted(event_d, trade_d, side="right") - 1
    result  = np.full(len(trade_d), np.nan)
    mask    = idx >= 0
    result[mask] = (trade_d[mask] - event_d[idx[mask]]).astype(float)
    return result


def trailing_event_count(trading_dates: pd.DatetimeIndex,
                         event_dates, window_days: int) -> np.ndarray:
    """Count of events in the trailing [window_days] calendar days for each trading date."""
    trade_d = _to_day_ints(trading_dates)
    event_d = np.sort(_to_day_ints(event_dates))
    hi = np.searchsorted(event_d, trade_d, side="right")
    lo = np.searchsorted(event_d, trade_d - window_days, side="left")
    return (hi - lo).astype(float)


def align_to_daily(monthly: pd.Series, daily_index: pd.DatetimeIndex) -> pd.Series:
    """Forward-fill a monthly/quarterly series onto a daily trading-day index."""
    combined = monthly.reindex(monthly.index.union(daily_index)).sort_index().ffill()
    return combined.reindex(daily_index)


def zscore_global(s: pd.Series) -> pd.Series:
    """Global z-score over the full series (no future label info, just macro scale)."""
    return (s - s.mean()) / (s.std() + 1e-9)


def section(title: str) -> None:
    print(f"\n  {'-' * 56}")
    print(f"  {title}")
    print(f"  {'-' * 56}")


# ── FRED fetch (once, reused for all tickers) ─────────────────────────────────

def fetch_fred(series_id: str, start: str = "1993-01-01") -> pd.Series:
    url = FRED_BASE.format(series_id)
    r   = requests.get(url, timeout=20)
    r.raise_for_status()
    df  = pd.read_csv(
        io.StringIO(r.text),
        parse_dates=["observation_date"],
        index_col="observation_date",
    )
    df.columns = [series_id]
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
    s = df[series_id].dropna().sort_index()
    return s[s.index >= start]


def prefetch_fred() -> dict:
    """Download all FRED series once. Returns dict of monthly pd.Series."""
    print("  Pre-fetching FRED series (shared across all tickers) ...")
    fred = {}
    for series_id in ["FEDFUNDS", "CPIAUCSL", "UNRATE"]:
        print(f"    {series_id} ...", end=" ", flush=True)
        fred[series_id] = fetch_fred(series_id)
        print(f"OK  ({len(fred[series_id])} obs, {fred[series_id].index[-1].date()})")
    return fred


# ── Earnings fetch (per ticker, via yfinance) ─────────────────────────────────

def fetch_earnings(ticker: str) -> pd.DataFrame:
    """
    Fetch EPS earnings dates via yfinance.
    Returns DataFrame with index=date, columns=[magnitude (surprise %)].
    Returns empty DataFrame if unavailable.
    """
    try:
        yf_ticker = yf.Ticker(ticker)
        eps_df    = yf_ticker.get_earnings_dates(limit=100)
        if eps_df is None or eps_df.empty:
            return pd.DataFrame()
        eps_df = eps_df.dropna(subset=["Reported EPS"])
        eps_df.index = pd.to_datetime(eps_df.index).tz_localize(None).normalize()
        eps_df = eps_df.sort_index()
        eps_df["magnitude"] = eps_df["Surprise(%)"].fillna(
            (eps_df["Reported EPS"] - eps_df["EPS Estimate"])
            / eps_df["EPS Estimate"].abs().replace(0, np.nan) * 100
        )
        return eps_df[["magnitude"]].dropna(subset=["magnitude"])
    except Exception as exc:
        print(f"    WARNING: earnings fetch failed for {ticker}: {exc}")
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# Feature group builders
# ══════════════════════════════════════════════════════════════════════════════

def add_earnings_features(feat: pd.DataFrame, ticker: str) -> None:
    """Group A — earnings proximity, surprise, streak."""
    trading_dates = feat.index

    print(f"    Fetching earnings (yfinance) ...", end=" ", flush=True)
    time.sleep(0.2)   # polite yfinance pacing
    eps_df = fetch_earnings(ticker)

    if eps_df.empty:
        print("no data")
        feat["days_to_next_earnings"]   = CAP_EARNINGS
        feat["days_since_last_earnings"] = CAP_EARNINGS
        feat["has_earnings_data"]       = 0
        feat["last_eps_surprise_pct"]   = 0.0
        feat["earnings_streak"]         = 0
        return

    eps_dates = eps_df.index.unique()
    print(f"{len(eps_df)} quarters  ({eps_dates.min().date()} to {eps_dates.max().date()})")

    # Days to / since (capped at sentinel)
    feat["days_to_next_earnings"] = np.where(
        np.isnan(days_to_next(trading_dates, eps_dates.values)),
        CAP_EARNINGS,
        np.minimum(days_to_next(trading_dates, eps_dates.values), CAP_EARNINGS),
    )
    raw_since = days_since_last(trading_dates, eps_dates.values)
    feat["days_since_last_earnings"] = np.where(
        np.isnan(raw_since), CAP_EARNINGS, np.minimum(raw_since, CAP_EARNINGS)
    )
    feat["has_earnings_data"] = (~np.isnan(
        days_since_last(trading_dates, eps_dates.values)
    )).astype(int)

    # last_eps_surprise_pct — forward-fill from last known event
    surp = pd.Series(np.nan, index=trading_dates)
    for dt, row in eps_df.iterrows():
        if dt in surp.index:
            surp.loc[dt] = row["magnitude"]
    surp = surp.groupby(level=0).last()
    feat["last_eps_surprise_pct"] = surp.ffill().fillna(0.0)

    # earnings_streak — consecutive beats (+) / misses (-)
    streak = 0
    streak_by_date = {}
    for dt, row in eps_df.sort_index().iterrows():
        mag = row["magnitude"]
        if pd.isna(mag) or mag == 0:
            streak = 0
        elif mag > 0:
            streak = max(streak, 0) + 1
        else:
            streak = min(streak, 0) - 1
        streak_by_date[dt] = streak

    streak_s = pd.Series(np.nan, index=trading_dates)
    for dt, val in streak_by_date.items():
        if dt in streak_s.index:
            streak_s.loc[dt] = val
    streak_s = streak_s.groupby(level=0).last()
    feat["earnings_streak"] = streak_s.ffill().fillna(0).astype(int)


def add_macro_features(feat: pd.DataFrame, fred: dict) -> None:
    """Group B — fed rate, CPI, unemployment (all from pre-fetched FRED)."""
    trading_dates = feat.index
    fedfunds = fred["FEDFUNDS"]
    cpi      = fred["CPIAUCSL"]
    unrate   = fred["UNRATE"]

    # Fed funds rate
    feat["fed_rate_level"]     = align_to_daily(fedfunds, trading_dates).values
    ff_chg_1m = (fedfunds - fedfunds.shift(1)) * 100     # bps
    ff_chg_3m = (fedfunds - fedfunds.shift(3)) * 100     # bps
    feat["fed_rate_change_1m"] = align_to_daily(ff_chg_1m, trading_dates).values
    feat["fed_rate_change_3m"] = align_to_daily(ff_chg_3m, trading_dates).values

    # CPI — year-over-year change
    cpi_yoy = ((cpi - cpi.shift(12)) / cpi.shift(12) * 100).dropna()
    feat["cpi_yoy_change"] = align_to_daily(cpi_yoy, trading_dates).values

    # Unemployment
    feat["unemployment_level"]     = align_to_daily(unrate, trading_dates).values
    ur_chg_3m = unrate - unrate.shift(3)
    feat["unemployment_change_3m"] = align_to_daily(ur_chg_3m, trading_dates).values


def add_regime_features(feat: pd.DataFrame) -> None:
    """Group C — rate_environment, inflation_regime, macro_stress_score."""
    rate_3m = feat["fed_rate_change_3m"]
    cpi_yoy = feat["cpi_yoy_change"]
    urate_3m = feat["unemployment_change_3m"]

    # rate_environment: rising=+1, stable=0, falling=-1
    rate_env = pd.Series(0, index=feat.index, dtype=int)
    rate_env[rate_3m >  RATE_RISING_BPS]  =  1
    rate_env[rate_3m <  RATE_FALLING_BPS] = -1
    rate_env[rate_3m.isna()]              =  0
    feat["rate_environment"] = rate_env

    # inflation_regime: high=+1, normal=0, low=-1
    inf_reg = pd.Series(0, index=feat.index, dtype=int)
    inf_reg[cpi_yoy >= INFLATION_HIGH] =  1
    inf_reg[cpi_yoy <  INFLATION_LOW]  = -1
    inf_reg[cpi_yoy.isna()]            =  0
    feat["inflation_regime"] = inf_reg

    # macro_stress_score: equal-weighted z-score composite
    # Each component z-scored over this ticker's own history (no cross-ticker leakage)
    feat["macro_stress_score"] = (
        zscore_global(rate_3m.fillna(0)) +
        zscore_global(cpi_yoy.fillna(0)) +
        zscore_global(urate_3m.fillna(0))
    )


def add_fda_features(feat: pd.DataFrame, fda_df: pd.DataFrame, ticker: str) -> None:
    """
    Group D — FDA-specific features for biotech tickers.
    fda_df must be pre-loaded from data/events/biotech/fda_events.parquet.
    """
    trading_dates = feat.index
    ticker_fda = fda_df[fda_df["ticker"] == ticker].copy()

    if ticker_fda.empty:
        print(f"    WARNING: no FDA events found for {ticker}")
        for col in [
            "days_to_next_fda_decision", "days_since_last_fda_decision",
            "last_fda_outcome", "fda_decisions_trailing_12m",
            "is_pdufa_month", "fda_approval_rate_trailing",
        ]:
            feat[col] = 0
        return

    fda_dates = ticker_fda.index.unique().sort_values()
    print(f"    FDA events: {len(ticker_fda)}  ({fda_dates.min().date()} to {fda_dates.max().date()})")

    # Days to / since FDA decision (calendar days, capped at sentinel)
    raw_to   = days_to_next(trading_dates, fda_dates.values)
    raw_from = days_since_last(trading_dates, fda_dates.values)
    feat["days_to_next_fda_decision"]   = np.where(
        np.isnan(raw_to), CAP_FDA, np.minimum(raw_to, CAP_FDA)
    )
    feat["days_since_last_fda_decision"] = np.where(
        np.isnan(raw_from), CAP_FDA, np.minimum(raw_from, CAP_FDA)
    )

    # last_fda_outcome — forward-fill direction of most recent FDA event
    # direction: +1=approval, -1=rejection/CRL, 0=tentative/neutral
    outcome_s = pd.Series(np.nan, index=trading_dates)
    for dt, row in ticker_fda.sort_index().iterrows():
        if dt in outcome_s.index:
            outcome_s.loc[dt] = row["direction"]
    outcome_s = outcome_s.groupby(level=0).last()
    feat["last_fda_outcome"] = outcome_s.ffill().fillna(0).astype(int)

    # fda_decisions_trailing_12m — rolling count of FDA events in last 365 days
    feat["fda_decisions_trailing_12m"] = trailing_event_count(
        trading_dates, fda_dates.values, FDA_WINDOW_DAYS
    )

    # is_pdufa_month — 1 if any FDA decision falls in this calendar month
    # PDUFA dates are published by FDA months in advance; flagging the full month is valid
    fda_year_months = set(
        zip(pd.DatetimeIndex(fda_dates).year, pd.DatetimeIndex(fda_dates).month)
    )
    feat["is_pdufa_month"] = [
        1 if (d.year, d.month) in fda_year_months else 0
        for d in trading_dates
    ]

    # fda_approval_rate_trailing — rolling approval rate over last 3 years
    # For each trading day: count approvals (direction==1) / count total events
    # in trailing FDA_APPROVAL_WIN days
    approval_dates = ticker_fda[ticker_fda["direction"] == 1].index.unique().sort_values()
    trade_d   = _to_day_ints(trading_dates)
    all_d     = np.sort(_to_day_ints(fda_dates.values))
    appr_d    = np.sort(_to_day_ints(approval_dates.values)) if len(approval_dates) else np.array([], dtype=np.int64)

    approval_rates = np.full(len(trading_dates), np.nan)
    for i, t in enumerate(trade_d):
        lo = t - FDA_APPROVAL_WIN
        total = int(np.sum((all_d >= lo) & (all_d <= t)))
        if total > 0:
            n_appr = int(np.sum((appr_d >= lo) & (appr_d <= t))) if len(appr_d) else 0
            approval_rates[i] = n_appr / total
    feat["fda_approval_rate_trailing"] = approval_rates


def add_credit_spread_features(feat: pd.DataFrame) -> None:
    """
    Group E -- Credit spread features (financials sector only).
    Source: FRED BAMLH0A0HYM2 -- ICE BofA US HY Index OAS, saved by 04_events.py.
    Coverage: 1996-12-31 onwards.  Pre-coverage rows filled with sentinel 0.0.

    Features added:
      credit_spread_level    : daily OAS value (percent, e.g. 3.5 = 350 bps)
      credit_spread_change_1m: 21-trading-day change in OAS
      credit_spread_zscore   : 63-day rolling z-score (0.0 sentinel pre-1996)
    """
    trading_dates = feat.index

    if not CREDIT_SPREADS.exists():
        print(f"    [WARN] credit_spreads.parquet not found -- "
              f"run 04_events.py first (sentinel 0.0 used)")
        feat["credit_spread_level"]     = 0.0
        feat["credit_spread_change_1m"] = 0.0
        feat["credit_spread_zscore"]    = 0.0
        return

    cs_df = pd.read_parquet(CREDIT_SPREADS, engine="pyarrow")
    cs_df.index = pd.to_datetime(cs_df.index).normalize()
    cs_level = cs_df["credit_spread_level"]

    # Forward-fill daily values onto trading dates (handles weekends/holidays)
    cs_aligned = (
        cs_level
        .reindex(cs_level.index.union(trading_dates))
        .sort_index()
        .ffill()
        .reindex(trading_dates)
    )

    # Sentinel: rows before first valid date get 0.0
    feat["credit_spread_level"] = cs_aligned.fillna(0.0).values

    # 21-day change (forward-filled series, so shift is valid)
    cs_change = cs_aligned - cs_aligned.shift(21)
    feat["credit_spread_change_1m"] = cs_change.fillna(0.0).values

    # 63-day rolling z-score; NaN for first 62 rows -> 0.0 sentinel
    roll_mean = cs_aligned.rolling(63, min_periods=63).mean()
    roll_std  = cs_aligned.rolling(63, min_periods=63).std()
    cs_zscore = (cs_aligned - roll_mean) / (roll_std + 1e-9)
    feat["credit_spread_zscore"] = cs_zscore.fillna(0.0).values

    n_valid     = cs_aligned.notna().sum()
    first_valid = cs_aligned.first_valid_index()
    cov_pct     = n_valid / len(trading_dates) * 100
    print(f"    credit spread: {n_valid:,}/{len(trading_dates):,} obs ({cov_pct:.1f}%), "
          f"first valid: {first_valid.date() if first_valid else 'N/A'}")


def add_energy_features(feat: pd.DataFrame) -> None:
    """
    ENERGY_7 -- Energy commodity features (energy sector only).
    Source: FRED DCOILWTICO (WTI), DHHNGSP (natgas),
            saved as wide-format daily DataFrame by 04_events.py.

    Coverage:
      wti_price / changes / zscore : 1986-01-02 onwards
      natgas_price / changes       : 1997-01-07 onwards  (pre-1997 -> sentinel 0.0)

    Features added (ENERGY_7):
      wti_price         : WTI crude spot price (USD per barrel)
      wti_change_1w     : 5-day pct change in WTI (%)
      wti_change_1m     : 21-day pct change in WTI (%)
      wti_zscore_63d    : 63-day rolling z-score of WTI price
      natgas_price      : Henry Hub spot price (USD per MMBtu)
      natgas_change_1w  : 5-day pct change in natgas (%)
      natgas_change_1m  : 21-day pct change in natgas (%)

    NaN sentinel: 0.0 for pre-coverage rows (natgas pre-1997).
    """
    trading_dates = feat.index

    if not ENERGY_EVENTS.exists():
        print(f"    [WARN] energy_events.parquet not found -- "
              f"run 04_events.py first (sentinel 0.0 used)")
        for col in ["wti_price", "wti_change_1w", "wti_change_1m", "wti_zscore_63d",
                    "natgas_price", "natgas_change_1w", "natgas_change_1m"]:
            feat[col] = 0.0
        return

    en_df = pd.read_parquet(ENERGY_EVENTS, engine="pyarrow")
    en_df.index = pd.to_datetime(en_df.index).normalize()

    def _align(series: pd.Series) -> pd.Series:
        """Forward-fill a daily series onto ticker trading dates."""
        return (
            series
            .reindex(series.index.union(trading_dates))
            .sort_index()
            .ffill()
            .reindex(trading_dates)
        )

    # Align all columns from the pre-computed energy_events parquet
    for col in ["wti_price", "wti_change_1w", "wti_change_1m", "wti_zscore_63d",
                "natgas_price", "natgas_change_1w", "natgas_change_1m"]:
        aligned = _align(en_df[col])
        feat[col] = aligned.fillna(0.0).values

    n_valid_wti    = (feat["wti_price"] != 0.0).sum()
    n_valid_natgas = (feat["natgas_price"] != 0.0).sum()
    print(f"    WTI valid rows   : {n_valid_wti:,}/{len(trading_dates):,}")
    print(f"    natgas valid rows: {n_valid_natgas:,}/{len(trading_dates):,} "
          f"(pre-1997 -> 0.0 sentinel)")


def add_consumer_staples_features(feat: pd.DataFrame) -> None:
    """
    STAPLES_4 -- Consumer staples event features (consumer_staples sector only).
    Source: FRED RSXFS (retail sales monthly 1992+) + UMCSENT (sentiment monthly 1978+),
            saved as wide-format daily DataFrame by 04_events.py.

    Coverage:
      retail_sales_* : 1992-01+ (pre-1992 -> sentinel 0.0)
      consumer_sentiment_* : 1978-01+ (full history, no sentinel needed)

    Features added (STAPLES_4):
      retail_sales_mom_change   : month-over-month pct change in retail sales (21-day approx)
      retail_sales_zscore_3m    : 63-day rolling z-score of retail sales level
      consumer_sentiment_level  : University of Michigan sentiment index value
      consumer_sentiment_change_3m : 63-day change in sentiment level

    NaN sentinel: 0.0 for pre-1992 retail_sales rows.
    """
    trading_dates = feat.index

    if not CONSUMER_STAPLES_EVENTS.exists():
        print(f"    [WARN] consumer_staples_events.parquet not found -- "
              f"run 04_events.py first (sentinel 0.0 used)")
        for col in ["retail_sales_mom_change", "retail_sales_zscore_3m",
                    "consumer_sentiment_level", "consumer_sentiment_change_3m"]:
            feat[col] = 0.0
        return

    cs_df = pd.read_parquet(CONSUMER_STAPLES_EVENTS, engine="pyarrow")
    cs_df.index = pd.to_datetime(cs_df.index).normalize()

    def _align(series: pd.Series) -> pd.Series:
        """Forward-fill a daily/monthly series onto ticker trading dates."""
        return (
            series
            .reindex(series.index.union(trading_dates))
            .sort_index()
            .ffill()
            .reindex(trading_dates)
        )

    # Retail sales features (pre-1992 NaN -> 0.0 sentinel)
    for col in ["retail_sales_mom_change", "retail_sales_zscore_3m"]:
        feat[col] = _align(cs_df[col]).fillna(0.0).values

    # Sentiment features (full history back to 1978; sentinel only for earliest rows)
    for col in ["consumer_sentiment_level", "consumer_sentiment_change_3m"]:
        feat[col] = _align(cs_df[col]).fillna(0.0).values

    n_valid_rs   = (feat["retail_sales_mom_change"] != 0.0).sum()
    n_valid_sent = (feat["consumer_sentiment_level"] != 0.0).sum()
    print(f"    retail sales valid rows : {n_valid_rs:,}/{len(trading_dates):,} "
          f"(pre-1992 -> 0.0 sentinel)")
    print(f"    sentiment valid rows    : {n_valid_sent:,}/{len(trading_dates):,}")


# ══════════════════════════════════════════════════════════════════════════════
# Coverage report
# ══════════════════════════════════════════════════════════════════════════════

def coverage_report(combined: pd.DataFrame, new_cols: list, ticker: str) -> None:
    """Print per-feature coverage statistics."""
    print(f"\n  Feature coverage ({ticker}, {len(combined):,} rows):")
    print(f"  {'Feature':<38} {'Coverage':>8}  {'Range'}")
    print(f"  {'-'*70}")
    for col in new_cols:
        s       = combined[col]
        n_ok    = s.notna().sum()
        pct     = n_ok / len(combined) * 100
        rng_str = ""
        if n_ok > 0:
            mn, mx = s.dropna().min(), s.dropna().max()
            rng_str = f"[{mn:.2f}, {mx:.2f}]"
        print(f"  {col:<38} {pct:>7.1f}%  {rng_str}")


# ══════════════════════════════════════════════════════════════════════════════
# Per-ticker processing
# ══════════════════════════════════════════════════════════════════════════════

def process_ticker(
    ticker:     str,
    fred:       dict,
    fda_df:     pd.DataFrame,
    regime_df:  "pd.DataFrame | None" = None,
) -> bool:
    labeled_path = PROCESSED_DIR / f"{ticker}_labeled.parquet"
    out_path     = PROCESSED_DIR / f"{ticker}_with_events.parquet"
    sector       = TICKER_SECTOR[ticker]
    is_biotech   = (sector == "biotech")

    if out_path.exists():
        size_kb = out_path.stat().st_size / 1024
        print(f"  {ticker}: already exists ({size_kb:.0f} KB) -- skipping")
        return True

    if not labeled_path.exists():
        print(f"  {ticker}: labeled file missing -- run 03_labels.py first")
        return False

    base = pd.read_parquet(labeled_path)
    base.index = pd.to_datetime(base.index).normalize()
    base = base.sort_index()
    trading_dates = base.index

    print(f"\n  {ticker} ({sector}) -- {len(base):,} rows  "
          f"({trading_dates.min().date()} to {trading_dates.max().date()})")

    feat = pd.DataFrame(index=trading_dates)

    # A. Earnings
    section("A. Earnings features")
    add_earnings_features(feat, ticker)

    # B. Macro
    section("B. Macro features (FRED)")
    add_macro_features(feat, fred)

    # C. Regime / stress
    section("C. Regime + stress features")
    add_regime_features(feat)

    # D. FDA (biotech only)
    if is_biotech:
        section("D. FDA decision features (biotech)")
        add_fda_features(feat, fda_df, ticker)

    # D2. Credit spread (financials only)
    is_financials = (sector == "financials")
    if is_financials:
        section("D2. Credit spread features (financials)")
        add_credit_spread_features(feat)

    # D3. Energy commodity features (energy only)
    is_energy = (sector == "energy")
    if is_energy:
        section("D3. Energy commodity features (energy)")
        add_energy_features(feat)

    # D4. Consumer staples features (consumer_staples only)
    is_consumer_staples = (sector == "consumer_staples")
    if is_consumer_staples:
        section("D4. Consumer staples features (consumer_staples)")
        add_consumer_staples_features(feat)

    # E. Regime features from Step 11 (VIX, yield spread, sentiment, breadth, HMM)
    if regime_df is not None and len(regime_df) > 0:
        section("E. Regime features (Step 11: VIX, yield spread, sentiment, breadth, HMM)")
        # Forward-fill regime features onto ticker trading dates
        combined_idx = trading_dates.union(regime_df.index)
        regime_aligned = (
            regime_df
            .reindex(combined_idx)
            .sort_index()
            .ffill()
            .reindex(trading_dates)
        )
        for col in regime_df.columns:
            feat[col] = regime_aligned[col].values
        print(f"    Joined {len(regime_df.columns)} regime columns: {list(regime_df.columns)}")
    else:
        print("  [WARN] regime_features.parquet not found -- run 11_regime_features.py first")

    # Join and save
    new_cols = feat.columns.tolist()
    combined = base.join(feat, how="left")

    coverage_report(combined, new_cols, ticker)

    combined.to_parquet(out_path)
    size_kb = out_path.stat().st_size / 1024
    print(f"\n  Saved -> {out_path.relative_to(ROOT)}  "
          f"({combined.shape[0]:,} rows x {combined.shape[1]} cols, {size_kb:.0f} KB)")
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

    print(f"\nmarket_ml -- event feature engineering")
    print(f"Tickers  : {[t for _, t in all_tickers]}")
    print(f"Output   : {PROCESSED_DIR.relative_to(ROOT)}/\n")

    # Pre-fetch FRED once (shared across all tickers)
    fred = prefetch_fred()

    # Load FDA events once
    print(f"\n  Loading FDA events ...", end=" ")
    fda_df = pd.read_parquet(FDA_EVENTS) if FDA_EVENTS.exists() else pd.DataFrame()
    fda_df.index = pd.to_datetime(fda_df.index).normalize()
    print(f"{len(fda_df)} events loaded")

    # Load regime features (from Step 11) if available
    regime_path = PROCESSED_DIR / "regime_features.parquet"
    if regime_path.exists():
        regime_df = pd.read_parquet(regime_path)
        regime_df.index = pd.to_datetime(regime_df.index).normalize()
        print(f"\n  Loading regime features ... {len(regime_df):,} rows x "
              f"{regime_df.shape[1]} cols  "
              f"({regime_df.index[0].date()} to {regime_df.index[-1].date()})")
    else:
        regime_df = None
        print("\n  [WARN] regime_features.parquet not found -- run 11_regime_features.py first")

    results = {"ok": [], "skip": [], "err": []}

    for sector, ticker in all_tickers:
        out_path = PROCESSED_DIR / f"{ticker}_with_events.parquet"
        print(f"\n{'=' * 60}")
        print(f"[{sector.upper()}] {ticker}")
        print(f"{'=' * 60}")

        if out_path.exists():
            size_kb = out_path.stat().st_size / 1024
            print(f"  Already exists ({size_kb:.0f} KB) -- skipping")
            results["skip"].append(ticker)
            continue

        try:
            success = process_ticker(ticker, fred, fda_df, regime_df)
            results["ok"].append(ticker) if success else results["err"].append((ticker, "failed"))
        except Exception as exc:
            print(f"  ERROR: {exc}")
            import traceback; traceback.print_exc()
            results["err"].append((ticker, str(exc)))

    print(f"\n{'=' * 60}")
    print(f"DONE.  Computed: {len(results['ok'])}  "
          f"Skipped: {len(results['skip'])}  "
          f"Errors: {len(results['err'])}")
    if results["err"]:
        print("Failed tickers:")
        for t, msg in results["err"]:
            print(f"  {t}: {msg}")
    print(f"{'=' * 60}")

    # Cross-ticker summary
    print(f"\nNew event columns added:")
    sample_ticker = results["ok"][0] if results["ok"] else (results["skip"][0] if results["skip"] else None)
    if sample_ticker:
        sp = pd.read_parquet(PROCESSED_DIR / f"{sample_ticker}_with_events.parquet")
        base_cols = set(pd.read_parquet(PROCESSED_DIR / f"{sample_ticker}_labeled.parquet").columns)
        new_cols  = [c for c in sp.columns if c not in base_cols]
        for c in new_cols:
            print(f"  {c}")

    print(f"\nNext step: src/pipeline/06_train.py\n")


def run_single(ticker: str) -> None:
    ticker = ticker.upper()
    if ticker not in TICKER_SECTOR:
        print(f"ERROR: '{ticker}' not in config/tickers.py")
        sys.exit(1)

    print(f"\nEvent features: {ticker} ({TICKER_SECTOR[ticker]})\n")
    fred   = prefetch_fred()
    fda_df = pd.read_parquet(FDA_EVENTS) if FDA_EVENTS.exists() else pd.DataFrame()
    fda_df.index = pd.to_datetime(fda_df.index).normalize()
    regime_path = PROCESSED_DIR / "regime_features.parquet"
    regime_df = None
    if regime_path.exists():
        regime_df = pd.read_parquet(regime_path)
        regime_df.index = pd.to_datetime(regime_df.index).normalize()
    success = process_ticker(ticker, fred, fda_df, regime_df)
    if not success:
        sys.exit(1)
    print("\nDone.\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_single(sys.argv[1])
    else:
        run_all()

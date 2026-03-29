"""
Step 11 -- Regime Feature Engineering
======================================
Builds a daily regime feature matrix and fits a 3-state Gaussian HMM.

Saved to data/processed/regime_features.parquet (9 columns):

  vix_close              VIX level (yfinance ^VIX, 1990+)
  vix_change_1w          5-day change in VIX  (sentinel 0.0 for first 5 rows)
  vix_zscore_63d         63-day rolling z-score of VIX  (sentinel 0.0)
  yield_spread           T10Y2Y yield spread (FRED daily, 1976+, sentinel 0.0 for NaN gaps)
  yield_spread_change_1m 21-day change in yield spread  (sentinel 0.0)
  sentiment_zscore       Univ. of Michigan Consumer Sentiment UMCSENT z-score
                         vs 12-month rolling mean (FRED monthly, 1952+)
                         NOTE: AAII Sentiment not available on FRED public API;
                         UMCSENT is used as a sentiment proxy.
  put_call_ratio         1.0 sentinel for all dates.
                         NOTE: CBOE put/call ratio not available from any free
                         public API (not on FRED, not on yfinance). Constant
                         value -- XGBoost will learn to ignore it.
  breadth_pct_above_200d % of 11 tickers with close > 200d SMA (from _features.parquet)
                         (sentinel 0.5 for dates before sufficient price history)
  hmm_regime             3-state Gaussian HMM output: 0=range-bound, 1=trending,
                         2=volatile (fit on vix_close + yield_spread + breadth)

HMM model saved to models/hmm_regime_detector.pkl

Usage:
    C:\\Users\\borra\\anaconda3\\python.exe src\\pipeline\\11_regime_features.py
"""

import sys
import pickle
import warnings
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import io
import numpy as np
import pandas as pd
import requests
import yfinance as yf

try:
    from hmmlearn import hmm as hmmlearn_hmm
except ImportError:
    import subprocess
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "hmmlearn", "--break-system-packages"
    ])
    from hmmlearn import hmm as hmmlearn_hmm

from config.tickers import SECTORS, TICKER_SECTOR

ALL_TICKERS = [t for cfg in SECTORS.values() for t in cfg["tickers"]]

# ── Paths ──────────────────────────────────────────────────────────────────────
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR    = ROOT / "models"
OUTPUT_PATH   = PROCESSED_DIR / "regime_features.parquet"
HMM_PATH      = MODELS_DIR / "hmm_regime_detector.pkl"

# ── Constants ──────────────────────────────────────────────────────────────────
FRED_BASE      = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
VIX_START      = "1990-01-01"
FRED_T10Y2Y_START = "1976-01-01"
FRED_UMCSENT_START = "1952-01-01"

# Sentinel values for partial-history features
SENTINEL_YIELD    = 0.0   # neutral spread when T10Y2Y unavailable
SENTINEL_SENTIMENT = 0.0  # neutral z-score when pre-UMCSENT history
SENTINEL_PUTCALL  = 1.0   # neutral P/C ratio (puts = calls)
SENTINEL_BREADTH  = 0.5   # half above, half below 200d SMA
SENTINEL_HMM      = 0     # range-bound as default state
SENTINEL_VIX_DERIVED = 0.0  # z-score / change when insufficient history


# ── Section printer ────────────────────────────────────────────────────────────
def section(title: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")


# ── FRED fetch ─────────────────────────────────────────────────────────────────
def fetch_fred(series_id: str, start: str = "1976-01-01") -> pd.Series:
    url = FRED_BASE.format(series_id)
    r   = requests.get(url, timeout=20)
    r.raise_for_status()
    if r.status_code != 200 or r.text.startswith("<!"):
        raise ValueError(f"FRED returned HTML for {series_id} -- series not public")
    df = pd.read_csv(
        io.StringIO(r.text),
        parse_dates=["observation_date"],
        index_col="observation_date",
    )
    df.columns = [series_id]
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
    s = df[series_id].dropna().sort_index()
    return s[s.index >= start]


# ── 1. VIX ─────────────────────────────────────────────────────────────────────
def fetch_vix() -> pd.DataFrame:
    """Fetch VIX daily close and compute derived features."""
    print("  Fetching ^VIX via yfinance ...", end=" ", flush=True)
    raw = yf.download("^VIX", start=VIX_START, progress=False, auto_adjust=True)
    if raw.empty:
        raise RuntimeError("^VIX download returned empty DataFrame")

    close = raw["Close"].squeeze().sort_index()
    close.name = "vix_close"
    close.index = pd.to_datetime(close.index).normalize()
    print(f"OK  ({len(close)} rows, {close.index[0].date()} to {close.index[-1].date()})")

    df = pd.DataFrame(index=close.index)
    df["vix_close"]     = close
    df["vix_change_1w"] = close.diff(5).fillna(SENTINEL_VIX_DERIVED)

    # 63-day rolling z-score
    roll_mean = close.rolling(63, min_periods=63).mean()
    roll_std  = close.rolling(63, min_periods=63).std()
    zscore    = (close - roll_mean) / (roll_std + 1e-9)
    df["vix_zscore_63d"] = zscore.fillna(SENTINEL_VIX_DERIVED)

    return df


# ── 2. Treasury yield spread ───────────────────────────────────────────────────
def fetch_yield_spread() -> pd.DataFrame:
    """Fetch T10Y2Y (10Y - 2Y treasury spread) from FRED."""
    print("  Fetching T10Y2Y from FRED ...", end=" ", flush=True)
    s = fetch_fred("T10Y2Y", start=FRED_T10Y2Y_START)
    print(f"OK  ({len(s)} obs, {s.index[0].date()} to {s.index[-1].date()})")

    df = pd.DataFrame(index=s.index)
    df["yield_spread"] = s.values
    # 21-day change (FRED T10Y2Y is daily so diff(21) = ~1 month)
    df["yield_spread_change_1m"] = s.diff(21)
    return df


# ── 3. Sentiment (UMCSENT proxy for AAII) ─────────────────────────────────────
def fetch_sentiment() -> pd.DataFrame:
    """
    Fetch University of Michigan Consumer Sentiment (UMCSENT) from FRED.
    AAII Bull/Bear Sentiment is not available on the FRED public CSV endpoint.
    UMCSENT (monthly, 1952+) is used as a sentiment proxy.
    Computes z-score vs 12-month rolling mean.
    """
    print("  Fetching UMCSENT (sentiment proxy) from FRED ...", end=" ", flush=True)
    s = fetch_fred("UMCSENT", start=FRED_UMCSENT_START)
    print(f"OK  ({len(s)} obs, {s.index[0].date()} to {s.index[-1].date()})")

    # Z-score vs rolling 12-month mean
    roll_mean = s.rolling(12, min_periods=12).mean()
    roll_std  = s.rolling(12, min_periods=12).std()
    zscore    = (s - roll_mean) / (roll_std + 1e-9)
    zscore.name = "sentiment_zscore"
    df = pd.DataFrame({"sentiment_zscore": zscore})
    return df


# ── 4. Put/call ratio (sentinel) ───────────────────────────────────────────────
def build_put_call_sentinel(index: pd.DatetimeIndex) -> pd.Series:
    """
    CBOE put/call ratio is not available from any free public API:
      - Not on FRED (404 for CPCE/CPCETM)
      - Not on yfinance (^PCALL returns 404)
    Returns 1.0 (neutral) for all dates. XGBoost will learn to ignore this constant.
    """
    print("  [WARN] put_call_ratio: no free API available -- using 1.0 sentinel for all dates")
    return pd.Series(SENTINEL_PUTCALL, index=index, name="put_call_ratio")


# ── 5. Market breadth ──────────────────────────────────────────────────────────
def compute_market_breadth() -> pd.Series:
    """
    Compute % of 11 tickers with close above their 200-day SMA.
    Uses close_vs_sma200 from {TICKER}_features.parquet.
    close_vs_sma200 > 0 means close > SMA200.
    Pre-data sentinel: 0.5 (half above, half below).
    """
    print("  Computing breadth from features parquets ...", end=" ", flush=True)
    frames = []
    for ticker in ALL_TICKERS:
        path = PROCESSED_DIR / f"{ticker}_features.parquet"
        if not path.exists():
            print(f"\n  [WARN] {ticker}_features.parquet not found -- run 02_features.py first")
            continue
        df = pd.read_parquet(path, columns=["close_vs_sma200"])
        df.columns = [ticker]
        frames.append(df)

    if not frames:
        raise RuntimeError("No features parquets found -- run 02_features.py first")

    combined = pd.concat(frames, axis=1)
    # fraction of tickers above their 200d SMA on each trading day
    above    = (combined > 0).sum(axis=1)
    total    = combined.notna().sum(axis=1)
    breadth  = (above / total.replace(0, np.nan)).fillna(SENTINEL_BREADTH)
    breadth.name = "breadth_pct_above_200d"
    print(f"OK  ({len(breadth)} rows, {breadth.index[0].date()} to {breadth.index[-1].date()})")
    return breadth.sort_index()


# ── 6. Build daily aligned frame ───────────────────────────────────────────────
def build_daily_frame(
    vix_df:    pd.DataFrame,
    yield_df:  pd.DataFrame,
    sent_df:   pd.DataFrame,
    breadth:   pd.Series,
) -> pd.DataFrame:
    """
    Align all features onto the VIX daily trading-day index (1990+).
    Monthly/weekly series are forward-filled to daily.
    """
    idx = vix_df.index   # VIX trading days = US market days from 1990

    out = pd.DataFrame(index=idx)

    # VIX (already daily, no fill needed)
    for col in vix_df.columns:
        out[col] = vix_df[col].reindex(idx)

    # Yield spread (daily FRED; reindex then ffill gaps around holidays)
    yield_aligned = yield_df.reindex(idx.union(yield_df.index)).sort_index().ffill()
    for col in yield_df.columns:
        out[col] = yield_aligned[col].reindex(idx).fillna(SENTINEL_YIELD)

    # Sentiment (monthly FRED -> forward-fill to daily, then sentinel pre-data)
    sent_aligned = sent_df.reindex(idx.union(sent_df.index)).sort_index().ffill()
    out["sentiment_zscore"] = sent_aligned["sentiment_zscore"].reindex(idx).fillna(SENTINEL_SENTIMENT)

    # Market breadth (daily, from features parquets; ffill and sentinel)
    breadth_aligned = breadth.reindex(idx.union(breadth.index)).sort_index().ffill()
    out["breadth_pct_above_200d"] = breadth_aligned.reindex(idx).fillna(SENTINEL_BREADTH)

    # Put/call ratio (constant sentinel)
    out["put_call_ratio"] = SENTINEL_PUTCALL

    return out


# ── 7. HMM regime detector ─────────────────────────────────────────────────────
def fit_hmm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit a 3-state Gaussian HMM on {vix_close, yield_spread, breadth_pct_above_200d}.
    States are relabelled by ascending mean VIX:
      0 = range-bound  (low VIX)
      1 = trending     (medium VIX)
      2 = volatile     (high VIX)
    Saves the fitted model to models/hmm_regime_detector.pkl.
    """
    print("  Fitting 3-state Gaussian HMM ...", end=" ", flush=True)

    input_cols = ["vix_close", "yield_spread", "breadth_pct_above_200d"]
    X_full = df[input_cols].copy()

    # Use rows where ALL three inputs are non-sentinel and non-NaN
    # vix_close is always available from 1990; yield_spread pre-1976 gets 0.0 sentinel;
    # breadth pre-~1996 gets 0.5 sentinel.  All rows have values, but early rows use sentinels.
    # For HMM fitting use rows from 1996-01-01 onwards (breadth has real data by then).
    fit_mask = X_full.index >= "1996-01-01"
    X_fit = X_full[fit_mask].values

    # Standardize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_fit)

    model = hmmlearn_hmm.GaussianHMM(
        n_components=3,
        covariance_type="full",
        n_iter=200,
        random_state=42,
        verbose=False,
    )
    model.fit(X_scaled)
    print(f"converged={model.monitor_.converged}  log-likelihood={model.monitor_.history[-1]:.1f}")

    # Predict states for full history (standardize ALL rows using scaler fit on post-1996)
    X_all_scaled = scaler.transform(X_full.values)
    raw_states = model.predict(X_all_scaled)

    # Relabel states by mean VIX per state (ascending = 0:range-bound, 1:trending, 2:volatile)
    mean_vix_per_state = {
        s: float(df["vix_close"].values[raw_states == s].mean())
        for s in range(3)
    }
    sorted_states  = sorted(mean_vix_per_state, key=mean_vix_per_state.get)
    state_remap    = {old: new for new, old in enumerate(sorted_states)}
    relabelled     = np.vectorize(state_remap.get)(raw_states)

    df = df.copy()
    df["hmm_regime"] = relabelled.astype(int)

    # Print state statistics
    print(f"  State distribution (0=range-bound, 1=trending, 2=volatile):")
    new_mean_vix = {}
    for new_s in range(3):
        mask = relabelled == new_s
        label = {0: "range-bound", 1: "trending", 2: "volatile"}[new_s]
        mvix  = float(df["vix_close"].values[mask].mean())
        msprd = float(df["yield_spread"].values[mask].mean())
        mbrd  = float(df["breadth_pct_above_200d"].values[mask].mean())
        pct   = mask.sum() / len(mask) * 100
        new_mean_vix[new_s] = mvix
        print(f"    State {new_s} ({label:12s}): {pct:4.1f}%  "
              f"mean_VIX={mvix:.1f}  mean_spread={msprd:.2f}  mean_breadth={mbrd:.2f}")

    # Save model + scaler + remapping
    HMM_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HMM_PATH, "wb") as f:
        pickle.dump({
            "model":       model,
            "scaler":      scaler,
            "state_remap": state_remap,
            "input_cols":  input_cols,
        }, f)
    size_kb = HMM_PATH.stat().st_size / 1024
    print(f"  [OK]  HMM model saved -> {HMM_PATH.relative_to(ROOT)}  ({size_kb:.0f} KB)")

    return df


# ── 8. Coverage report ─────────────────────────────────────────────────────────
def print_coverage(df: pd.DataFrame, start_year: int = 1995) -> None:
    """Print coverage % and sentinel notes for each regime feature."""
    subset = df[df.index >= f"{start_year}-01-01"]
    n = len(subset)
    print(f"\n  Coverage report (rows from {start_year}-01-01 to present, n={n:,}):")
    print(f"  {'Column':<28} {'Coverage':>8}  {'Range':<24}  Notes")
    print("  " + "-" * 80)

    notes = {
        "vix_close":              "yfinance ^VIX, 1990+",
        "vix_change_1w":          f"5-day diff; sentinel {SENTINEL_VIX_DERIVED} for first 5 rows",
        "vix_zscore_63d":         f"63d rolling z-score; sentinel {SENTINEL_VIX_DERIVED}",
        "yield_spread":           f"FRED T10Y2Y; sentinel {SENTINEL_YIELD} for NaN gaps",
        "yield_spread_change_1m": f"21-day diff; sentinel {SENTINEL_YIELD}",
        "sentiment_zscore":       f"UMCSENT z-score (AAII unavailable); sentinel {SENTINEL_SENTIMENT}",
        "put_call_ratio":         f"No free API -- constant sentinel {SENTINEL_PUTCALL}",
        "breadth_pct_above_200d": f"% above 200d SMA; sentinel {SENTINEL_BREADTH}",
        "hmm_regime":             f"3-state HMM; sentinel {SENTINEL_HMM} for pre-1996",
    }
    for col in df.columns:
        s      = subset[col]
        n_ok   = s.notna().sum()
        pct    = n_ok / n * 100
        rng    = ""
        if n_ok > 0:
            mn, mx = s.dropna().min(), s.dropna().max()
            rng = f"[{mn:.2f}, {mx:.2f}]"
        note = notes.get(col, "")
        print(f"  {col:<28} {pct:>7.1f}%  {rng:<24}  {note}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    section("market_ml -- Step 11: Regime Feature Engineering")
    print(f"  Output : {OUTPUT_PATH.relative_to(ROOT)}")
    print(f"  HMM    : {HMM_PATH.relative_to(ROOT)}")

    if OUTPUT_PATH.exists():
        print(f"\n  [SKIP] {OUTPUT_PATH.name} already exists. Delete to re-run.")
        sys.exit(0)

    # 1. VIX
    section("1. VIX Features")
    vix_df = fetch_vix()

    # 2. Yield spread
    section("2. Treasury Yield Spread (T10Y2Y)")
    yield_df = fetch_yield_spread()

    # 3. Sentiment
    section("3. Sentiment Proxy (UMCSENT)")
    print("  NOTE: AAII Bull/Bear Sentiment not available on FRED public CSV endpoint.")
    print("        Using University of Michigan Consumer Sentiment (UMCSENT) as proxy.")
    sent_df = fetch_sentiment()

    # 4. Market breadth
    section("4. Market Breadth (% above 200d SMA)")
    breadth = compute_market_breadth()

    # 5. Align all to daily VIX index
    section("5. Aligning All Features to Daily Trading Index")
    print(f"  Using VIX trading dates as index: {len(vix_df)} rows")
    df = build_daily_frame(vix_df, yield_df, sent_df, breadth)
    print(f"  [OK]  Combined frame: {df.shape[0]:,} rows x {df.shape[1]} cols")

    # 6. HMM
    section("6. HMM Regime Detector")
    df = fit_hmm(df)

    # 7. Coverage report
    section("7. Coverage Report")
    print_coverage(df, start_year=1995)

    # 8. Save
    section("8. Saving")
    df.index.name = "date"
    df.to_parquet(OUTPUT_PATH, engine="pyarrow", index=True)
    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"  [OK]  {OUTPUT_PATH.relative_to(ROOT)}  "
          f"({len(df):,} rows x {df.shape[1]} cols, {size_kb:.0f} KB)")
    print(f"  Columns: {list(df.columns)}")

    section("Step 11 Complete")
    print("  Next step: re-run 05_event_features.py to join regime features into")
    print("  each ticker's feature matrix, then re-run 06_train.py.\n")


if __name__ == "__main__":
    main()

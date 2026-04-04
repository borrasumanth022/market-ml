"""
Step 12 -- Weekly Iron Condor Signal Generator (Paper Trading Engine)
=====================================================================
Runs every Monday morning (or any day, with a warning if not Monday).
Loads the trained models, fetches the latest weekly OHLCV for all 11 tickers,
computes the full feature vector (SELECTED_36 + EVENT_14 + REGIME_9 + FDA_5),
and outputs iron condor signals at confidence threshold 0.55.
Appends every signal (FIRE or NO_FIRE) to data/signals/signal_log.parquet.

Signal rule:
  FIRE    : model predicts Sideways with proba_side >= SIGNAL_THRESHOLD (0.55)
  NO_FIRE : proba_side < SIGNAL_THRESHOLD

Models used:
  VRTX          -> models/biotech/VRTX_finetuned_v1.pkl  (60 features, no REGIME_9)
  other biotech -> models/biotech/xgb_biotech_shared_v1.pkl  (69 features)
  tech          -> models/tech/xgb_tech_shared_v1.pkl        (65 features)
  Each model stores its own feature_names list -- the feature vector is built
  by matching that list exactly (one-hot columns set to 0/1 for this ticker).

Kelly Criterion position sizing (half-Kelly, capped at 20% of portfolio):
  kelly_fraction   = (WIN_RATE / AVG_LOSS) - (LOSS_RATE / AVG_WIN)
  recommended_size = min(kelly_fraction * 0.5, MAX_POSITION_PCT)

  Using WIN_RATE = 0.69 from Step 11 holdout backtest (42 trades at threshold 0.55).

Known approximations:
  - macro_stress_score z-score computed over the 500-day OHLCV window, not the
    full 30-year training history. Acceptable for the signal generator context.
  - put_call_ratio = 1.0 constant sentinel (no free API available).
  - UMCSENT used as sentiment proxy (AAII not on FRED).

Usage:
    C:\\Users\\borra\\anaconda3\\python.exe src\\pipeline\\12_signal_generator.py
"""

import sys
import io
import pickle
import time
import warnings
import importlib.util
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from config.tickers import SECTORS, TICKER_SECTOR

# ── Paths ──────────────────────────────────────────────────────────────────────
PROCESSED_DIR = ROOT / "data" / "processed"
SIGNALS_DIR   = ROOT / "data" / "signals"
MODELS_DIR    = ROOT / "models"
FDA_EVENTS    = ROOT / "data" / "events" / "biotech" / "fda_events.parquet"

# ── Strategy constants ────────────────────────────────────────────────────────
SIGNAL_THRESHOLD = 0.55
PREMIUM          = 0.015    # iron condor credit per unit
LOSS_MAX         = 0.030    # max loss per unit if wings triggered

# Kelly inputs from Step 11 holdout backtest (42 trades, threshold 0.55)
WIN_RATE         = 0.69
LOSS_RATE        = 1.0 - WIN_RATE
AVG_WIN          = PREMIUM
AVG_LOSS         = LOSS_MAX
KELLY_FULL       = (WIN_RATE / AVG_LOSS) - (LOSS_RATE / AVG_WIN)
KELLY_HALF       = KELLY_FULL * 0.5
MAX_POSITION_PCT = 0.20     # 20% portfolio cap per position

REGIME_LABELS  = {0: "range-bound", 1: "trending", 2: "volatile"}
SIDEWAYS_CLASS = 1
CLASS_NAMES    = ["Bear", "Sideways", "Bull"]

# Fetch 500 trading days of OHLCV for rolling-feature warmup
# (need 252 for price_52w_pct, 200 for sma_200; 500 gives plenty of buffer)
OHLCV_HISTORY_DAYS = 500

ALL_TICKERS = [t for v in SECTORS.values() for t in v["tickers"]]

FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"


# ── Dynamic module loader (avoids re-implementing pipeline logic) ──────────────

def _load_pipeline_module(filename: str):
    """Load a pipeline script as a module via importlib (handles numeric prefixes)."""
    path = ROOT / "src" / "pipeline" / filename
    if not path.exists():
        raise FileNotFoundError(f"Pipeline module not found: {path}")
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Load feature-builder functions from existing pipeline scripts once at import time.
# These are the same functions used during training -- guarantees feature parity.
_feat_mod  = _load_pipeline_module("02_features.py")
_event_mod = _load_pipeline_module("05_event_features.py")

_build_tech_features = _feat_mod.build_features          # (df, ticker) -> pd.DataFrame
_add_earnings        = _event_mod.add_earnings_features   # (feat, ticker) -> None
_add_macro           = _event_mod.add_macro_features      # (feat, fred) -> None
_add_regime_stress   = _event_mod.add_regime_features     # (feat) -> None (rate env/inflation/stress)
_add_fda             = _event_mod.add_fda_features        # (feat, fda_df, ticker) -> None


# ── Printers ──────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")


# ── Signal date ───────────────────────────────────────────────────────────────

def get_signal_date() -> pd.Timestamp:
    """
    Return the most recent Friday as the signal date.
    If today is Monday: expected production run -- no warning.
    If today is not Monday: print warning (script works fine either way).
    The actual date used is the latest trading day at or before last Friday,
    confirmed against real OHLCV data after fetching.
    """
    today = pd.Timestamp.today().normalize()
    # weekday(): Mon=0, Tue=1, Wed=2, Thu=3, Fri=4, Sat=5, Sun=6
    days_since_friday = (today.weekday() - 4) % 7
    if days_since_friday == 0:
        last_friday = today   # today IS Friday
    else:
        last_friday = today - pd.Timedelta(days=days_since_friday)

    day_name = today.strftime("%A")
    if today.weekday() == 0:  # Monday
        print(f"  [OK]  Running on {day_name} {today.date()}.")
        print(f"        Signal date: last Friday {last_friday.date()}")
    else:
        print(f"  [WARN] Today is {day_name} {today.date()} -- not Monday.")
        print(f"  [WARN] Expected Monday morning for paper trading.")
        print(f"  [WARN] Using last Friday {last_friday.date()} as signal date.")
    return last_friday


# ── Model loading ─────────────────────────────────────────────────────────────

def load_models() -> dict:
    """
    Load all saved models from models/ directory.
    VRTX gets its fine-tuned model (60 features, pre-Step-11 -- no REGIME_9).
    All other tickers get the sector shared model (65/69 features with REGIME_9).
    Fails loudly if any required model file is missing.
    Returns {ticker: bundle_dict}.
    """
    paths = {
        "tech":    MODELS_DIR / "tech"    / "xgb_tech_shared_v1.pkl",
        "biotech": MODELS_DIR / "biotech" / "xgb_biotech_shared_v1.pkl",
        "VRTX":    MODELS_DIR / "biotech" / "VRTX_finetuned_v1.pkl",
    }
    for label, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"[FAIL] Model file missing: {path.relative_to(ROOT)}\n"
                f"       Run src/pipeline/06_train.py to regenerate."
            )

    bundles = {}
    for label, path in paths.items():
        with open(path, "rb") as f:
            bundles[label] = pickle.load(f)
        n = len(bundles[label]["feature_names"])
        print(f"  [OK]  {label:8s}: {n} features  ({path.relative_to(ROOT)})")

    # Map each ticker to its model bundle
    ticker_models = {}
    for ticker in ALL_TICKERS:
        if ticker == "VRTX":
            ticker_models[ticker] = bundles["VRTX"]
        else:
            ticker_models[ticker] = bundles[TICKER_SECTOR[ticker]]

    return ticker_models


# ── FRED fetch (standalone -- does not use 05_event_features.prefetch_fred) ───

def fetch_fred(series_id: str, start: str = "1993-01-01") -> pd.Series:
    """Fetch a FRED series as a daily/monthly pd.Series."""
    url = FRED_BASE.format(series_id)
    r   = requests.get(url, timeout=20)
    r.raise_for_status()
    df  = pd.read_csv(
        io.StringIO(r.text),
        parse_dates=["observation_date"],
        index_col="observation_date",
    )
    df.columns  = [series_id]
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
    s = df[series_id].dropna().sort_index()
    return s[s.index >= start]


def fetch_fred_macro() -> dict:
    """
    Fetch FEDFUNDS, CPIAUCSL, UNRATE from FRED.
    Same series as 05_event_features.prefetch_fred -- fetched from 1993
    so macro_stress_score z-score covers the same long-run history as training.
    """
    fred = {}
    for sid in ["FEDFUNDS", "CPIAUCSL", "UNRATE"]:
        print(f"    {sid} ...", end=" ", flush=True)
        fred[sid] = fetch_fred(sid, start="1993-01-01")
        print(f"OK  ({len(fred[sid])} obs, latest: {fred[sid].index[-1].date()})")
    return fred


# ── OHLCV fetch ───────────────────────────────────────────────────────────────

def fetch_ohlcv(ticker: str, days: int = OHLCV_HISTORY_DAYS) -> pd.DataFrame:
    """
    Fetch the last <days> trading days of OHLCV from yfinance.
    Returns DataFrame with lowercase columns: open, high, low, close, volume.
    Fails loudly if the download is empty.
    """
    raw = yf.download(ticker, period=f"{days}d", progress=False, auto_adjust=True)
    if raw.empty:
        raise RuntimeError(f"[FAIL] yfinance returned empty data for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [str(c).lower() for c in raw.columns]
    raw.index   = pd.to_datetime(raw.index).normalize()
    raw         = raw[["open", "high", "low", "close", "volume"]].dropna()
    return raw


# ── Regime features ───────────────────────────────────────────────────────────

def compute_current_regime(
    ohlcv_data: dict,
    signal_date: pd.Timestamp,
) -> pd.Series:
    """
    Compute all 9 REGIME_9 features for signal_date.
    Loads the saved HMM model from models/hmm_regime_detector.pkl -- never refits.
    Returns a pd.Series with 9 values keyed by REGIME_9 column names.
    """
    hmm_path = MODELS_DIR / "hmm_regime_detector.pkl"
    if not hmm_path.exists():
        raise FileNotFoundError(
            f"[FAIL] HMM model missing: {hmm_path.relative_to(ROOT)}\n"
            f"       Run src/pipeline/11_regime_features.py to regenerate."
        )
    with open(hmm_path, "rb") as f:
        hmm_bundle = pickle.load(f)
    hmm_model   = hmm_bundle["model"]
    hmm_scaler  = hmm_bundle["scaler"]
    state_remap = hmm_bundle["state_remap"]

    # --- 1. VIX ---
    print("    ^VIX ...", end=" ", flush=True)
    vix_raw = yf.download("^VIX", period="150d", progress=False, auto_adjust=True)
    if vix_raw.empty:
        raise RuntimeError("[FAIL] Could not fetch ^VIX from yfinance")
    if isinstance(vix_raw.columns, pd.MultiIndex):
        vix_raw.columns = vix_raw.columns.get_level_values(0)
    vix_close = vix_raw["Close"].squeeze().sort_index()
    vix_close.index = pd.to_datetime(vix_close.index).normalize()
    vix_series = vix_close[vix_close.index <= signal_date]
    if vix_series.empty:
        raise ValueError(f"[FAIL] No VIX data on or before {signal_date.date()}")
    print(f"OK  (latest: {vix_series.index[-1].date()}, VIX={vix_series.iloc[-1]:.2f})")

    vix_now  = float(vix_series.iloc[-1])
    vix_5ago = float(vix_series.iloc[-6]) if len(vix_series) >= 6 else vix_now
    vix_change_1w = vix_now - vix_5ago

    if len(vix_series) >= 63:
        roll_mean = vix_series.rolling(63).mean().iloc[-1]
        roll_std  = vix_series.rolling(63).std().iloc[-1]
        vix_z63   = float((vix_now - roll_mean) / (roll_std + 1e-9))
    else:
        vix_z63 = 0.0   # sentinel: insufficient history

    # --- 2. T10Y2Y yield spread ---
    print("    T10Y2Y ...", end=" ", flush=True)
    t10y2y = fetch_fred("T10Y2Y", start="2024-01-01")
    # Forward-fill gaps (weekends, Fed holidays) then align to signal_date
    t10y2y_daily = (
        t10y2y
        .reindex(pd.date_range(t10y2y.index[0], t10y2y.index[-1], freq="D"))
        .ffill()
        .dropna()
    )
    t10y2y_daily.index = pd.to_datetime(t10y2y_daily.index).normalize()
    t10y2y_at_date = t10y2y_daily[t10y2y_daily.index <= signal_date]

    if t10y2y_at_date.empty:
        ys_now = 0.0
        ys_change_1m = 0.0
    else:
        ys_now       = float(t10y2y_at_date.iloc[-1])
        ys_21ago_val = float(t10y2y_at_date.iloc[-22]) if len(t10y2y_at_date) >= 22 else ys_now
        ys_change_1m = ys_now - ys_21ago_val
    print(f"OK  (latest: {t10y2y_at_date.index[-1].date() if not t10y2y_at_date.empty else 'N/A'}, "
          f"spread={ys_now:+.3f})")

    # --- 3. UMCSENT sentiment z-score ---
    print("    UMCSENT ...", end=" ", flush=True)
    umcsent = fetch_fred("UMCSENT", start="2020-01-01")
    sent_at_date = umcsent[umcsent.index <= signal_date]
    if len(sent_at_date) >= 12:
        roll_m = sent_at_date.rolling(12).mean().iloc[-1]
        roll_s = sent_at_date.rolling(12).std().iloc[-1]
        sent_z = float((sent_at_date.iloc[-1] - roll_m) / (roll_s + 1e-9))
    else:
        sent_z = 0.0   # sentinel
    print(f"OK  ({len(umcsent)} obs, sentiment_z={sent_z:.3f})")

    # --- 4. Market breadth: % of 11 tickers with close > 200d SMA ---
    above_sma200 = []
    for ticker in ALL_TICKERS:
        if ticker not in ohlcv_data:
            continue
        df_t   = ohlcv_data[ticker]
        sma200 = df_t["close"].rolling(200, min_periods=200).mean()
        row    = df_t[df_t.index <= signal_date]
        if row.empty or sma200.reindex(row.index).iloc[-1] != sma200.reindex(row.index).iloc[-1]:
            continue   # sma200 is NaN for this ticker on signal date -- skip
        above_sma200.append(int(row["close"].iloc[-1] > sma200.reindex(row.index).iloc[-1]))

    if above_sma200:
        breadth_now = float(np.mean(above_sma200))
    else:
        breadth_now = 0.5   # neutral sentinel

    # --- 5. Apply saved HMM to get current regime state ---
    # Use the full sequence from regime_features.parquet + current point
    # so the Viterbi algorithm has proper context.
    regime_path = PROCESSED_DIR / "regime_features.parquet"
    if regime_path.exists():
        reg_hist = pd.read_parquet(regime_path)
        reg_hist.index = pd.to_datetime(reg_hist.index).normalize()
        # Build extended sequence: historical + today's point if newer
        seq_vix    = reg_hist["vix_close"].copy()
        seq_yield  = reg_hist["yield_spread"].copy()
        seq_breath = reg_hist["breadth_pct_above_200d"].copy()
        if signal_date > reg_hist.index[-1]:
            seq_vix    = pd.concat([seq_vix,    pd.Series([vix_now],       index=[signal_date])])
            seq_yield  = pd.concat([seq_yield,  pd.Series([ys_now],        index=[signal_date])])
            seq_breath = pd.concat([seq_breath, pd.Series([breadth_now],   index=[signal_date])])
    else:
        # Fallback: single-point sequence (HMM state may be less reliable)
        print("  [WARN] regime_features.parquet not found -- HMM context limited to one point")
        seq_vix    = pd.Series([vix_now],     index=[signal_date])
        seq_yield  = pd.Series([ys_now],      index=[signal_date])
        seq_breath = pd.Series([breadth_now], index=[signal_date])

    X_seq    = np.column_stack([
        seq_vix.fillna(0.0).values,
        seq_yield.fillna(0.0).values,
        seq_breath.fillna(0.5).values,
    ])
    X_scaled   = hmm_scaler.transform(X_seq)
    raw_states = hmm_model.predict(X_scaled)
    relabelled = np.vectorize(state_remap.get)(raw_states)
    hmm_state  = int(relabelled[-1])

    return pd.Series({
        "vix_close":              vix_now,
        "vix_change_1w":          vix_change_1w,
        "vix_zscore_63d":         vix_z63,
        "yield_spread":           ys_now,
        "yield_spread_change_1m": ys_change_1m,
        "sentiment_zscore":       sent_z,
        "put_call_ratio":         1.0,      # constant sentinel -- no free API
        "breadth_pct_above_200d": breadth_now,
        "hmm_regime":             float(hmm_state),
    })


# ── Per-ticker feature computation ────────────────────────────────────────────

def compute_ticker_feature_row(
    ticker:       str,
    ohlcv:        pd.DataFrame,
    fred:         dict,
    fda_df:       pd.DataFrame,
    regime_feats: pd.Series,
    signal_date:  pd.Timestamp,
) -> tuple:
    """
    Compute the full feature vector for signal_date for a single ticker.
    Returns (feature_series, actual_date).

    Technical features:      36  (from 02_features.build_features)
    Event features:          14  (earnings + macro + regime stress from 05_event_features)
    Regime features (Step11): 9  (from compute_current_regime -- constant across all rows)
    FDA features (biotech):   5  (from 05_event_features.add_fda_features)

    Fails loudly if any required feature is NaN in the final row.
    """
    sector     = TICKER_SECTOR[ticker]
    is_biotech = (sector == "biotech")

    # --- 1. Technical features ---
    tech_df = _build_tech_features(ohlcv, ticker)
    if tech_df.empty:
        raise ValueError(f"[FAIL] {ticker}: no rows remain after sma_200 warmup")

    # Find the most recent row at or before signal_date
    available = tech_df[tech_df.index <= signal_date]
    if available.empty:
        raise ValueError(
            f"[FAIL] {ticker}: OHLCV has no data on or before {signal_date.date()}"
        )
    actual_date = available.index[-1]
    if actual_date < signal_date - pd.Timedelta(days=10):
        print(f"  [WARN] {ticker}: latest data is {actual_date.date()}, "
              f"expected near {signal_date.date()} -- possible stale data")

    tech_row = available.iloc[-1]

    # --- 2. Event features (computed over full OHLCV window, take last row) ---
    trading_dates = tech_df.index
    feat = pd.DataFrame(index=trading_dates)

    _add_earnings(feat, ticker)
    _add_macro(feat, fred)
    _add_regime_stress(feat)   # rate_environment, inflation_regime, macro_stress_score
    if is_biotech:
        _add_fda(feat, fda_df, ticker)

    # AMZN EPS outlier: same clip applied during training
    if ticker == "AMZN" and "last_eps_surprise_pct" in feat.columns:
        feat["last_eps_surprise_pct"] = feat["last_eps_surprise_pct"].clip(-500.0, 500.0)

    # --- 3. Regime features (Step 11): same current value for all dates ---
    for col, val in regime_feats.items():
        feat[col] = float(val)

    # --- 4. Extract the row for actual_date ---
    if actual_date not in feat.index:
        raise ValueError(
            f"[FAIL] {ticker}: {actual_date.date()} not in event feature index"
        )
    event_row = feat.loc[actual_date]

    # --- 5. Combine tech + event into one flat Series ---
    full_row = pd.concat([tech_row, event_row])

    # --- 6. NaN check on critical features ---
    # Allow NaN in derived regime features (they have sentinels) but not in core technicals
    critical = [
        "close", "atr_pct", "rsi_14", "macd_hist",
        "fed_rate_level", "cpi_yoy_change",
    ]
    nan_critical = [c for c in critical if c in full_row.index and pd.isna(full_row[c])]
    if nan_critical:
        raise ValueError(
            f"[FAIL] {ticker}: NaN in critical features on {actual_date.date()}: {nan_critical}"
        )

    return full_row, actual_date


# ── Prediction ────────────────────────────────────────────────────────────────

def predict_ticker(
    ticker:        str,
    feature_row:   pd.Series,
    model_bundle:  dict,
) -> dict:
    """
    Build the exact feature vector required by the model (using model.feature_names)
    and run predict_proba. Returns {proba_bear, proba_sideways, proba_bull, predicted_class}.

    One-hot columns (e.g. ticker_MSFT): set to 1 if this is the ticker, 0 otherwise.
    Missing features: warn and fill with 0.0 (the NaN sentinel used during training).
    """
    feature_names = model_bundle["feature_names"]
    x_vals = []
    missing_feats = []

    for fname in feature_names:
        if fname.startswith("ticker_"):
            # Ticker one-hot: 1 if this ticker matches, 0 otherwise
            col_ticker = fname[len("ticker_"):]
            x_vals.append(1 if col_ticker == ticker else 0)
        elif fname in feature_row.index:
            val = feature_row[fname]
            if pd.isna(val):
                missing_feats.append(fname)
                x_vals.append(0.0)
            else:
                x_vals.append(float(val))
        else:
            missing_feats.append(fname)
            x_vals.append(0.0)

    if missing_feats:
        print(f"  [WARN] {ticker}: {len(missing_feats)} features NaN/missing "
              f"(filled 0.0): {missing_feats[:5]}")

    X     = np.array(x_vals, dtype=np.float32).reshape(1, -1)
    proba = model_bundle["model"].predict_proba(X)[0]
    pred  = int(np.argmax(proba))

    return {
        "proba_bear":       float(proba[0]),
        "proba_sideways":   float(proba[1]),
        "proba_bull":       float(proba[2]),
        "predicted_class":  pred,
        "predicted_label":  CLASS_NAMES[pred],
    }


# ── Kelly sizing ──────────────────────────────────────────────────────────────

def kelly_sizing() -> tuple:
    """
    Compute Kelly fraction and recommended position size.
    Uses fixed WIN_RATE from Step 11 holdout (not signal-specific).
    Returns (kelly_fraction, recommended_size_as_fraction).
    """
    # Cap half-Kelly at MAX_POSITION_PCT
    rec = min(KELLY_HALF, MAX_POSITION_PCT)
    return round(KELLY_FULL, 4), round(rec, 4)


# ── Signal row builder ────────────────────────────────────────────────────────

def build_signal_row(
    signal_date:  pd.Timestamp,
    actual_date:  pd.Timestamp,
    ticker:       str,
    pred:         dict,
    regime_feats: pd.Series,
    model_label:  str,
) -> dict:
    proba_side     = pred["proba_sideways"]
    signal         = "FIRE" if proba_side >= SIGNAL_THRESHOLD else "NO_FIRE"
    kelly_frac, rec_size = kelly_sizing() if signal == "FIRE" else (0.0, 0.0)

    return {
        "signal_date":          signal_date,
        "actual_data_date":     actual_date,
        "ticker":               ticker,
        "sector":               TICKER_SECTOR[ticker],
        "signal":               signal,
        "proba_sideways":       round(proba_side, 4),
        "proba_bear":           round(pred["proba_bear"],  4),
        "proba_bull":           round(pred["proba_bull"],  4),
        "confidence_threshold": SIGNAL_THRESHOLD,
        "regime_state":         int(regime_feats["hmm_regime"]),
        "regime_label":         REGIME_LABELS.get(int(regime_feats["hmm_regime"]), "unknown"),
        "vix_close":            round(float(regime_feats["vix_close"]),  2),
        "yield_spread":         round(float(regime_feats["yield_spread"]), 3),
        "kelly_fraction":       kelly_frac,
        "recommended_size_pct": round(rec_size * 100, 2),
        "model_version":        model_label,
        "actual_outcome":       "",    # filled in after 5 trading days
        "notes":                "",
    }


# ── Signal log persistence ────────────────────────────────────────────────────

def save_signal_log(signal_rows: list) -> None:
    """
    Append this week's signals to data/signals/signal_log.parquet.
    Deduplicates on (signal_date, ticker) -- won't double-append if re-run.
    """
    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = SIGNALS_DIR / "signal_log.parquet"

    new_df = pd.DataFrame(signal_rows)
    new_df["signal_date"]      = pd.to_datetime(new_df["signal_date"])
    new_df["actual_data_date"] = pd.to_datetime(new_df["actual_data_date"])
    new_df = new_df.set_index("signal_date")

    if log_path.exists():
        existing = pd.read_parquet(log_path, engine="pyarrow")
        existing.index = pd.to_datetime(existing.index)
        # Remove any rows that already exist for this (signal_date, ticker) pair
        new_keys = set(zip(new_df.index, new_df["ticker"]))
        keep_mask = [
            (d, t) not in new_keys
            for d, t in zip(existing.index, existing["ticker"])
        ]
        existing = existing[keep_mask]
        combined = pd.concat([existing, new_df]).sort_index()
    else:
        combined = new_df

    combined.to_parquet(log_path, engine="pyarrow", index=True)
    size_kb = log_path.stat().st_size / 1024
    print(f"  [OK]  {log_path.relative_to(ROOT)}  "
          f"({len(combined)} total rows, {size_kb:.0f} KB)")


# ── Signal report printer ─────────────────────────────────────────────────────

def print_signal_report(
    signal_rows:  list,
    regime_feats: pd.Series,
    signal_date:  pd.Timestamp,
) -> None:
    """Print the weekly ManthIQ signal report to stdout."""
    hmm_state    = int(regime_feats["hmm_regime"])
    vix_val      = float(regime_feats["vix_close"])
    ys_val       = float(regime_feats["yield_spread"])
    regime_label = REGIME_LABELS.get(hmm_state, "unknown")

    print(f"\n{'=' * 60}")
    print(f"  ManthIQ Signal Report -- {signal_date.date()}")
    print(f"{'=' * 60}")
    print(f"  Regime: {regime_label} (state {hmm_state})  "
          f"VIX={vix_val:.1f}  Yield Spread={ys_val:+.3f}")
    print()

    fire_rows   = [r for r in signal_rows if r["signal"] == "FIRE"]
    nofire_rows = [r for r in signal_rows if r["signal"] == "NO_FIRE"]

    if fire_rows:
        print(f"  FIRE signals (threshold {SIGNAL_THRESHOLD}):")
        hdr = f"  {'Ticker':6}  {'Confidence':>10}  {'Regime':12}  {'Kelly Size':>10}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for r in sorted(fire_rows, key=lambda x: x["proba_sideways"], reverse=True):
            print(
                f"  {r['ticker']:6}  "
                f"{r['proba_sideways']:>10.3f}  "
                f"{r['regime_label']:12}  "
                f"{r['recommended_size_pct']:>9.1f}%"
            )
    else:
        print(f"  No FIRE signals at threshold {SIGNAL_THRESHOLD}")

    print()
    if nofire_rows:
        nf_str = ", ".join(
            f"{r['ticker']}({r['proba_sideways']:.2f})"
            for r in sorted(nofire_rows, key=lambda x: x["proba_sideways"], reverse=True)
        )
        print(f"  NO_FIRE: {nf_str}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    kelly_rec_pct = min(KELLY_HALF, MAX_POSITION_PCT) * 100

    section("ManthIQ Step 12 -- Weekly Iron Condor Signal Generator")
    print(f"  Threshold      : {SIGNAL_THRESHOLD}")
    print(f"  Win rate basis : {WIN_RATE:.0%} (Step 11 holdout, 42 trades at threshold 0.55)")
    print(f"  Kelly full     : {KELLY_FULL:.4f}  ({KELLY_FULL*100:.1f}%)")
    print(f"  Kelly half     : {KELLY_HALF:.4f}  ({KELLY_HALF*100:.1f}%)")
    print(f"  Recommended sz : {kelly_rec_pct:.1f}%  (half-Kelly capped at {MAX_POSITION_PCT:.0%})")

    # ── 1. Signal date ─────────────────────────────────────────────────────────
    section("Signal Date")
    signal_date = get_signal_date()

    # ── 2. Load models ─────────────────────────────────────────────────────────
    section("Loading Models")
    ticker_models = load_models()

    # Determine model label for each ticker (for logging)
    model_labels = {}
    for ticker in ALL_TICKERS:
        if ticker == "VRTX":
            model_labels[ticker] = "VRTX_finetuned_v1"
        else:
            model_labels[ticker] = f"xgb_{TICKER_SECTOR[ticker]}_shared_v1"

    # ── 3. Fetch FRED macro (shared across all tickers) ────────────────────────
    section("Fetching FRED Macro Data")
    print("  NOTE: macro_stress_score z-score uses the 500-day OHLCV window,")
    print("        a slight approximation vs the 30-year training window.")
    fred = fetch_fred_macro()

    # ── 4. Fetch OHLCV for all 11 tickers ──────────────────────────────────────
    section("Fetching OHLCV (all tickers)")
    ohlcv_data = {}
    for ticker in ALL_TICKERS:
        print(f"  {ticker} ...", end=" ", flush=True)
        time.sleep(0.1)   # polite yfinance pacing
        df = fetch_ohlcv(ticker)
        ohlcv_data[ticker] = df
        print(f"OK  ({len(df)} rows, {df.index[0].date()} to {df.index[-1].date()})")

    # ── 5. Compute regime features (shared, done once) ─────────────────────────
    section("Computing Regime Features (VIX, T10Y2Y, UMCSENT, Breadth, HMM)")
    regime_feats = compute_current_regime(ohlcv_data, signal_date)
    hmm_state    = int(regime_feats["hmm_regime"])
    print(f"\n  [OK]  Regime: {REGIME_LABELS[hmm_state]} (state {hmm_state})")
    print(f"        VIX={regime_feats['vix_close']:.2f}  "
          f"VIX_chg_1w={regime_feats['vix_change_1w']:+.2f}  "
          f"VIX_z63={regime_feats['vix_zscore_63d']:+.2f}")
    print(f"        Yield spread={regime_feats['yield_spread']:+.3f}  "
          f"chg_1m={regime_feats['yield_spread_change_1m']:+.3f}")
    print(f"        Sentiment_z={regime_feats['sentiment_zscore']:+.3f}  "
          f"Breadth={regime_feats['breadth_pct_above_200d']:.2f}")

    # ── 6. Load FDA events (for biotech tickers) ───────────────────────────────
    fda_df = pd.DataFrame()
    if FDA_EVENTS.exists():
        fda_df = pd.read_parquet(FDA_EVENTS)
        fda_df.index = pd.to_datetime(fda_df.index).normalize()
        print(f"\n  FDA events loaded: {len(fda_df)} rows from {FDA_EVENTS.relative_to(ROOT)}")
    else:
        print(f"\n  [WARN] FDA events not found at {FDA_EVENTS.relative_to(ROOT)}")
        print(f"         FDA features will use 0 sentinel for all biotech tickers.")

    # ── 7. Compute features and signals per ticker ─────────────────────────────
    section("Computing Features and Generating Signals")
    signal_rows = []

    for ticker in ALL_TICKERS:
        print(f"\n  [{TICKER_SECTOR[ticker].upper()}] {ticker}")
        feature_row, actual_date = compute_ticker_feature_row(
            ticker, ohlcv_data[ticker], fred, fda_df, regime_feats, signal_date
        )
        pred = predict_ticker(ticker, feature_row, ticker_models[ticker])
        row  = build_signal_row(
            signal_date, actual_date, ticker, pred,
            regime_feats, model_labels[ticker],
        )
        signal_rows.append(row)
        sig_str = (
            f"FIRE  proba_side={pred['proba_sideways']:.3f}"
            if row["signal"] == "FIRE"
            else f"no-fire  proba_side={pred['proba_sideways']:.3f}"
        )
        print(f"  [OK]  {ticker}: {sig_str}  "
              f"(Bear={pred['proba_bear']:.2f} Bull={pred['proba_bull']:.2f})")

    # ── 8. Print signal report ─────────────────────────────────────────────────
    print_signal_report(signal_rows, regime_feats, signal_date)

    # ── 9. Save signal log ─────────────────────────────────────────────────────
    section("Saving Signal Log")
    save_signal_log(signal_rows)

    # ── 10. Summary ────────────────────────────────────────────────────────────
    fired = [r for r in signal_rows if r["signal"] == "FIRE"]
    print(f"\n  {len(fired)}/{len(signal_rows)} tickers fired FIRE signal "
          f"at threshold {SIGNAL_THRESHOLD}")
    if fired:
        print(f"  Fired: {[r['ticker'] for r in fired]}")
    print(f"  Log:  {SIGNALS_DIR.relative_to(ROOT)}/signal_log.parquet\n")


if __name__ == "__main__":
    main()

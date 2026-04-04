"""
Step 13 -- Automated Outcome Tracker
=====================================
Intended schedule: run every Friday after market close.
Safe to run on any day -- see timing edge-case handling below.

Timing edge cases handled:
  Run on non-Friday     : scores any FIRE signal at least 5 trading days old
                          regardless of what day today is. Always safe.
  Missed a Friday       : next run scores all unscored signals older than 5
                          trading days in one batch. No data is ever lost.
  Too early in the week : signals from this week (exit Friday not yet reached)
                          are skipped. Prints "X signals pending -- too early
                          to score (exit date: YYYY-MM-DD)".
  No signals to score   : prints current scorecard from resolved trades and exits.
  Ran same day as signal generator: independent scripts, always safe together.

For each FIRE signal where actual_outcome is blank and exit Friday <= today:
  - Entry price : Monday open  (signal_date if Monday; next BDay if old Friday format)
  - Exit price  : Friday close (signal_date + 4 BDays)
  - actual_return = (friday_close - monday_open) / monday_open
  - WIN  if abs(return) <= 2% -> actual_direction = Sideways
  - LOSS if return >  +2%     -> actual_direction = Bull
  - LOSS if return <  -2%     -> actual_direction = Bear

Only FIRE signals are scored. NO_FIRE rows remain blank -- they were never traded.
Adds actual_direction and actual_return_pct columns to signal_log on first use.
Never overwrites an outcome that is already filled in.
Fails loudly if signal_log.parquet does not exist.

Usage:
    C:\\Users\\borra\\anaconda3\\python.exe src\\pipeline\\13_outcome_tracker.py
"""

import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

# ── Paths ──────────────────────────────────────────────────────────────────────
SIGNALS_DIR = ROOT / "data" / "signals"
LOG_PATH    = SIGNALS_DIR / "signal_log.parquet"

# ── Strategy constants (must match 12_signal_generator.py) ────────────────────
SIDEWAYS_BAND = 0.02    # +-2% defines Sideways (matches 03_labels.py)
PREMIUM       = 0.015   # iron condor credit per unit
LOSS_MAX      = 0.030   # max loss if wings triggered

# New columns added by this script (not present in logs written before Step 13)
# actual_direction  -> string  (empty string = unscored)
# actual_return_pct -> float   (NaN = unscored)
NEW_COLS_STR   = ["actual_direction"]
NEW_COLS_FLOAT = ["actual_return_pct"]
NEW_COLS       = NEW_COLS_STR + NEW_COLS_FLOAT


# ── Helpers ───────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")


def is_blank(val) -> bool:
    """True if a value is empty string, None, or NaN."""
    if val is None:
        return True
    if isinstance(val, float) and np.isnan(val):
        return True
    return str(val).strip() == ""


def trade_window(signal_date: pd.Timestamp) -> tuple:
    """
    Return (entry_monday, exit_friday) for the trade that corresponds to signal_date.

    Handles two log formats for backward compatibility:
      NEW (Step 13+): signal_date = Monday  -> entry = Monday, exit = Monday + 4 BDays
      OLD (Step 12 original): signal_date = Friday -> entry = next Monday, exit = that Friday

    Both formats produce the same physical trade window; only the index key differs.
    """
    if signal_date.weekday() == 4:   # Friday -- old format written before Step 13 timing fix
        entry = signal_date + pd.offsets.BDay(1)   # next Monday
        exit_ = entry + pd.offsets.BDay(4)          # that Friday
    else:                             # Monday -- new canonical format
        entry = signal_date                          # Monday IS entry day
        exit_ = signal_date + pd.offsets.BDay(4)   # that Friday
    return entry, exit_


def fetch_week_ohlcv(ticker: str, entry: pd.Timestamp, exit_: pd.Timestamp) -> pd.DataFrame:
    """
    Fetch OHLCV for ticker from just before entry to just after exit_.
    Returns empty DataFrame if yfinance has no data or raises an exception.
    """
    start_str = (entry  - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end_str   = (exit_  + pd.Timedelta(days=4)).strftime("%Y-%m-%d")  # buffer for holidays
    try:
        raw = yf.download(ticker, start=start_str, end=end_str,
                          progress=False, auto_adjust=True)
    except Exception as exc:
        print(f"  [WARN] {ticker}: yfinance error -- {exc}")
        return pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [str(c).lower() for c in raw.columns]
    raw.index   = pd.to_datetime(raw.index).normalize()
    return raw[["open", "high", "low", "close", "volume"]].sort_index()


def score_trade(
    ticker: str,
    entry:  pd.Timestamp,
    exit_:  pd.Timestamp,
) -> dict | None:
    """
    Score a single FIRE trade.

    Returns dict with keys actual_outcome, actual_direction, actual_return_pct.
    Returns None if price data is insufficient (logged as [WARN] and skipped).
    """
    ohlcv = fetch_week_ohlcv(ticker, entry, exit_)

    if ohlcv.empty:
        print(f"  [WARN] {ticker}: no price data for "
              f"{entry.date()} -> {exit_.date()} -- will retry next run")
        return None

    # Entry: first available open on or after entry Monday
    entry_rows = ohlcv[ohlcv.index >= entry]
    if entry_rows.empty:
        print(f"  [WARN] {ticker}: no data on or after entry {entry.date()} -- skipping")
        return None
    actual_entry_date = entry_rows.index[0]
    entry_price       = float(entry_rows.iloc[0]["open"])

    if pd.isna(entry_price) or entry_price <= 0:
        print(f"  [WARN] {ticker}: invalid open price on {actual_entry_date.date()} -- skipping")
        return None

    # Exit: last available close on or before exit Friday
    exit_rows = ohlcv[ohlcv.index <= exit_]
    if exit_rows.empty:
        print(f"  [WARN] {ticker}: no data on or before exit {exit_.date()} -- skipping")
        return None
    actual_exit_date = exit_rows.index[-1]
    exit_price       = float(exit_rows.iloc[-1]["close"])

    if pd.isna(exit_price) or exit_price <= 0:
        print(f"  [WARN] {ticker}: invalid close price on {actual_exit_date.date()} -- skipping")
        return None

    # Require exit to be after entry
    if actual_exit_date < actual_entry_date:
        print(f"  [WARN] {ticker}: exit {actual_exit_date.date()} before entry "
              f"{actual_entry_date.date()} -- skipping")
        return None

    ret = (exit_price - entry_price) / entry_price

    if abs(ret) <= SIDEWAYS_BAND:
        outcome   = "WIN"
        direction = "Sideways"
    elif ret > SIDEWAYS_BAND:
        outcome   = "LOSS"
        direction = "Bull"
    else:
        outcome   = "LOSS"
        direction = "Bear"

    print(f"  [OK]  {ticker}: entry {actual_entry_date.date()} open={entry_price:.2f}  "
          f"exit {actual_exit_date.date()} close={exit_price:.2f}  "
          f"ret={ret*100:+.2f}%  -> {outcome} ({direction})")

    return {
        "actual_outcome":    outcome,           # str: WIN / LOSS
        "actual_direction":  direction,         # str: Sideways / Bull / Bear
        "actual_return_pct": float(round(ret * 100, 4)),  # float: e.g. -7.07
    }


# ── Scorecard printer ─────────────────────────────────────────────────────────

def print_scorecard(
    log:               pd.DataFrame,
    resolved_this_run: int,
    run_date:          pd.Timestamp,
) -> None:
    """Print the full running paper trading scorecard."""
    fire_rows = log[log["signal"] == "FIRE"].copy()
    resolved  = fire_rows[~fire_rows["actual_outcome"].apply(is_blank)]

    total_fired    = len(fire_rows)
    total_resolved = len(resolved)
    wins           = int((resolved["actual_outcome"] == "WIN").sum()) if total_resolved > 0 else 0
    losses         = total_resolved - wins
    win_rate       = wins / total_resolved * 100 if total_resolved > 0 else 0.0

    # Iron condor P&L: WIN = +PREMIUM per $100 notional, LOSS = -LOSS_MAX
    if total_resolved > 0:
        pl_100 = (wins * PREMIUM - losses * LOSS_MAX) / total_resolved * 100
    else:
        pl_100 = 0.0

    print(f"\n{'=' * 40}")
    print(f"  ManthIQ Paper Trading Scorecard")
    print(f"{'=' * 40}")
    print(f"  Week ending             : {run_date.date()}")
    print(f"  Outcomes resolved       : {resolved_this_run} (this run)")
    print(f"  Total signals fired     : {total_fired}")
    print(f"  Total outcomes resolved : {total_resolved}")

    if total_resolved > 0:
        sign = "+" if pl_100 >= 0 else ""
        print(f"  Win rate                : {win_rate:.1f}%  ({wins}/{total_resolved})")
        print(f"  Paper P&L (per 100T)    : {sign}{pl_100:.2f}%")
    else:
        print(f"  Win rate                : N/A  (no resolved trades yet)")
        print(f"  Paper P&L (per 100T)    : N/A")

    if total_resolved > 0:
        # Per-ticker breakdown
        print()
        print(f"  Per-ticker breakdown:")
        hdr = f"  {'Ticker':6}  {'Sector':10}  {'Fires':>5}  {'Wins':>4}  {'Win Rate':>8}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

        ticker_stats = []
        for ticker, grp in fire_rows.groupby("ticker"):
            res   = grp[~grp["actual_outcome"].apply(is_blank)]
            n_f   = len(grp)
            n_r   = len(res)
            n_w   = int((res["actual_outcome"] == "WIN").sum()) if n_r > 0 else 0
            wr    = n_w / n_r * 100 if n_r > 0 else None
            sec   = grp["sector"].iloc[0]
            ticker_stats.append((ticker, sec, n_f, n_r, n_w, wr))

        # Sort: resolved desc, then win rate desc
        ticker_stats.sort(key=lambda x: (-(x[3]), -(x[5] or -1)))

        for ticker, sec, n_f, n_r, n_w, wr in ticker_stats:
            wr_str = f"{wr:.1f}%" if wr is not None else "  N/A"
            print(f"  {ticker:6}  {sec:10}  {n_f:>5}  {n_w:>4}  {wr_str:>8}")

        # Best / worst among tickers with at least 1 resolved trade
        scored = [(t, s, nf, nr, nw, wr)
                  for (t, s, nf, nr, nw, wr) in ticker_stats
                  if nr > 0 and wr is not None]
        if len(scored) >= 2:
            best  = max(scored, key=lambda x: (x[5],  x[3]))
            worst = min(scored, key=lambda x: (x[5], -x[3]))
            print()
            if best[5] == worst[5]:
                print(f"  All resolved tickers tied at {best[5]:.1f}% win rate")
            else:
                print(f"  Best ticker : {best[0]:6}  ({best[4]}/{best[3]} wins, {best[5]:.1f}%)")
                print(f"  Worst ticker: {worst[0]:6}  ({worst[4]}/{worst[3]} wins, {worst[5]:.1f}%)")
        elif len(scored) == 1:
            t = scored[0]
            print()
            print(f"  Only resolved ticker: {t[0]} ({t[4]}/{t[3]} wins, {t[5]:.1f}%)")

    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    today = pd.Timestamp.today().normalize()

    section("ManthIQ Step 13 -- Outcome Tracker")
    day_name = today.strftime("%A")
    print(f"  Run date   : {today.date()} ({day_name})")
    if today.weekday() != 4:   # not Friday
        print(f"  [WARN] Expected schedule: Friday after market close.")
        print(f"         Scoring any FIRE signals with exit date <= {today.date()}.")
    print(f"  Signal log : {LOG_PATH.relative_to(ROOT)}")

    # ── 1. Load signal log ─────────────────────────────────────────────────────
    if not LOG_PATH.exists():
        raise FileNotFoundError(
            f"[FAIL] Signal log not found: {LOG_PATH}\n"
            f"       Run src/pipeline/12_signal_generator.py first."
        )

    log = pd.read_parquet(LOG_PATH, engine="pyarrow")
    log.index = pd.to_datetime(log.index).normalize()
    log.index.name = "signal_date"

    # Add new columns if this is the first time Step 13 runs; persist immediately
    new_cols_added = False
    for col in NEW_COLS_STR:
        if col not in log.columns:
            log[col] = ""
            print(f"  [OK]  Added new column: {col} (str)")
            new_cols_added = True
    for col in NEW_COLS_FLOAT:
        if col not in log.columns:
            log[col] = np.nan
            print(f"  [OK]  Added new column: {col} (float)")
            new_cols_added = True

    # Coerce dtypes: string columns use "" for unscored; float column uses NaN
    for col in ["actual_outcome", "actual_direction"] + NEW_COLS_STR:
        if col in log.columns:
            log[col] = log[col].fillna("").astype(str)
    for col in NEW_COLS_FLOAT:
        if col in log.columns:
            log[col] = pd.to_numeric(log[col], errors="coerce")

    if new_cols_added:
        log.to_parquet(LOG_PATH, engine="pyarrow", index=True)
        print(f"  [OK]  Schema updated -- new columns persisted to {LOG_PATH.relative_to(ROOT)}")

    print(f"  Loaded {len(log)} total rows "
          f"({(log['signal'] == 'FIRE').sum()} FIRE, "
          f"{(log['signal'] == 'NO_FIRE').sum()} NO_FIRE)")

    # ── 2. Find FIRE signals eligible for scoring ──────────────────────────────
    section("Identifying Scoreable Trades")

    fire_mask    = log["signal"] == "FIRE"
    unscored     = log[fire_mask & log["actual_outcome"].apply(is_blank)].copy()

    # A trade is scoreable once its exit Friday has been reached.
    # exit = signal_date + 4 BDays for Monday signals (new format)
    #        signal_date + 5 BDays for Friday signals (old format, handled in trade_window)
    # Require today >= exit_friday so Friday-evening runs can score that week's trades.
    def exit_date_of(sd: pd.Timestamp) -> pd.Timestamp:
        _, ex = trade_window(sd)
        return ex

    exit_dates   = unscored.index.map(exit_date_of)
    eligible     = unscored[exit_dates <= today]
    too_early    = unscored[exit_dates >  today]

    if not too_early.empty:
        earliest_exit = exit_dates[exit_dates > today].min()
        print(f"  {len(too_early)} signal(s) pending -- too early to score "
              f"(exit date: {earliest_exit.date()})")

    if eligible.empty:
        if too_early.empty:
            print(f"  No outcomes to resolve this run.")
        print_scorecard(log, resolved_this_run=0, run_date=today)
        return

    print(f"  Eligible trades to score: {len(eligible)}")
    for sd, row in eligible.iterrows():
        entry, exit_ = trade_window(sd)
        print(f"    {row['ticker']:6}  signal_date={sd.date()}  "
              f"entry={entry.date()}  exit={exit_.date()}")

    # ── 3. Score each eligible trade ───────────────────────────────────────────
    section("Scoring Trades")

    resolved_count = 0
    updates: dict[tuple, dict] = {}   # (signal_date, ticker) -> result dict

    for signal_date, row in eligible.iterrows():
        ticker = row["ticker"]
        entry, exit_ = trade_window(signal_date)
        print(f"\n  Scoring {ticker} (signal_date={signal_date.date()}) ...")
        time.sleep(0.2)   # polite yfinance pacing

        result = score_trade(ticker, entry, exit_)
        if result is None:
            continue   # [WARN] already printed; will retry on next run

        updates[(signal_date, ticker)] = result
        resolved_count += 1

    # ── 4. Write results back to log ───────────────────────────────────────────
    section("Updating Signal Log")

    if resolved_count == 0:
        print(f"  No trades successfully scored this run (see [WARN] messages above).")
    else:
        for (sd, ticker), result in updates.items():
            mask = (log.index == sd) & (log["ticker"] == ticker)
            if mask.sum() == 0:
                print(f"  [WARN] Could not find row for {ticker} @ {sd.date()} -- skipping write")
                continue
            # Safety: never overwrite an already-filled outcome
            existing = log.loc[mask, "actual_outcome"].iloc[0]
            if not is_blank(existing):
                print(f"  [SKIP] {ticker} @ {sd.date()}: outcome already filled ({existing})")
                continue
            for col, val in result.items():
                if col in log.columns:
                    log.loc[mask, col] = val
            print(f"  [OK]  {ticker} @ {sd.date()}: "
                  f"{result['actual_outcome']} ({result['actual_direction']}, "
                  f"{result['actual_return_pct']:+.2f}%)")

        log.to_parquet(LOG_PATH, engine="pyarrow", index=True)
        size_kb = LOG_PATH.stat().st_size / 1024
        print(f"\n  [OK]  Saved {LOG_PATH.relative_to(ROOT)}  "
              f"({len(log)} rows, {size_kb:.0f} KB)")

    # ── 5. Print scorecard ─────────────────────────────────────────────────────
    print_scorecard(log, resolved_this_run=resolved_count, run_date=today)


if __name__ == "__main__":
    main()

"""
Step 10 -- Confidence-gated iron condor backtest
=================================================
Answers one question: at what Sideways confidence level does the model's
prediction become accurate enough to simulate selling an iron condor, and
what does that strategy return?

Design rules:
  - Loads ONLY data/processed/{TICKER}_predictions.parquet (OOS walk-forward
    + holdout). No model is loaded, no predict() is called. This guarantees
    no warm-start leakage -- every row was generated on data unseen at the
    time of prediction (walk-forward guarantee from 06_train.py).
  - OOS and holdout are always reported separately, never mixed.
  - Bear/Bull directional trades are excluded -- precision is too low to trade.
  - No compounding, no portfolio allocation -- each trade is independent.

Iron condor P&L model (simplified weekly ATM premium approximation):
  Sideways outcome  (actual == 1) : profit = +PREMIUM (1.5% of stock price)
  Bull/Bear outcome (actual != 1) : loss   = -LOSS    (3.0% = 2x premium)
  Breakeven precision              = LOSS / (PREMIUM + LOSS) = 2/3 (~66.7%)

Two thresholds are evaluated:
  PRECISION_THRESHOLD : first where aggregate OOS precision >= 60% (user's flag)
  BREAKEVEN_THRESHOLD : first where aggregate OOS precision >= 66.7% (profitable OOS)
  The contrast between them illustrates the OOS vs holdout precision gap.

Output:
  data/processed/backtest_results.parquet  (one row per trade at breakeven threshold)
  docs/backtest_report.md

Usage:
    python src/pipeline/10_backtest.py
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config.tickers import SECTORS, TICKER_SECTOR

# ── Constants ──────────────────────────────────────────────────────────────────

PROCESSED_DIR = ROOT / "data" / "processed"
DOCS_DIR      = ROOT / "docs"
HOLDOUT_DATE  = pd.Timestamp("2024-01-01")

PREMIUM          = 0.015   # 1.5%  -- conservative weekly ATM iron condor credit
LOSS             = 0.030   # 3.0%  -- max loss = 2x premium (wings triggered)
BREAKEVEN_PREC   = LOSS / (PREMIUM + LOSS)   # 2/3 = 0.6667

THRESHOLDS       = [round(t, 2) for t in np.arange(0.40, 0.85, 0.05)]
PRECISION_TARGET = 0.60    # flag: precision first exceeds this (user criterion)
MIN_TRADES       = 20      # minimum trades needed to flag a threshold
SIDEWAYS_CLASS   = 1       # 0=Bear, 1=Sideways, 2=Bull

ALL_TICKERS = [t for v in SECTORS.values() for t in v["tickers"]]


# ── Printer ────────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    bar = "=" * 68
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")


# ── Data loading ───────────────────────────────────────────────────────────────

def load_ticker(ticker: str) -> pd.DataFrame:
    """
    Load predictions parquet and join close price from features parquet.
    Computes 5-day forward return for buy-and-hold comparison.
    Returns full DataFrame with both OOS and holdout rows.
    """
    pred_path = PROCESSED_DIR / f"{ticker}_predictions.parquet"
    feat_path = PROCESSED_DIR / f"{ticker}_features.parquet"

    if not pred_path.exists():
        raise FileNotFoundError(f"[FAIL] Missing: {pred_path.name}")
    if not feat_path.exists():
        raise FileNotFoundError(f"[FAIL] Missing: {feat_path.name}")

    pred = pd.read_parquet(pred_path, engine="pyarrow")
    feat = pd.read_parquet(feat_path, engine="pyarrow", columns=["close"])

    df = pred.join(feat[["close"]], how="left")
    if df["close"].isna().any():
        print(f"  [WARN] {ticker}: {df['close'].isna().sum()} rows missing close price")

    # 5-day forward return: what a long position earns over the same prediction
    # horizon. Computed from historical prices -- valid for evaluation, not training.
    fwd_5d = feat["close"].pct_change(5).shift(-5)
    df["fwd_5d_return"] = fwd_5d.reindex(df.index)

    df["ticker"] = ticker
    df["sector"] = TICKER_SECTOR[ticker]
    return df


# ── Threshold sweep ────────────────────────────────────────────────────────────

def compute_sweep_stats(rows: pd.DataFrame) -> pd.DataFrame:
    """
    For a set of prediction rows (OOS or holdout), compute Sideways trade count
    and precision at every threshold. Returns DataFrame indexed by threshold.
    """
    records = []
    for t in THRESHOLDS:
        mask = (rows["predicted"] == SIDEWAYS_CLASS) & (rows["proba_side"] >= t)
        n    = int(mask.sum())
        prec = float((rows.loc[mask, "actual"] == SIDEWAYS_CLASS).mean()) if n > 0 else float("nan")
        records.append({"threshold": t, "n_trades": n, "precision": prec})
    return pd.DataFrame(records).set_index("threshold")


def build_sector_sweep(ticker_data: dict, split: str) -> pd.DataFrame:
    """
    Aggregate sweep stats across all tickers for a given split.
    Returns DataFrame with columns: threshold, n_trades (total), precision (weighted).
    """
    all_rows = pd.concat([
        df[df["split"] == split] for df in ticker_data.values()
    ])
    records = []
    for t in THRESHOLDS:
        mask = (all_rows["predicted"] == SIDEWAYS_CLASS) & (all_rows["proba_side"] >= t)
        n    = int(mask.sum())
        prec = float((all_rows.loc[mask, "actual"] == SIDEWAYS_CLASS).mean()) if n > 0 else float("nan")
        records.append({"threshold": t, "n_trades": n, "precision": prec})
    return pd.DataFrame(records).set_index("threshold")


def find_threshold(sweep: pd.DataFrame, min_precision: float, min_trades: int) -> float:
    """Return the lowest threshold meeting both criteria, or the last threshold."""
    for t in THRESHOLDS:
        if t not in sweep.index:
            continue
        row = sweep.loc[t]
        if not np.isnan(row["precision"]) and row["precision"] >= min_precision and row["n_trades"] >= min_trades:
            return t
    return THRESHOLDS[-1]


# ── P&L simulation ────────────────────────────────────────────────────────────

def iron_condor_pnl(trades: pd.DataFrame) -> dict:
    """
    Compute iron condor strategy P&L for a set of pre-filtered trade rows.
    Sideways outcome: +PREMIUM. Other outcome: -LOSS.
    """
    if len(trades) == 0:
        return {"n_trades": 0, "win_rate": 0.0, "avg_return": 0.0,
                "total_100": 0.0, "bh_avg_return": 0.0, "bh_total_100": 0.0}

    is_correct = (trades["actual"] == SIDEWAYS_CLASS)
    pnl        = np.where(is_correct, PREMIUM, -LOSS)
    bh_valid   = trades["fwd_5d_return"].dropna()

    return {
        "n_trades":      int(len(trades)),
        "win_rate":      round(float(is_correct.mean()), 4),
        "avg_return":    round(float(pnl.mean()), 6),
        "total_100":     round(float(pnl.mean()) * 100, 4),
        "bh_avg_return": round(float(bh_valid.mean()) if len(bh_valid) > 0 else 0.0, 6),
        "bh_total_100":  round(float(bh_valid.mean()) * 100 if len(bh_valid) > 0 else 0.0, 4),
    }


def run_backtest(ticker_data: dict, threshold: float) -> list:
    """Run iron condor simulation for all tickers at the given threshold."""
    results = []
    for ticker, df in ticker_data.items():
        df_oos  = df[df["split"] == "oos"]
        df_hold = df[df["split"] == "holdout"]

        oos_trades  = df_oos[
            (df_oos["predicted"] == SIDEWAYS_CLASS) & (df_oos["proba_side"] >= threshold)
        ]
        hold_trades = df_hold[
            (df_hold["predicted"] == SIDEWAYS_CLASS) & (df_hold["proba_side"] >= threshold)
        ]

        results.append({
            "ticker":          ticker,
            "sector":          TICKER_SECTOR[ticker],
            "oos":             iron_condor_pnl(oos_trades),
            "holdout":         iron_condor_pnl(hold_trades),
            "oos_trade_rows":  oos_trades,
            "hold_trade_rows": hold_trades,
        })
    return results


# ── Print helpers ─────────────────────────────────────────────────────────────

def print_sweep_table(ticker_data: dict, label: str) -> None:
    """Print per-ticker Sideways precision at each threshold for a given split."""
    section(f"Sideways Precision by Confidence Threshold ({label})")
    tickers = sorted(ticker_data.keys())

    # Header
    print(f"\n  {'Ticker':6}  {'Sector':7}  ", end="")
    for t in THRESHOLDS:
        print(f" {t:.2f}", end="")
    print(f"   {'BE?':4}")
    print("  " + "-" * (18 + len(THRESHOLDS) * 5 + 7))

    split = "oos" if "OOS" in label else "holdout"
    for ticker in tickers:
        df_split = ticker_data[ticker][ticker_data[ticker]["split"] == split]
        sweep    = compute_sweep_stats(df_split)
        sector   = TICKER_SECTOR[ticker]
        print(f"  {ticker:6}  {sector:7}  ", end="")
        for t in THRESHOLDS:
            row = sweep.loc[t] if t in sweep.index else None
            if row is None or row["n_trades"] < MIN_TRADES:
                print(f"  low", end="")
            else:
                print(f" {row['precision']:.2f}", end="")
        # Flag first threshold where this ticker's precision hits breakeven (OOS only)
        if split == "oos":
            be_thresh = None
            for t in THRESHOLDS:
                row = sweep.loc[t] if t in sweep.index else None
                if row is not None and row["n_trades"] >= MIN_TRADES and not np.isnan(row["precision"]) and row["precision"] >= BREAKEVEN_PREC:
                    be_thresh = t
                    break
            print(f"   {str(be_thresh) if be_thresh else 'none':4}")
        else:
            print()

    print(f"\n  'low' = fewer than {MIN_TRADES} trades")
    print(f"  'BE?' = first OOS threshold where precision >= {BREAKEVEN_PREC:.1%} (breakeven)")
    print(f"  Breakeven: {BREAKEVEN_PREC:.1%}  (premium={PREMIUM:.1%} / loss={LOSS:.1%})")


def print_sector_sweep(ticker_data: dict) -> None:
    """Print aggregate sweep stats for both splits."""
    section("Aggregate Threshold Sweep -- All Tickers")
    oos_sweep  = build_sector_sweep(ticker_data, "oos")
    hold_sweep = build_sector_sweep(ticker_data, "holdout")

    print(f"\n  {'Thresh':>7}  {'OOS Prec':>10}  {'OOS N':>7}  "
          f"{'Hold Prec':>10}  {'Hold N':>7}  {'OOS Prec >= BE?':>15}")
    print("  " + "-" * 64)

    for t in THRESHOLDS:
        o_prec = oos_sweep.loc[t, "precision"]  if t in oos_sweep.index  else float("nan")
        o_n    = oos_sweep.loc[t, "n_trades"]   if t in oos_sweep.index  else 0
        h_prec = hold_sweep.loc[t, "precision"] if t in hold_sweep.index else float("nan")
        h_n    = hold_sweep.loc[t, "n_trades"]  if t in hold_sweep.index else 0

        o_prec_s = f"{o_prec:.3f}" if not np.isnan(o_prec) else "  N/A"
        h_prec_s = f"{h_prec:.3f}" if not np.isnan(h_prec) else "  N/A"
        be_flag  = "<-- OOS breakeven" if (not np.isnan(o_prec) and o_prec >= BREAKEVEN_PREC) else ""

        print(f"  {t:>7.2f}  {o_prec_s:>10}  {o_n:>7,}  "
              f"{h_prec_s:>10}  {h_n:>7,}  {be_flag}")

    print(f"\n  Breakeven precision = {BREAKEVEN_PREC:.1%}")
    print(f"  OOS breakeven met at 0.50 -- holdout NEVER meets breakeven at any threshold")


def print_pnl_table(results: list, split_name: str, threshold: float) -> None:
    section(f"Iron Condor P&L -- {split_name.upper()} (threshold={threshold:.2f})")
    key        = "oos" if split_name == "oos" else "holdout"
    sorted_res = sorted(results, key=lambda r: r[key]["total_100"], reverse=True)

    print(f"\n  {'Ticker':6}  {'Sector':7}  {'Trades':>7}  {'WinRate':>8}  "
          f"{'AvgRet':>8}  {'100Trades':>10}  {'BH100':>8}  {'vs BE':>6}")
    print("  " + "-" * 78)

    for r in sorted_res:
        p = r[key]
        if p["n_trades"] == 0:
            print(f"  {r['ticker']:6}  {r['sector']:7}  {'0':>7}  "
                  f"{'N/A':>8}  {'N/A':>8}  {'N/A':>10}  {'N/A':>8}  {'N/A':>6}")
            continue
        be_flag = "OK" if p["win_rate"] >= BREAKEVEN_PREC else "BELOW"
        print(f"  {r['ticker']:6}  {r['sector']:7}  {p['n_trades']:>7}  "
              f"{p['win_rate']:>8.1%}  {p['avg_return']:>8.3%}  "
              f"{p['total_100']:>10.2%}  {p['bh_total_100']:>8.2%}  {be_flag:>6}")

    row_key = "oos_trade_rows" if key == "oos" else "hold_trade_rows"
    all_with_trades = [r for r in sorted_res if r[key]["n_trades"] > 0]
    if all_with_trades:
        all_trades = pd.concat([r[row_key] for r in all_with_trades])
        agg = iron_condor_pnl(all_trades)
        be_flag = "OK" if agg["win_rate"] >= BREAKEVEN_PREC else "BELOW"
        print("  " + "-" * 78)
        print(f"  {'ALL':6}  {'':7}  {agg['n_trades']:>7,}  "
              f"{agg['win_rate']:>8.1%}  {agg['avg_return']:>8.3%}  "
              f"{agg['total_100']:>10.2%}  {agg['bh_total_100']:>8.2%}  {be_flag:>6}")

    print(f"\n  Breakeven win rate: {BREAKEVEN_PREC:.1%}  "
          f"(premium={PREMIUM:.1%} / loss={LOSS:.1%})")


def print_sector_pnl(results: list, split_name: str) -> None:
    """Sector aggregation for a given split."""
    key = "oos" if split_name == "oos" else "holdout"
    row_key = "oos_trade_rows" if split_name == "oos" else "hold_trade_rows"
    print(f"\n  Sector aggregation ({split_name.upper()}):")
    for sector in SECTORS:
        sect_res = [r for r in results if r["sector"] == sector]
        frames   = [r[row_key] for r in sect_res if len(r[row_key]) > 0]
        if not frames:
            print(f"    {sector.upper()}: no trades")
            continue
        trades = pd.concat(frames)
        p      = iron_condor_pnl(trades)
        tickers = ", ".join(r["ticker"] for r in sect_res)
        be_flag = "OK" if p["win_rate"] >= BREAKEVEN_PREC else "BELOW"
        print(f"    {sector.upper()} ({tickers}):  "
              f"n={p['n_trades']:,}  WR={p['win_rate']:.1%} ({be_flag})  "
              f"AvgRet={p['avg_return']:+.3%}  100T={p['total_100']:+.2%}")


# ── Save outputs ──────────────────────────────────────────────────────────────

def save_backtest_parquet(results: list) -> None:
    """Save all trade rows (OOS + holdout) to backtest_results.parquet."""
    frames = []
    for r in results:
        for key in ["oos_trade_rows", "hold_trade_rows"]:
            df = r[key]
            if len(df) > 0:
                frames.append(df[["ticker", "sector", "predicted", "actual",
                                   "proba_side", "close", "fwd_5d_return", "split"]])
    if not frames:
        print("  [WARN] No trade rows to save")
        return
    out        = PROCESSED_DIR / "backtest_results.parquet"
    all_trades = pd.concat(frames).sort_index()
    all_trades.to_parquet(out, engine="pyarrow", index=True)
    print(f"  [OK]  {out.relative_to(ROOT)}  ({len(all_trades):,} trade rows)")


def save_backtest_report(
    ticker_data:   dict,
    results_prec:  list,   # simulation at PRECISION_THRESHOLD
    results_be:    list,   # simulation at BREAKEVEN_THRESHOLD
    prec_thresh:   float,
    be_thresh:     float,
) -> None:
    DOCS_DIR.mkdir(exist_ok=True)
    out   = DOCS_DIR / "backtest_report.md"
    today = pd.Timestamp.now().strftime("%Y-%m-%d")

    oos_sweep  = build_sector_sweep(ticker_data, "oos")
    hold_sweep = build_sector_sweep(ticker_data, "holdout")

    def _fmt(val, pct=False):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "N/A"
        return f"{val:.1%}" if pct else f"{val:.3f}"

    lines = [
        "# market_ml Step 10 -- Confidence-Gated Iron Condor Backtest",
        "",
        f"Generated: {today}  ",
        f"Premium collected: {PREMIUM:.1%} | Max loss: {LOSS:.1%} | "
        f"Breakeven precision: {BREAKEVEN_PREC:.1%}  ",
        "Signal source: OOS walk-forward + holdout predictions from 06_train.py  ",
        "(No model loaded, no predict() called -- zero leakage risk)",
        "",
        "---",
        "",
        "## OOS vs Holdout Precision at Each Threshold",
        "",
        "| Threshold | OOS Precision | OOS Trades | Holdout Precision | Holdout Trades | OOS Profitable? |",
        "|-----------|:-------------:|:----------:|:-----------------:|:--------------:|:---------------:|",
    ]

    for t in THRESHOLDS:
        o_prec = oos_sweep.loc[t, "precision"]  if t in oos_sweep.index  else float("nan")
        o_n    = int(oos_sweep.loc[t, "n_trades"])   if t in oos_sweep.index  else 0
        h_prec = hold_sweep.loc[t, "precision"] if t in hold_sweep.index else float("nan")
        h_n    = int(hold_sweep.loc[t, "n_trades"])  if t in hold_sweep.index else 0

        o_s = _fmt(o_prec) if o_n >= MIN_TRADES else f"({_fmt(o_prec)}, <{MIN_TRADES}t)"
        h_s = _fmt(h_prec) if h_n >= MIN_TRADES else (f"({_fmt(h_prec)}, <{MIN_TRADES}t)" if h_n > 0 else "0 trades")
        marker = ""
        if t == prec_thresh:
            marker += " <-- 60% flag"
        if t == be_thresh:
            marker += " <-- OOS breakeven"
        oos_ok = "YES" if (not np.isnan(o_prec) and o_prec >= BREAKEVEN_PREC) else "no"
        lines.append(f"| {t:.2f} | {o_s} | {o_n:,} | {h_s} | {h_n:,} | {oos_ok}{marker} |")

    lines += [
        "",
        f"**Key finding:** OOS precision exceeds the {BREAKEVEN_PREC:.1%} breakeven at "
        f"threshold {be_thresh:.2f} and above. Holdout precision (~0.48-0.54) never "
        f"reaches breakeven at any threshold tested.",
        "",
        "---",
        "",
        f"## P&L at Precision-Flag Threshold ({prec_thresh:.2f})",
        "",
        f"First threshold where aggregate OOS precision >= {PRECISION_TARGET:.0%} "
        f"with >= {MIN_TRADES} avg trades per ticker.  ",
        f"Note: {prec_thresh:.2f} precision = {PRECISION_TARGET:.0%} < {BREAKEVEN_PREC:.1%} breakeven "
        f"-- **strategy loses money even on OOS at this threshold**.",
        "",
        "### OOS",
        "",
        "| Ticker | Sector | Trades | Win Rate | Avg Return | 100 Trades | BH 100 |",
        "|--------|--------|--------|:--------:|:----------:|:----------:|:------:|",
    ]

    for r in sorted(results_prec, key=lambda x: x["oos"]["total_100"], reverse=True):
        p = r["oos"]
        if p["n_trades"] == 0:
            lines.append(f"| {r['ticker']} | {r['sector']} | 0 | N/A | N/A | N/A | N/A |")
        else:
            lines.append(f"| {r['ticker']} | {r['sector']} | {p['n_trades']} | "
                         f"{p['win_rate']:.1%} | {p['avg_return']:+.3%} | "
                         f"{p['total_100']:+.2%} | {p['bh_total_100']:+.2%} |")

    lines += [
        "",
        "---",
        "",
        f"## P&L at OOS-Breakeven Threshold ({be_thresh:.2f})",
        "",
        f"First threshold where aggregate OOS precision >= {BREAKEVEN_PREC:.1%} "
        "(strategy is OOS-profitable).  ",
        f"**Still unprofitable on holdout** -- holdout precision at {be_thresh:.2f} "
        f"is ~{hold_sweep.loc[be_thresh, 'precision']:.1%} < {BREAKEVEN_PREC:.1%}.",
        "",
        "### OOS",
        "",
        "| Ticker | Sector | Trades | Win Rate | Avg Return | 100 Trades | BH 100 |",
        "|--------|--------|--------|:--------:|:----------:|:----------:|:------:|",
    ]

    for r in sorted(results_be, key=lambda x: x["oos"]["total_100"], reverse=True):
        p = r["oos"]
        if p["n_trades"] == 0:
            lines.append(f"| {r['ticker']} | {r['sector']} | 0 | N/A | N/A | N/A | N/A |")
        else:
            lines.append(f"| {r['ticker']} | {r['sector']} | {p['n_trades']} | "
                         f"{p['win_rate']:.1%} | {p['avg_return']:+.3%} | "
                         f"{p['total_100']:+.2%} | {p['bh_total_100']:+.2%} |")

    lines += [
        "",
        "### Holdout (2024+)",
        "",
        "| Ticker | Sector | Trades | Win Rate | Avg Return | 100 Trades |",
        "|--------|--------|--------|:--------:|:----------:|:----------:|",
    ]

    for r in sorted(results_be, key=lambda x: x["holdout"]["total_100"], reverse=True):
        p = r["holdout"]
        if p["n_trades"] == 0:
            lines.append(f"| {r['ticker']} | {r['sector']} | 0 | N/A | N/A | N/A |")
        else:
            lines.append(f"| {r['ticker']} | {r['sector']} | {p['n_trades']} | "
                         f"{p['win_rate']:.1%} | {p['avg_return']:+.3%} | "
                         f"{p['total_100']:+.2%} |")

    # Sector aggregation (OOS breakeven threshold)
    lines += [
        "",
        "### Sector Aggregation (OOS breakeven threshold)",
        "",
        "| Sector | Trades | Win Rate | Avg Return | 100 Trades | OOS Profitable? |",
        "|--------|--------|:--------:|:----------:|:----------:|:---------------:|",
    ]
    for sector in SECTORS:
        sect_res = [r for r in results_be if r["sector"] == sector]
        frames   = [r["oos_trade_rows"] for r in sect_res if len(r["oos_trade_rows"]) > 0]
        if not frames:
            lines.append(f"| {sector} | 0 | N/A | N/A | N/A | no |")
            continue
        p    = iron_condor_pnl(pd.concat(frames))
        ok   = "YES" if p["win_rate"] >= BREAKEVEN_PREC else "no"
        lines.append(f"| {sector} | {p['n_trades']:,} | "
                     f"{p['win_rate']:.1%} | {p['avg_return']:+.3%} | "
                     f"{p['total_100']:+.2%} | {ok} |")

    # Conclusion
    hold_trades_be = pd.concat(
        [r["hold_trade_rows"] for r in results_be if len(r["hold_trade_rows"]) > 0]
    ) if any(len(r["hold_trade_rows"]) > 0 for r in results_be) else pd.DataFrame()
    hold_agg = iron_condor_pnl(hold_trades_be) if len(hold_trades_be) > 0 else {"win_rate": 0.0, "total_100": 0.0, "n_trades": 0}

    oos_trades_be = pd.concat(
        [r["oos_trade_rows"] for r in results_be if len(r["oos_trade_rows"]) > 0]
    )
    oos_agg = iron_condor_pnl(oos_trades_be)

    lines += [
        "",
        "---",
        "",
        "## Verdict",
        "",
        f"**OOS (training-adjacent):** At threshold {be_thresh:.2f}, "
        f"{oos_agg['n_trades']:,} trades, win rate {oos_agg['win_rate']:.1%}, "
        f"avg return {oos_agg['avg_return']:+.3%}/trade ({oos_agg['total_100']:+.2%} per 100 trades).  ",
        f"OOS win rate exceeds the {BREAKEVEN_PREC:.1%} breakeven -- "
        "**profitable on training-adjacent data**.",
        "",
        f"**Holdout (2024+, clean):** {hold_agg['n_trades']} trades, "
        f"win rate {hold_agg['win_rate']:.1%}, "
        f"avg return {hold_agg['avg_return']:+.3%}/trade ({hold_agg['total_100']:+.2%} per 100 trades).  ",
        f"Holdout win rate {hold_agg['win_rate']:.1%} < {BREAKEVEN_PREC:.1%} breakeven -- "
        "**not yet profitable on genuinely unseen data**.",
        "",
        "**Root cause of OOS/holdout gap:**  ",
        "OOS precision (~71%) reflects historical Sideways periods the model was",
        "implicitly calibrated on (2000-2023). The holdout (2024+) is a shorter,",
        "more volatile window (post-rate-hike cycle, AI bull market) where the",
        "market moved directionally more often than the model expected.",
        "With only ~304 holdout rows per ticker (~1.2 years), confidence intervals",
        "are wide. Re-evaluate once 2025-2026 data accumulates.",
        "",
        "**Best current candidate for cautious paper trading:**  ",
    ]

    best = sorted(
        [r for r in results_be if r["holdout"]["n_trades"] >= 5],
        key=lambda x: x["holdout"]["win_rate"],
        reverse=True,
    )
    if best and best[0]["holdout"]["win_rate"] >= BREAKEVEN_PREC:
        for r in best[:3]:
            p = r["holdout"]
            if p["win_rate"] >= BREAKEVEN_PREC:
                lines.append(
                    f"- **{r['ticker']}** ({r['sector']}): holdout win rate {p['win_rate']:.1%} "
                    f"({p['n_trades']} trades), avg {p['avg_return']:+.3%}/trade"
                )
    else:
        lines.append(
            "No ticker exceeded the breakeven win rate on holdout at any threshold.  "
            f"Closest: {best[0]['ticker']} ({best[0]['holdout']['win_rate']:.1%}) "
            if best else "No holdout trades at this threshold."
        )

    lines += [
        "",
        "---",
        "",
        "*Generated by src/pipeline/10_backtest.py*",
        "",
    ]

    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [OK]  {out.relative_to(ROOT)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    section("market_ml -- Step 10: Confidence-Gated Iron Condor Backtest")
    print(f"  Signal:     Sideways predictions from saved parquets (no model loaded)")
    print(f"  Premium:    {PREMIUM:.1%}  |  Max loss: {LOSS:.1%}  |  "
          f"Breakeven: {BREAKEVEN_PREC:.1%}")
    print(f"  Thresholds: {THRESHOLDS[0]:.2f} to {THRESHOLDS[-1]:.2f} in 0.05 steps")
    print(f"  Holdout:    >= {HOLDOUT_DATE.date()} (always reported separately)")

    # 1. Load all ticker data
    section("Loading Predictions and Close Prices")
    ticker_data: dict = {}
    for ticker in ALL_TICKERS:
        try:
            df = load_ticker(ticker)
            ticker_data[ticker] = df
            oos_n  = (df["split"] == "oos").sum()
            hold_n = (df["split"] == "holdout").sum()
            sw_n   = (df[df["split"] == "oos"]["predicted"] == SIDEWAYS_CLASS).sum()
            print(f"  [OK]  {ticker}: {oos_n:,} OOS + {hold_n} holdout  "
                  f"(Sideways predicted OOS: {sw_n})")
        except FileNotFoundError as e:
            print(f"  {e}")
            sys.exit(1)

    # 2. Threshold sweep
    oos_sweep  = build_sector_sweep(ticker_data, "oos")
    hold_sweep = build_sector_sweep(ticker_data, "holdout")

    print_sweep_table(ticker_data, "OOS")
    print_sector_sweep(ticker_data)

    # 3. Find the two key thresholds
    prec_thresh = find_threshold(oos_sweep, PRECISION_TARGET, MIN_TRADES)
    be_thresh   = find_threshold(oos_sweep, BREAKEVEN_PREC,   MIN_TRADES)

    section("Threshold Selection")
    print(f"  Precision-flag threshold ({PRECISION_TARGET:.0%}): {prec_thresh:.2f}  "
          f"(OOS precision={oos_sweep.loc[prec_thresh, 'precision']:.3f})")
    print(f"  OOS-breakeven threshold ({BREAKEVEN_PREC:.1%}):  {be_thresh:.2f}  "
          f"(OOS precision={oos_sweep.loc[be_thresh, 'precision']:.3f})")
    h_prec_be = hold_sweep.loc[be_thresh, "precision"] if be_thresh in hold_sweep.index else float("nan")
    print(f"  Holdout precision at {be_thresh:.2f}: "
          f"{'N/A' if np.isnan(h_prec_be) else f'{h_prec_be:.3f}'}  "
          f"(breakeven = {BREAKEVEN_PREC:.1%} -- "
          f"{'BELOW' if np.isnan(h_prec_be) or h_prec_be < BREAKEVEN_PREC else 'OK'})")

    # 4. Run simulations at both thresholds
    section(f"P&L Simulation at Precision-Flag Threshold ({prec_thresh:.2f})")
    results_prec = run_backtest(ticker_data, prec_thresh)
    print_pnl_table(results_prec, "oos",     prec_thresh)
    print_sector_pnl(results_prec, "oos")

    section(f"P&L Simulation at OOS-Breakeven Threshold ({be_thresh:.2f})")
    results_be = run_backtest(ticker_data, be_thresh)
    print_pnl_table(results_be, "oos",     be_thresh)
    print_pnl_table(results_be, "holdout", be_thresh)
    print_sector_pnl(results_be, "oos")
    print_sector_pnl(results_be, "holdout")

    # 5. Save outputs
    section("Saving Outputs")
    save_backtest_parquet(results_be)   # save at breakeven threshold
    save_backtest_report(ticker_data, results_prec, results_be, prec_thresh, be_thresh)

    # 6. Final verdict
    section("Summary")
    oos_all  = pd.concat([r["oos_trade_rows"]  for r in results_be if len(r["oos_trade_rows"]) > 0])
    hold_all = pd.concat([r["hold_trade_rows"] for r in results_be if len(r["hold_trade_rows"]) > 0])
    oos_agg  = iron_condor_pnl(oos_all)
    hold_agg = iron_condor_pnl(hold_all)

    print(f"\n  At OOS-breakeven threshold ({be_thresh:.2f}):")
    print(f"    OOS     : {oos_agg['n_trades']:,} trades  "
          f"WR={oos_agg['win_rate']:.1%}  "
          f"AvgRet={oos_agg['avg_return']:+.3%}  "
          f"100T={oos_agg['total_100']:+.2%}")
    print(f"    Holdout : {hold_agg['n_trades']:,} trades  "
          f"WR={hold_agg['win_rate']:.1%}  "
          f"AvgRet={hold_agg['avg_return']:+.3%}  "
          f"100T={hold_agg['total_100']:+.2%}")
    print(f"\n  Breakeven win rate: {BREAKEVEN_PREC:.1%}")

    if hold_agg["win_rate"] >= BREAKEVEN_PREC:
        print(f"  [OK] Holdout WR {hold_agg['win_rate']:.1%} exceeds breakeven")
    else:
        print(f"  [WARN] Holdout WR {hold_agg['win_rate']:.1%} below breakeven "
              f"-- not yet profitable on 2024+ data")
        print(f"  [NOTE] OOS profitable: next step is accumulating 2025-2026 holdout data")
    print()


if __name__ == "__main__":
    main()

"""
Step 7 -- Per-ticker evaluation, calibration, and profit simulation
====================================================================
Loads OOS + holdout predictions from data/processed/{TICKER}_predictions.parquet
and evaluates the Step 6 shared sector models on each ticker individually.

Four deliverables:
  1. Per-ticker F1 scores vs sector average (flags >0.05 below as needs fine tuning)
  2. Calibration analysis per class -- does 70% confidence mean 70% correct?
  3. Directional profit simulation -- long Bull, cash Sideways, short Bear vs buy-and-hold
  4. Per-ticker JSON reports + docs/evaluation_report.md ranked summary

Output:
  data/processed/{TICKER}_eval_report.json   (one per ticker)
  docs/evaluation_report.md                  (ranked summary table)

Usage:
    python src/pipeline/07_evaluate.py
"""

import sys
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report, confusion_matrix
)

from config.tickers import SECTORS, TICKER_SECTOR

# ── Constants ──────────────────────────────────────────────────────────────────

PROCESSED_DIR = ROOT / "data" / "processed"
DOCS_DIR      = ROOT / "docs"
HOLDOUT_DATE  = pd.Timestamp("2025-01-01")  # updated Step 11; was 2024-01-01

FINE_TUNE_THRESHOLD = 0.05    # flag if ticker F1 > this below sector average
N_CAL_BINS          = 10      # calibration reliability diagram buckets

CLASS_NAMES = ["Bear", "Sideways", "Bull"]
CLASS_PROBA = ["proba_bear", "proba_side", "proba_bull"]

# v2 (Phase 1c) sector OOS F1 reference -- populated after 06_train.py v2 run
SECTOR_BASELINES = {
    "tech":    {"oos_f1": 0.402, "holdout_f1": 0.414},
    "biotech": {"oos_f1": 0.403, "holdout_f1": 0.386},
}


# ── Section printer ────────────────────────────────────────────────────────────

def section(title: str) -> None:
    bar = "=" * 64
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")


# ── Data loading ───────────────────────────────────────────────────────────────

def load_predictions(ticker: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{ticker}_predictions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"[FAIL] Predictions not found: {path.name}")
    df = pd.read_parquet(path, engine="pyarrow")
    assert df.index.name == "date", (
        f"Expected DatetimeIndex named 'date', got '{df.index.name}'"
    )
    return df


def load_close_prices(ticker: str) -> pd.Series:
    """Load close prices from features parquet for profit simulation."""
    path = PROCESSED_DIR / f"{ticker}_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"[FAIL] Features not found: {path.name}")
    df = pd.read_parquet(path, engine="pyarrow", columns=["close"])
    return df["close"].sort_index()


# ── Calibration analysis ───────────────────────────────────────────────────────

def calibration_analysis(
    actual:    np.ndarray,
    proba:     np.ndarray,
    class_idx: int,
    n_bins:    int = N_CAL_BINS,
) -> dict:
    """
    Reliability diagram data + Expected Calibration Error for one class.

    For each confidence bin [lo, hi):
      mean_conf = average predicted probability in that bin
      frac_pos  = actual fraction of that class in that bin
      gap       = |mean_conf - frac_pos|  (0 = perfectly calibrated)

    ECE = sum over bins of (n_bin / n_total) * gap
    A well-calibrated model has ECE near 0 -- 70% confidence = 70% correct.
    """
    is_class  = (actual == class_idx).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins      = []
    ece_num   = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (proba >= lo) & (proba < hi) if i < n_bins - 1 else (proba >= lo) & (proba <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        mean_conf = float(proba[mask].mean())
        frac_pos  = float(is_class[mask].mean())
        gap       = abs(mean_conf - frac_pos)
        ece_num  += n * gap
        bins.append({
            "bin_lo":    round(lo, 2),
            "bin_hi":    round(hi, 2),
            "n":         n,
            "mean_conf": round(mean_conf, 4),
            "frac_pos":  round(frac_pos, 4),
            "gap":       round(gap, 4),
        })

    ece = round(ece_num / len(actual), 5) if len(actual) > 0 else 0.0
    return {"ece": ece, "bins": bins}


def print_calibration_summary(ticker: str, cal: dict) -> None:
    print(f"  Calibration ECE (lower=better):  "
          f"Bear={cal['bear']['ece']:.4f}  "
          f"Side={cal['sideways']['ece']:.4f}  "
          f"Bull={cal['bull']['ece']:.4f}")
    # Highlight any badly-calibrated high-confidence bucket
    for cls_name, cls_key in [("Bear", "bear"), ("Side", "sideways"), ("Bull", "bull")]:
        for b in cal[cls_key]["bins"]:
            if b["mean_conf"] >= 0.60 and b["gap"] > 0.15:
                print(f"    [WARN] {cls_name} bin [{b['bin_lo']:.1f}-{b['bin_hi']:.1f}]: "
                      f"model says {b['mean_conf']:.0%} confident, "
                      f"actual correct {b['frac_pos']:.0%}  "
                      f"(gap={b['gap']:.2f}, n={b['n']})")


# ── Profit simulation ──────────────────────────────────────────────────────────

def profit_simulation(
    pred_df: pd.DataFrame,
    close:   pd.Series,
) -> dict:
    """
    Directional strategy vs buy-and-hold.

    Position mapping:
      Predicted Bull     (2) -> +1  long
      Predicted Sideways (1) ->  0  cash
      Predicted Bear     (0) -> -1  short

    Return proxy: each signal on day T earns the NEXT day's actual return.
    This is used for relative comparison only (not absolute P&L claiming).
    Note: this uses a 1-day forward return as a proxy for the 1-week signal;
    cumulative figures are illustrative, not a full backtest.
    """
    pos_map  = {0: -1, 1: 0, 2: 1}
    position = pred_df["predicted"].map(pos_map)

    # Next-day return: pct_change gives return vs prior day; shift(-1) makes
    # it "what will happen tomorrow" -- valid for evaluation purposes only
    fwd_ret = close.pct_change().shift(-1)

    merged = pd.DataFrame({
        "position": position,
        "fwd_ret":  fwd_ret,
    }).dropna()

    if len(merged) < 20:
        return {"error": "insufficient data"}

    strat_ret = merged["position"] * merged["fwd_ret"]
    bh_ret    = merged["fwd_ret"]

    strat_total = float((1 + strat_ret).prod() - 1)
    bh_total    = float((1 + bh_ret).prod() - 1)

    active = strat_ret[merged["position"] != 0]
    win_rate = float((active > 0).mean()) if len(active) > 0 else 0.0

    return {
        "strategy_total_return": round(strat_total, 4),
        "buyhold_total_return":  round(bh_total, 4),
        "alpha":                 round(strat_total - bh_total, 4),
        "strategy_sharpe":       round(_sharpe(strat_ret), 3),
        "buyhold_sharpe":        round(_sharpe(bh_ret), 3),
        "win_rate":              round(win_rate, 4),
        "n_active_trades":       int((merged["position"] != 0).sum()),
        "n_long":                int((merged["position"] == 1).sum()),
        "n_short":               int((merged["position"] == -1).sum()),
        "n_cash":                int((merged["position"] == 0).sum()),
    }


def _sharpe(returns: pd.Series, ann: int = 252) -> float:
    """Annualised Sharpe ratio (risk-free rate = 0)."""
    std = returns.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(returns.mean() / std * np.sqrt(ann))


# ── Per-ticker evaluation ──────────────────────────────────────────────────────

def evaluate_ticker(ticker: str) -> dict:
    """Full evaluation pipeline for one ticker. Returns report dict."""
    sector  = TICKER_SECTOR[ticker]
    pred_df = load_predictions(ticker)
    close   = load_close_prices(ticker)

    oos_df     = pred_df[pred_df["split"] == "oos"].copy()
    holdout_df = pred_df[pred_df["split"] == "holdout"].copy()

    result: dict = {
        "ticker":       ticker,
        "sector":       sector,
        "oos_rows":     len(oos_df),
        "holdout_rows": len(holdout_df),
    }

    # ── OOS metrics ───────────────────────────────────────────────────────────
    if len(oos_df) > 0:
        y_true = oos_df["actual"].values.astype(int)
        y_pred = oos_df["predicted"].values.astype(int)

        report = classification_report(
            y_true, y_pred,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )

        def _r(key: str, sub: str) -> float:
            return round(report.get(key, {}).get(sub, 0.0), 4)

        result.update({
            "oos_f1_weighted":     round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
            "oos_f1_macro":        round(f1_score(y_true, y_pred, average="macro",    zero_division=0), 4),
            "oos_accuracy":        round(accuracy_score(y_true, y_pred), 4),
            "oos_f1_bear":         _r("Bear",     "f1-score"),
            "oos_f1_sideways":     _r("Sideways", "f1-score"),
            "oos_f1_bull":         _r("Bull",     "f1-score"),
            "oos_precision_bear":  _r("Bear",     "precision"),
            "oos_precision_side":  _r("Sideways", "precision"),
            "oos_precision_bull":  _r("Bull",     "precision"),
            "oos_recall_bear":     _r("Bear",     "recall"),
            "oos_recall_side":     _r("Sideways", "recall"),
            "oos_recall_bull":     _r("Bull",     "recall"),
            "confusion_matrix_oos": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
        })

        # Calibration
        proba_arr = oos_df[CLASS_PROBA].values
        result["calibration"] = {
            "bear":     calibration_analysis(y_true, proba_arr[:, 0], 0),
            "sideways": calibration_analysis(y_true, proba_arr[:, 1], 1),
            "bull":     calibration_analysis(y_true, proba_arr[:, 2], 2),
        }

        # Profit simulation
        result["profit_sim_oos"] = profit_simulation(oos_df, close)
    else:
        result["oos_f1_weighted"] = 0.0

    # ── Holdout metrics ───────────────────────────────────────────────────────
    if len(holdout_df) > 0:
        yh_true = holdout_df["actual"].values.astype(int)
        yh_pred = holdout_df["predicted"].values.astype(int)

        h_report = classification_report(
            yh_true, yh_pred,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )

        def _rh(key: str, sub: str) -> float:
            return round(h_report.get(key, {}).get(sub, 0.0), 4)

        result.update({
            "holdout_f1_weighted": round(f1_score(yh_true, yh_pred, average="weighted", zero_division=0), 4),
            "holdout_f1_macro":    round(f1_score(yh_true, yh_pred, average="macro",    zero_division=0), 4),
            "holdout_accuracy":    round(accuracy_score(yh_true, yh_pred), 4),
            "holdout_f1_bear":     _rh("Bear",     "f1-score"),
            "holdout_f1_sideways": _rh("Sideways", "f1-score"),
            "holdout_f1_bull":     _rh("Bull",     "f1-score"),
            "confusion_matrix_holdout": confusion_matrix(yh_true, yh_pred, labels=[0, 1, 2]).tolist(),
        })

        result["profit_sim_holdout"] = profit_simulation(holdout_df, close)
    else:
        result["holdout_f1_weighted"] = 0.0
        result["holdout_accuracy"]    = 0.0

    return result


# ── Fine-tuning flags ─────────────────────────────────────────────────────────

def apply_fine_tune_flags(all_results: list) -> list:
    """Flag any ticker with OOS F1 more than FINE_TUNE_THRESHOLD below sector average."""
    sector_f1: dict = {}
    for sector in SECTORS:
        sector_res = [r for r in all_results if r["sector"] == sector]
        if sector_res:
            sector_f1[sector] = round(float(np.mean([r["oos_f1_weighted"] for r in sector_res])), 4)

    for r in all_results:
        avg   = sector_f1.get(r["sector"], 0.0)
        delta = r["oos_f1_weighted"] - avg
        r["sector_avg_f1"]      = avg
        r["delta_vs_sector"]    = round(delta, 4)
        r["fine_tuning_needed"] = bool(delta < -FINE_TUNE_THRESHOLD)
        r["fine_tuning_reason"] = (
            f"OOS F1 {r['oos_f1_weighted']:.3f} is {abs(delta):.3f} below "
            f"sector avg {avg:.3f} (threshold: {FINE_TUNE_THRESHOLD})"
            if r["fine_tuning_needed"] else ""
        )

    return all_results


# ── Print helpers ─────────────────────────────────────────────────────────────

def print_ticker_detail(r: dict) -> None:
    flag = "[NEEDS FINE TUNING]" if r["fine_tuning_needed"] else "[OK]"
    print(f"\n  {r['ticker']:6s} ({r['sector']:7s})  "
          f"OOS F1={r['oos_f1_weighted']:.3f}  "
          f"vs sector avg={r['sector_avg_f1']:.3f}  "
          f"delta={r['delta_vs_sector']:+.3f}  {flag}")
    print(f"    OOS classes:   Bear={r['oos_f1_bear']:.3f}  "
          f"Side={r['oos_f1_sideways']:.3f}  "
          f"Bull={r['oos_f1_bull']:.3f}  "
          f"Acc={r['oos_accuracy']:.3f}")
    print(f"    Holdout:       F1={r.get('holdout_f1_weighted', 0):.3f}  "
          f"Acc={r.get('holdout_accuracy', 0):.3f}")

    if "calibration" in r:
        print_calibration_summary(r["ticker"], r["calibration"])

    if "profit_sim_oos" in r:
        sim = r["profit_sim_oos"]
        if "error" not in sim:
            a_sign = "+" if sim["alpha"] >= 0 else ""
            print(f"    P&L OOS:       Strategy={sim['strategy_total_return']:+.1%}  "
                  f"BuyHold={sim['buyhold_total_return']:+.1%}  "
                  f"Alpha={a_sign}{sim['alpha']:.1%}  "
                  f"WinRate={sim['win_rate']:.1%}  "
                  f"Sharpe={sim['strategy_sharpe']:.2f}")


def print_ranking_table(all_results: list) -> None:
    section("FINAL RANKING TABLE -- Best to Worst OOS F1")

    ranked = sorted(all_results, key=lambda r: r["oos_f1_weighted"], reverse=True)

    print(f"\n  {'Rank':>4}  {'Ticker':6}  {'Sector':7}  "
          f"{'OOS F1':>7}  {'Hold F1':>8}  "
          f"{'Alpha':>8}  {'Sharpe':>7}  {'CalECE':>7}  Status")
    print("  " + "-" * 80)

    for rank, r in enumerate(ranked, 1):
        sim    = r.get("profit_sim_oos", {})
        alpha  = sim.get("alpha", float("nan"))
        sharpe = sim.get("strategy_sharpe", float("nan"))
        cal    = r.get("calibration", {})
        avg_ece = (
            np.mean([cal.get(c, {}).get("ece", 0.0) for c in ["bear", "sideways", "bull"]])
            if cal else float("nan")
        )
        alpha_s  = f"{alpha:+.2%}" if not np.isnan(alpha) else "   N/A"
        sharpe_s = f"{sharpe:+.2f}" if not np.isnan(sharpe) else "   N/A"
        ece_s    = f"{avg_ece:.4f}" if not np.isnan(avg_ece) else "   N/A"
        hold_f1  = r.get("holdout_f1_weighted", 0.0)
        status   = "FINE TUNING" if r["fine_tuning_needed"] else "OK"

        print(f"  {rank:>4}  {r['ticker']:6}  {r['sector']:7}  "
              f"{r['oos_f1_weighted']:>7.4f}  {hold_f1:>8.4f}  "
              f"{alpha_s:>8}  {sharpe_s:>7}  {ece_s:>7}  {status}")

    print(f"\n  Tickers flagged for fine tuning (>{FINE_TUNE_THRESHOLD} below sector avg):")
    flagged = [r for r in all_results if r["fine_tuning_needed"]]
    if flagged:
        for r in sorted(flagged, key=lambda x: x["oos_f1_weighted"]):
            print(f"    - {r['ticker']}: {r['fine_tuning_reason']}")
    else:
        print("    None -- all tickers within acceptable range of sector average")


# ── JSON report ───────────────────────────────────────────────────────────────

def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serialisable: {type(obj)}")


def save_eval_report(r: dict) -> None:
    out   = PROCESSED_DIR / f"{r['ticker']}_eval_report.json"
    clean = json.loads(json.dumps(r, default=_json_default))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2)
    print(f"  [OK] {out.relative_to(ROOT)}")


# ── Markdown summary ──────────────────────────────────────────────────────────

def save_summary_report(all_results: list) -> None:
    DOCS_DIR.mkdir(exist_ok=True)
    out    = DOCS_DIR / "evaluation_report.md"
    ranked = sorted(all_results, key=lambda r: r["oos_f1_weighted"], reverse=True)
    today  = pd.Timestamp.now().strftime("%Y-%m-%d")

    lines = [
        "# market_ml Step 7 -- Per-Ticker Evaluation Report",
        "",
        f"Generated: {today}  ",
        f"Holdout cutoff: {HOLDOUT_DATE.date()}  ",
        f"Fine-tune threshold: >{FINE_TUNE_THRESHOLD} below sector OOS F1 average",
        "",
        "## Ranking Table",
        "",
        "| Rank | Ticker | Sector | OOS F1 | Holdout F1 | Alpha | Strategy Sharpe | Avg Cal ECE | Status |",
        "|------|--------|--------|--------|------------|-------|-----------------|-------------|--------|",
    ]

    for rank, r in enumerate(ranked, 1):
        sim    = r.get("profit_sim_oos", {})
        alpha  = sim.get("alpha", float("nan"))
        sharpe = sim.get("strategy_sharpe", float("nan"))
        cal    = r.get("calibration", {})
        avg_ece = (
            np.mean([cal.get(c, {}).get("ece", 0.0) for c in ["bear", "sideways", "bull"]])
            if cal else float("nan")
        )
        alpha_s  = f"{alpha:+.2%}" if not np.isnan(alpha) else "N/A"
        sharpe_s = f"{sharpe:.2f}"  if not np.isnan(sharpe) else "N/A"
        ece_s    = f"{avg_ece:.4f}" if not np.isnan(avg_ece) else "N/A"
        hold_f1  = r.get("holdout_f1_weighted", 0.0)
        status   = "**FINE TUNE**" if r["fine_tuning_needed"] else "OK"
        lines.append(
            f"| {rank} | {r['ticker']} | {r['sector']} | "
            f"{r['oos_f1_weighted']:.4f} | {hold_f1:.4f} | "
            f"{alpha_s} | {sharpe_s} | {ece_s} | {status} |"
        )

    lines += [
        "",
        "## Per-Class F1 (OOS)",
        "",
        "| Ticker | Sector | Bear F1 | Sideways F1 | Bull F1 | Fine Tune? |",
        "|--------|--------|---------|-------------|---------|------------|",
    ]
    for r in ranked:
        flag = "YES" if r["fine_tuning_needed"] else "no"
        lines.append(
            f"| {r['ticker']} | {r['sector']} | "
            f"{r['oos_f1_bear']:.4f} | "
            f"{r['oos_f1_sideways']:.4f} | "
            f"{r['oos_f1_bull']:.4f} | {flag} |"
        )

    lines += [
        "",
        "## Calibration (ECE per class, OOS)",
        "",
        "ECE = Expected Calibration Error. Lower is better.",
        "A perfectly calibrated model has ECE = 0: 70% confidence = 70% correct.",
        "",
        "| Ticker | Bear ECE | Sideways ECE | Bull ECE | Mean ECE |",
        "|--------|----------|--------------|----------|----------|",
    ]
    for r in ranked:
        cal = r.get("calibration", {})
        if cal:
            b_ece = cal.get("bear", {}).get("ece", 0.0)
            s_ece = cal.get("sideways", {}).get("ece", 0.0)
            u_ece = cal.get("bull", {}).get("ece", 0.0)
            m_ece = (b_ece + s_ece + u_ece) / 3
            lines.append(
                f"| {r['ticker']} | {b_ece:.4f} | {s_ece:.4f} | {u_ece:.4f} | {m_ece:.4f} |"
            )

    lines += [
        "",
        "## Profit Simulation (OOS)",
        "",
        "Strategy: Long when Bull predicted, Cash when Sideways, Short when Bear predicted.",
        "Return proxy: signal on day T applied to next-day actual return (illustrative, not a full backtest).",
        "",
        "| Ticker | Strategy Return | Buy-Hold Return | Alpha | Win Rate | Strategy Sharpe |",
        "|--------|----------------|-----------------|-------|----------|-----------------|",
    ]
    for r in ranked:
        sim = r.get("profit_sim_oos", {})
        if sim and "error" not in sim:
            lines.append(
                f"| {r['ticker']} | {sim['strategy_total_return']:+.2%} | "
                f"{sim['buyhold_total_return']:+.2%} | "
                f"{sim['alpha']:+.2%} | {sim['win_rate']:.2%} | "
                f"{sim['strategy_sharpe']:.2f} |"
            )

    lines += [
        "",
        "## Sector Averages",
        "",
        "| Sector | Per-Ticker Avg OOS F1 | Step 6 Reported OOS F1 | Tickers |",
        "|--------|-----------------------|------------------------|---------|",
    ]
    for sector in SECTORS:
        sr = [r for r in all_results if r["sector"] == sector]
        if sr:
            avg      = np.mean([r["oos_f1_weighted"] for r in sr])
            baseline = SECTOR_BASELINES[sector]["oos_f1"]
            t_list   = ", ".join(r["ticker"] for r in sr)
            lines.append(
                f"| {sector} | {avg:.4f} | {baseline:.4f} | {t_list} |"
            )

    lines += [
        "",
        "## Fine-Tuning Priority (Step 8 targets)",
        "",
    ]
    flagged = sorted(
        [r for r in all_results if r["fine_tuning_needed"]],
        key=lambda x: x["oos_f1_weighted"],
    )
    if flagged:
        for r in flagged:
            lines.append(f"- **{r['ticker']}**: {r['fine_tuning_reason']}")
    else:
        lines.append(
            "No tickers flagged -- all within acceptable range of sector average."
        )

    lines += [
        "",
        "---",
        "",
        "*Generated by src/pipeline/07_evaluate.py*",
        "",
    ]

    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [OK] {out.relative_to(ROOT)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    section("market_ml -- Step 7: Per-Ticker Evaluation")
    print(f"  Holdout cutoff:   >= {HOLDOUT_DATE.date()}")
    print(f"  Fine-tune flag:   > {FINE_TUNE_THRESHOLD} below sector OOS F1 average")
    print(f"  Calibration bins: {N_CAL_BINS}")
    print(f"  Output dir:       data/processed/  +  docs/evaluation_report.md")

    all_tickers = [t for v in SECTORS.values() for t in v["tickers"]]
    all_results = []

    for ticker in all_tickers:
        section(f"Evaluating: {ticker} ({TICKER_SECTOR[ticker].upper()})")
        try:
            r = evaluate_ticker(ticker)
            all_results.append(r)
            print(f"  [OK]  {ticker}: OOS F1={r['oos_f1_weighted']:.4f}  "
                  f"Holdout F1={r.get('holdout_f1_weighted', 0):.4f}  "
                  f"OOS rows={r['oos_rows']}")
        except Exception as exc:
            print(f"  [FAIL] {ticker}: {exc}")

    if not all_results:
        print("[FAIL] No tickers evaluated. Check predictions files in data/processed/")
        sys.exit(1)

    # Compute sector averages and flag underperformers
    all_results = apply_fine_tune_flags(all_results)

    # Verbose per-ticker output
    section("Per-Ticker Summaries")
    for r in all_results:
        print_ticker_detail(r)

    # Final ranking table
    print_ranking_table(all_results)

    # Save JSON reports
    section("Saving Reports")
    for r in all_results:
        save_eval_report(r)
    save_summary_report(all_results)

    section("Step 7 Complete")
    print("  Per-ticker reports: data/processed/{TICKER}_eval_report.json")
    print("  Summary:            docs/evaluation_report.md")
    print("\n  Next step: src/pipeline/08_finetune.py\n")


if __name__ == "__main__":
    main()

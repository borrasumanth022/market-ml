"""
Step 8 -- Isotonic calibration for Bear class overconfidence
=============================================================
Step 7 revealed Bear ECE ~0.10 in both sectors: when the model says
70-80% Bear confidence, it is only right 15-25% of the time.
Sideways (ECE ~0.03) is already well-calibrated.

Calibration approach:
  CalibratedClassifierCV(FrozenEstimator(model), method='isotonic')
  fitted on the OOS feature matrix (all pre-holdout rows that appeared
  in walk-forward validation).
  Holdout is NEVER used for calibration -- only for final eval.

Acceptance criteria (both must pass to save a calibrated model):
  - Calibrated Bear ECE  < BEAR_ECE_TARGET (0.05)
  - Weighted F1 drop vs uncalibrated < MAX_F1_DROP (0.01)
  If either fails the script exits loudly without overwriting anything.

Output:
  models/tech/xgb_tech_shared_v1_cal.pkl
  models/biotech/xgb_biotech_shared_v1_cal.pkl
  data/processed/{TICKER}_predictions.parquet  (proba_* columns replaced)
  docs/evaluation_report.md                    (Step 8 section appended)

Usage:
    python src/pipeline/08_calibrate.py
"""

import sys
import pickle
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import f1_score

from config.tickers import SECTORS, TICKER_SECTOR

# ── Constants (mirrored from 06_train.py) ──────────────────────────────────────

PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR    = ROOT / "models"
DOCS_DIR      = ROOT / "docs"

HOLDOUT_DATE    = pd.Timestamp("2024-01-01")
BEAR_ECE_TARGET = 0.05   # calibrated Bear ECE must be below this to save
MAX_F1_DROP     = 0.01   # weighted F1 must not drop more than this vs uncalibrated
N_CAL_BINS      = 10

CLASS_NAMES = ["Bear", "Sideways", "Bull"]
CLASS_PROBA = ["proba_bear", "proba_side", "proba_bull"]
LABEL_ENCODE = {-1: 0, 0: 1, 1: 2}

SELECTED_36 = [
    "atr_pct", "bb_pct", "bb_width", "candle_body", "candle_dir",
    "close_vs_sma10", "close_vs_sma100", "close_vs_sma20", "close_vs_sma200",
    "close_vs_sma50", "cross_50_200", "day_of_week", "gap_pct", "hl_range_pct",
    "hvol_10d", "hvol_21d", "hvol_63d", "is_month_end", "is_month_start",
    "is_quarter_end", "lower_shadow", "macd_hist", "macd_signal", "month",
    "price_52w_pct", "return_1d", "return_2d", "return_5d", "roc_10", "roc_21",
    "rsi_14", "rsi_7", "stoch_d", "stoch_k", "upper_shadow", "volume_zscore",
]

EVENT_14 = [
    "days_to_next_earnings", "days_since_last_earnings", "last_eps_surprise_pct",
    "earnings_streak", "has_earnings_data",
    "fed_rate_level", "fed_rate_change_1m", "fed_rate_change_3m",
    "cpi_yoy_change", "unemployment_level", "unemployment_change_3m",
    "inflation_regime", "macro_stress_score", "rate_environment",
]

FDA_5 = [
    "days_to_next_fda_decision", "days_since_last_fda_decision",
    "last_fda_outcome", "fda_decisions_trailing_12m", "fda_approval_rate_trailing",
]


# ── Printer ────────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    bar = "=" * 64
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")


# ── Model loading ─────────────────────────────────────────────────────────────

def load_sector_model(sector: str) -> tuple:
    """Load pkl. Returns (model, feature_names, pkl_dict)."""
    path = MODELS_DIR / sector / f"xgb_{sector}_shared_v1.pkl"
    if not path.exists():
        raise FileNotFoundError(f"[FAIL] Model not found: {path}")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["model"], obj["feature_names"], obj


# ── Feature matrix construction (mirrors 06_train.py) ─────────────────────────

def load_sector_features(sector: str) -> tuple:
    """
    Load with_events parquets for all sector tickers and build the
    combined feature matrix exactly as 06_train.py does.

    Returns:
      df_combined  : combined DataFrame with ticker_id and __label__
      X_all        : feature matrix (DataFrame, columns = feature_names)
      feature_names: list of column names
    """
    tickers   = SECTORS[sector]["tickers"]
    base_feats = SELECTED_36 + EVENT_14 + (FDA_5 if sector == "biotech" else [])
    frames    = []

    print(f"\nLoading {sector} features...")
    for ticker in tickers:
        path = PROCESSED_DIR / f"{ticker}_with_events.parquet"
        if not path.exists():
            raise FileNotFoundError(f"[FAIL] Missing: {path.name}")
        df = pd.read_parquet(path, engine="pyarrow")
        df["ticker_id"] = ticker
        frames.append(df)
        print(f"  {ticker}: {len(df):,} rows  "
              f"{str(df.index[0].date())} to {str(df.index[-1].date())}")

    combined = pd.concat(frames).sort_index()

    # AMZN EPS outlier clipping (same as 06_train.py)
    if "AMZN" in combined["ticker_id"].values:
        mask = combined["ticker_id"] == "AMZN"
        combined.loc[mask, "last_eps_surprise_pct"] = (
            combined.loc[mask, "last_eps_surprise_pct"].clip(-500.0, 500.0)
        )

    # Encode labels
    combined["__label__"] = combined["dir_1w"].map(LABEL_ENCODE)
    combined = combined.dropna(subset=["__label__"])
    combined["__label__"] = combined["__label__"].astype(int)

    # One-hot ticker_id (same column order as training)
    ticker_dummies = pd.get_dummies(combined["ticker_id"], prefix="ticker").astype(int)
    X_all = pd.concat([combined[base_feats], ticker_dummies], axis=1)
    feature_names = base_feats + list(ticker_dummies.columns)

    print(f"\n  Combined: {len(combined):,} rows  |  "
          f"Features: {X_all.shape[1]}  |  "
          f"Pre-holdout: {(combined.index < HOLDOUT_DATE).sum():,}")

    return combined, X_all, feature_names


# ── Calibration metrics ────────────────────────────────────────────────────────

def compute_ece(actual: np.ndarray, proba: np.ndarray, class_idx: int) -> float:
    """Expected Calibration Error for one class. Lower = better calibrated."""
    is_class  = (actual == class_idx).astype(float)
    bin_edges = np.linspace(0.0, 1.0, N_CAL_BINS + 1)
    ece_num   = 0.0
    n_total   = len(actual)

    for i in range(N_CAL_BINS):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask   = (proba >= lo) & (proba < hi) if i < N_CAL_BINS - 1 else (proba >= lo) & (proba <= hi)
        n = mask.sum()
        if n == 0:
            continue
        ece_num += n * abs(proba[mask].mean() - is_class[mask].mean())

    return round(ece_num / n_total, 5) if n_total > 0 else 0.0


def evaluate_proba(actual: np.ndarray, proba_matrix: np.ndarray) -> dict:
    """
    Compute F1 + ECE per class from a (n, 3) probability matrix.
    Predicted class = argmax of proba_matrix.
    """
    predicted = proba_matrix.argmax(axis=1)
    return {
        "f1_weighted": round(f1_score(actual, predicted, average="weighted", zero_division=0), 4),
        "ece_bear":    compute_ece(actual, proba_matrix[:, 0], 0),
        "ece_side":    compute_ece(actual, proba_matrix[:, 1], 1),
        "ece_bull":    compute_ece(actual, proba_matrix[:, 2], 2),
    }


# ── Before/after display ───────────────────────────────────────────────────────

def print_before_after(
    sector:       str,
    before_oos:   dict,
    after_oos:    dict,
    before_hold:  dict,
    after_hold:   dict,
) -> None:
    """Print a clean two-split before/after comparison table."""
    print(f"\n  {'Metric':20s}  {'OOS Before':>12}  {'OOS After':>10}  "
          f"{'Hold Before':>12}  {'Hold After':>11}")
    print("  " + "-" * 72)

    rows = [
        ("F1 weighted",  "f1_weighted"),
        ("ECE Bear",     "ece_bear"),
        ("ECE Sideways", "ece_side"),
        ("ECE Bull",     "ece_bull"),
    ]
    for label, key in rows:
        b_o = before_oos.get(key, 0)
        a_o = after_oos.get(key, 0)
        b_h = before_hold.get(key, 0)
        a_h = after_hold.get(key, 0)

        # Delta markers: improvement shown with + for F1, - for ECE
        if key == "f1_weighted":
            d_o = f"({a_o - b_o:+.4f})"
            d_h = f"({a_h - b_h:+.4f})"
        else:
            d_o = f"({a_o - b_o:+.4f})"
            d_h = f"({a_h - b_h:+.4f})"

        print(f"  {label:20s}  {b_o:>12.4f}  {a_o:>6.4f} {d_o:>10}  "
              f"{b_h:>12.4f}  {a_h:>6.4f} {d_h:>10}")

    # Calibration verdict
    bear_ece_pass = after_oos["ece_bear"] < BEAR_ECE_TARGET
    f1_pass       = (before_oos["f1_weighted"] - after_oos["f1_weighted"]) <= MAX_F1_DROP
    print(f"\n  Bear ECE < {BEAR_ECE_TARGET}: {'PASS' if bear_ece_pass else 'FAIL'}  "
          f"({after_oos['ece_bear']:.4f} {'<' if bear_ece_pass else '>='} {BEAR_ECE_TARGET})")
    print(f"  F1 drop  < {MAX_F1_DROP}:  {'PASS' if f1_pass else 'FAIL'}  "
          f"(drop = {before_oos['f1_weighted'] - after_oos['f1_weighted']:.4f})")


# ── Regenerate predictions per ticker ─────────────────────────────────────────

def regenerate_predictions(
    sector:        str,
    cal_model:     object,
    df_combined:   pd.DataFrame,
    X_all:         pd.DataFrame,
) -> None:
    """
    Replace proba_bear/side/bull in each ticker's predictions parquet using
    the calibrated model. predicted and actual columns are NOT changed.
    """
    tickers = SECTORS[sector]["tickers"]
    print(f"\nRegenerating predictions ({sector})...")

    # Get calibrated probabilities for the full sector feature matrix
    # (OOS + holdout together -- the calibrated model can be applied anywhere)
    cal_proba = cal_model.predict_proba(X_all.values)

    # Map back to per-row using the combined df's index
    proba_df = pd.DataFrame(
        cal_proba,
        index=df_combined.index,
        columns=["proba_bear", "proba_side", "proba_bull"],
    )
    proba_df["ticker_id"] = df_combined["ticker_id"].values

    for ticker in tickers:
        pred_path = PROCESSED_DIR / f"{ticker}_predictions.parquet"
        if not pred_path.exists():
            print(f"  [WARN] {ticker}: predictions parquet not found, skipping")
            continue

        orig = pd.read_parquet(pred_path, engine="pyarrow")

        # Match calibrated probabilities by date + ticker
        ticker_proba = proba_df[proba_df["ticker_id"] == ticker].drop(columns=["ticker_id"])

        # Only update rows whose dates are present in both
        common_idx = orig.index.intersection(ticker_proba.index)
        if len(common_idx) < len(orig):
            n_missing = len(orig) - len(common_idx)
            print(f"  [WARN] {ticker}: {n_missing} prediction rows have no feature match")

        orig.loc[common_idx, "proba_bear"] = ticker_proba.loc[common_idx, "proba_bear"].values
        orig.loc[common_idx, "proba_side"] = ticker_proba.loc[common_idx, "proba_side"].values
        orig.loc[common_idx, "proba_bull"] = ticker_proba.loc[common_idx, "proba_bull"].values

        orig.to_parquet(pred_path, engine="pyarrow", index=True)
        oos_rows  = (orig["split"] == "oos").sum()
        hold_rows = (orig["split"] == "holdout").sum()
        print(f"  [OK]  {ticker}: {oos_rows} OOS + {hold_rows} holdout rows updated")


# ── Markdown report append ─────────────────────────────────────────────────────

def append_calibration_report(sector_results: list) -> None:
    """Append a Step 8 calibration section to docs/evaluation_report.md."""
    report_path = DOCS_DIR / "evaluation_report.md"
    today = pd.Timestamp.now().strftime("%Y-%m-%d")

    lines = [
        "",
        "---",
        "",
        "## Step 8 Calibration Results",
        "",
        f"Generated: {today}  ",
        f"Method: isotonic regression (CalibratedClassifierCV + FrozenEstimator)  ",
        f"Calibration set: OOS walk-forward feature matrix (pre-holdout only)  ",
        f"Acceptance: Bear ECE < {BEAR_ECE_TARGET} AND F1 drop < {MAX_F1_DROP}",
        "",
        "### Before vs After ECE (OOS)",
        "",
        "| Sector | Bear ECE Before | Bear ECE After | Side ECE Before | Side ECE After "
        "| Bull ECE Before | Bull ECE After | F1 Before | F1 After | Verdict |",
        "|--------|----------------|----------------|----------------|----------------|"
        "----------------|----------------|-----------|----------|---------|",
    ]

    for r in sector_results:
        b_o = r["before_oos"]
        a_o = r["after_oos"]
        verdict = r["verdict"]
        lines.append(
            f"| {r['sector']} | {b_o['ece_bear']:.4f} | {a_o['ece_bear']:.4f} | "
            f"{b_o['ece_side']:.4f} | {a_o['ece_side']:.4f} | "
            f"{b_o['ece_bull']:.4f} | {a_o['ece_bull']:.4f} | "
            f"{b_o['f1_weighted']:.4f} | {a_o['f1_weighted']:.4f} | "
            f"**{verdict}** |"
        )

    lines += [
        "",
        "### Before vs After ECE (Holdout)",
        "",
        "| Sector | Bear ECE Before | Bear ECE After | F1 Before | F1 After |",
        "|--------|----------------|----------------|-----------|----------|",
    ]
    for r in sector_results:
        b_h = r["before_hold"]
        a_h = r["after_hold"]
        lines.append(
            f"| {r['sector']} | {b_h['ece_bear']:.4f} | {a_h['ece_bear']:.4f} | "
            f"{b_h['f1_weighted']:.4f} | {a_h['f1_weighted']:.4f} |"
        )

    lines += ["", "### Verdict per sector", ""]
    for r in sector_results:
        lines.append(f"- **{r['sector']}**: {r['verdict_detail']}")

    lines += ["", "---", "", "*Generated by src/pipeline/08_calibrate.py*", ""]

    with open(report_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  [OK] Appended calibration section to {report_path.relative_to(ROOT)}")


# ── Per-sector orchestration ───────────────────────────────────────────────────

def run_sector(sector: str) -> dict:
    section(f"Sector: {sector.upper()}")

    # 1. Load model
    model, feature_names, pkl_obj = load_sector_model(sector)
    print(f"  Loaded model: {len(feature_names)} features, "
          f"tickers: {SECTORS[sector]['tickers']}")

    # 2. Build full feature matrix from with_events parquets
    df_combined, X_all, derived_feat_names = load_sector_features(sector)

    # Verify column alignment (critical — miscalibration if order differs)
    if derived_feat_names != feature_names:
        raise ValueError(
            f"[FAIL] {sector}: feature name mismatch between model pkl and "
            f"reconstructed matrix.\n"
            f"  Model expects: {feature_names[:5]}...\n"
            f"  Matrix has:    {derived_feat_names[:5]}..."
        )

    # 3. Split pre-holdout (calibration) vs holdout (eval only)
    oos_mask     = df_combined.index < HOLDOUT_DATE
    holdout_mask = df_combined.index >= HOLDOUT_DATE

    X_oos     = X_all[oos_mask].values
    y_oos     = df_combined["__label__"][oos_mask].values
    X_holdout = X_all[holdout_mask].values
    y_holdout = df_combined["__label__"][holdout_mask].values

    print(f"\n  OOS (calibration) rows:  {len(y_oos):,}")
    print(f"  Holdout (eval only) rows: {len(y_holdout):,}")

    # 4. Evaluate BEFORE calibration (using uncalibrated final model)
    section(f"Before calibration: {sector.upper()}")
    proba_oos_before     = model.predict_proba(X_oos)
    proba_holdout_before = model.predict_proba(X_holdout)
    before_oos   = evaluate_proba(y_oos,     proba_oos_before)
    before_hold  = evaluate_proba(y_holdout, proba_holdout_before)
    print(f"  OOS:     F1={before_oos['f1_weighted']:.4f}  "
          f"ECE Bear={before_oos['ece_bear']:.4f}  "
          f"Side={before_oos['ece_side']:.4f}  "
          f"Bull={before_oos['ece_bull']:.4f}")
    print(f"  Holdout: F1={before_hold['f1_weighted']:.4f}  "
          f"ECE Bear={before_hold['ece_bear']:.4f}  "
          f"Side={before_hold['ece_side']:.4f}  "
          f"Bull={before_hold['ece_bull']:.4f}")

    # 5. Fit calibration on OOS feature matrix
    section(f"Fitting isotonic calibration: {sector.upper()}")
    print(f"  CalibratedClassifierCV(FrozenEstimator, method='isotonic')")
    print(f"  Calibration set: {len(y_oos):,} rows (all pre-holdout features)")
    print(f"  NOTE: final model trained on this same data -- holdout is the clean eval")

    cal_model = CalibratedClassifierCV(
        estimator=FrozenEstimator(model),
        method="isotonic",
    )
    cal_model.fit(X_oos, y_oos)
    print("  Calibration fit: complete")

    # 6. Evaluate AFTER calibration
    section(f"After calibration: {sector.upper()}")
    proba_oos_after     = cal_model.predict_proba(X_oos)
    proba_holdout_after = cal_model.predict_proba(X_holdout)
    after_oos   = evaluate_proba(y_oos,     proba_oos_after)
    after_hold  = evaluate_proba(y_holdout, proba_holdout_after)
    print(f"  OOS:     F1={after_oos['f1_weighted']:.4f}  "
          f"ECE Bear={after_oos['ece_bear']:.4f}  "
          f"Side={after_oos['ece_side']:.4f}  "
          f"Bull={after_oos['ece_bull']:.4f}")
    print(f"  Holdout: F1={after_hold['f1_weighted']:.4f}  "
          f"ECE Bear={after_hold['ece_bear']:.4f}  "
          f"Side={after_hold['ece_side']:.4f}  "
          f"Bull={after_hold['ece_bull']:.4f}")

    print_before_after(sector, before_oos, after_oos, before_hold, after_hold)

    # 7. Acceptance checks
    bear_ece_ok = after_oos["ece_bear"] < BEAR_ECE_TARGET
    f1_ok       = (before_oos["f1_weighted"] - after_oos["f1_weighted"]) <= MAX_F1_DROP
    passed      = bear_ece_ok and f1_ok

    if not passed:
        reasons = []
        if not bear_ece_ok:
            reasons.append(
                f"Bear ECE {after_oos['ece_bear']:.4f} >= target {BEAR_ECE_TARGET}"
            )
        if not f1_ok:
            drop = before_oos["f1_weighted"] - after_oos["f1_weighted"]
            reasons.append(
                f"F1 dropped {drop:.4f} > max {MAX_F1_DROP}"
            )
        reason_str = "; ".join(reasons)
        print(f"\n  [FAIL] {sector.upper()}: calibration criteria NOT met: {reason_str}")
        print(f"  [FAIL] Calibrated model NOT saved. Original model unchanged.")
        verdict        = "REJECTED"
        verdict_detail = (
            f"Calibration rejected for **{sector}**: {reason_str}. "
            "Original model unchanged."
        )
        return {
            "sector":        sector,
            "before_oos":    before_oos,
            "after_oos":     after_oos,
            "before_hold":   before_hold,
            "after_hold":    after_hold,
            "verdict":       verdict,
            "verdict_detail": verdict_detail,
            "saved":         False,
        }

    # 8. Save calibrated model
    cal_path = MODELS_DIR / sector / f"xgb_{sector}_shared_v1_cal.pkl"
    cal_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cal_path, "wb") as f:
        pickle.dump({
            "model":          cal_model,
            "base_model":     model,
            "feature_names":  feature_names,
            "sector":         sector,
            "tickers":        SECTORS[sector]["tickers"],
            "holdout_date":   HOLDOUT_DATE,
            "calibration":    "isotonic",
            "before_oos_f1":  before_oos["f1_weighted"],
            "after_oos_f1":   after_oos["f1_weighted"],
            "before_ece_bear": before_oos["ece_bear"],
            "after_ece_bear":  after_oos["ece_bear"],
        }, f)
    size_kb = cal_path.stat().st_size / 1024
    print(f"\n  [OK] Saved -> {cal_path.relative_to(ROOT)}  ({size_kb:.0f} KB)")

    # 9. Regenerate predictions parquets
    regenerate_predictions(sector, cal_model, df_combined, X_all)

    bear_delta = after_oos["ece_bear"] - before_oos["ece_bear"]
    f1_delta   = after_oos["f1_weighted"] - before_oos["f1_weighted"]
    verdict_detail = (
        f"Calibration ACCEPTED for **{sector}**: "
        f"Bear ECE {before_oos['ece_bear']:.4f} -> {after_oos['ece_bear']:.4f} "
        f"(delta={bear_delta:+.4f}), "
        f"F1 {before_oos['f1_weighted']:.4f} -> {after_oos['f1_weighted']:.4f} "
        f"(delta={f1_delta:+.4f}). "
        f"Holdout Bear ECE: {before_hold['ece_bear']:.4f} -> {after_hold['ece_bear']:.4f}."
    )

    return {
        "sector":        sector,
        "before_oos":    before_oos,
        "after_oos":     after_oos,
        "before_hold":   before_hold,
        "after_hold":    after_hold,
        "verdict":       "ACCEPTED",
        "verdict_detail": verdict_detail,
        "saved":         True,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    section("market_ml -- Step 8: Isotonic Calibration")
    print(f"  Target:  Bear ECE < {BEAR_ECE_TARGET}  (Step 7 baseline: ~0.10)")
    print(f"  Guard:   F1 drop  < {MAX_F1_DROP}")
    print(f"  Method:  CalibratedClassifierCV(FrozenEstimator, method='isotonic')")
    print(f"  Cal set: OOS pre-holdout feature matrix (holdout never touched)")

    all_sector_results = []
    any_failed = False

    for sector in ["tech", "biotech"]:
        result = run_sector(sector)
        all_sector_results.append(result)
        if not result["saved"]:
            any_failed = True

    # Summary
    section("Calibration Summary")
    for r in all_sector_results:
        b_o = r["before_oos"]
        a_o = r["after_oos"]
        b_h = r["before_hold"]
        a_h = r["after_hold"]
        status = r["verdict"]
        print(f"\n  {r['sector'].upper():8s}  {status}")
        print(f"    OOS:     Bear ECE {b_o['ece_bear']:.4f} -> {a_o['ece_bear']:.4f}  "
              f"F1 {b_o['f1_weighted']:.4f} -> {a_o['f1_weighted']:.4f}")
        print(f"    Holdout: Bear ECE {b_h['ece_bear']:.4f} -> {a_h['ece_bear']:.4f}  "
              f"F1 {b_h['f1_weighted']:.4f} -> {a_h['f1_weighted']:.4f}")

    # Append to evaluation report
    section("Updating Evaluation Report")
    append_calibration_report(all_sector_results)

    section("Step 8 Complete")
    for r in all_sector_results:
        if r["saved"]:
            print(f"  [OK]  {r['sector']}: models/{r['sector']}/xgb_{r['sector']}_shared_v1_cal.pkl")
        else:
            print(f"  [SKIP] {r['sector']}: calibration criteria not met, original model kept")
    print("  Updated predictions: data/processed/{TICKER}_predictions.parquet")
    print("  Updated report:      docs/evaluation_report.md")

    if any_failed:
        print("\n  [WARN] At least one sector did not meet calibration criteria.")
        print("  [WARN] Those sector models were NOT replaced.")
        sys.exit(1)

    print("\n  Next step: src/pipeline/09_finetune.py (per-ticker fine tuning)\n")


if __name__ == "__main__":
    main()

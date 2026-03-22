#!/usr/bin/env bash
# ============================================================
# data-validation.sh
# Verifies that parquet files exist, are non-empty, have
# reasonable row counts, and have expected columns.
#
# Usage:
#   bash .claude/hooks/data-validation.sh           # validate all stages
#   bash .claude/hooks/data-validation.sh raw       # raw data only
#   bash .claude/hooks/data-validation.sh processed # features only
# ============================================================

set -euo pipefail

PYTHON="C:/Users/borra/anaconda3/python.exe"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

STAGE="${1:-all}"   # all | raw | processed | events

echo ""
echo "============================================================"
echo "  DATA VALIDATION"
echo "  market_ml — $(date '+%Y-%m-%d %H:%M')"
echo "  Stage: $STAGE"
echo "============================================================"
echo ""

FAIL=0

# ── Raw data validation ──────────────────────────────────────────────────────
if [[ "$STAGE" == "all" || "$STAGE" == "raw" ]]; then
    echo "[RAW] Validating data/raw/ ..."
    echo ""

    "$PYTHON" - <<'PYEOF'
import sys
from pathlib import Path
import pandas as pd
from config.tickers import SECTORS, TICKER_SECTOR

raw_dir = Path("data/raw")
expected_tickers = sorted(TICKER_SECTOR.keys())

# Min row thresholds per ticker (short-history tickers get lower bar)
MIN_ROWS = {
    "META": 500,   # 2012 IPO
    "MRNA": 200,   # 2018 IPO
}
DEFAULT_MIN_ROWS = 2000

REQUIRED_COLS = {"open", "high", "low", "close", "volume"}

fail = False
print(f"  Expected tickers: {expected_tickers}\n")

for ticker in expected_tickers:
    fpath = raw_dir / f"{ticker}_daily_raw.parquet"

    if not fpath.exists():
        print(f"  [FAIL] {ticker}: file missing -> {fpath}")
        fail = True
        continue

    df = pd.read_parquet(fpath)
    min_rows = MIN_ROWS.get(ticker, DEFAULT_MIN_ROWS)
    size_kb  = fpath.stat().st_size / 1024
    issues   = []

    # Row count check
    if len(df) < min_rows:
        issues.append(f"only {len(df)} rows (min {min_rows})")

    # Required columns
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        issues.append(f"missing columns: {missing}")

    # No NaN in close
    if df["close"].isna().any():
        n_nan = df["close"].isna().sum()
        issues.append(f"{n_nan} NaN values in close")

    # Index sanity
    if df.index.name != "date":
        issues.append(f"wrong index name: {df.index.name!r}")

    # Monotonic index
    if not df.index.is_monotonic_increasing:
        issues.append("index not sorted chronologically")

    sector = TICKER_SECTOR[ticker]
    if issues:
        print(f"  [FAIL] {ticker:6s} ({sector:7s}): {'; '.join(issues)}")
        fail = True
    else:
        print(f"  [PASS] {ticker:6s} ({sector:7s}): "
              f"{len(df):5d} rows  "
              f"{str(df.index[0].date()):12s} to {str(df.index[-1].date())}  "
              f"({size_kb:.0f} KB)")

if fail:
    sys.exit(1)
PYEOF

    STATUS=$?
    [ $STATUS -ne 0 ] && FAIL=1
    echo ""
fi

# ── Processed features validation ────────────────────────────────────────────
if [[ "$STAGE" == "all" || "$STAGE" == "processed" ]]; then
    echo "[PROCESSED] Validating data/processed/ ..."
    echo ""

    PROCESSED_DIR="$ROOT/data/processed"

    if [ ! -d "$PROCESSED_DIR" ] || [ -z "$(ls "$PROCESSED_DIR"/*.parquet 2>/dev/null)" ]; then
        echo "  [SKIP] No processed parquets found — run 02_features.py first."
    else
        "$PYTHON" - <<'PYEOF'
import sys
from pathlib import Path
import pandas as pd
from config.tickers import TICKER_SECTOR

processed_dir = Path("data/processed")

# Minimum features we expect in every features parquet
EXPECTED_FEATURES = [
    "atr_pct", "bb_pct", "bb_width", "candle_body", "candle_dir",
    "close_vs_sma10", "close_vs_sma200", "cross_50_200",
    "hvol_21d", "hvol_63d", "macd_hist", "price_52w_pct",
    "rsi_14", "volume_zscore", "return_1d",
]

fail = False

for f in sorted(processed_dir.glob("*_features.parquet")):
    ticker = f.stem.split("_")[0]
    if ticker not in TICKER_SECTOR:
        print(f"  [WARN] {ticker}: not in TICKER_SECTOR registry — orphan file?")
        continue

    df = pd.read_parquet(f)
    size_kb = f.stat().st_size / 1024
    issues  = []

    # Row count — after 200-day warm-up short tickers may have few rows
    if len(df) < 100:
        issues.append(f"only {len(df)} rows after warm-up")

    # Expected features present
    missing_feats = [c for c in EXPECTED_FEATURES if c not in df.columns]
    if missing_feats:
        issues.append(f"missing expected features: {missing_feats[:3]}...")

    # No all-NaN columns
    all_nan = [c for c in df.columns if df[c].isna().all()]
    if all_nan:
        issues.append(f"all-NaN columns: {all_nan}")

    # Index sorted
    if not df.index.is_monotonic_increasing:
        issues.append("index not chronologically sorted")

    if issues:
        print(f"  [FAIL] {ticker:6s}: {'; '.join(issues)}")
        fail = True
    else:
        print(f"  [PASS] {ticker:6s}: "
              f"{len(df):5d} rows  "
              f"{len(df.columns):2d} columns  "
              f"{str(df.index[0].date())} to {str(df.index[-1].date())}  "
              f"({size_kb:.0f} KB)")

if fail:
    sys.exit(1)
PYEOF
        STATUS=$?
        [ $STATUS -ne 0 ] && FAIL=1
    fi
    echo ""
fi

# ── Events validation ─────────────────────────────────────────────────────────
if [[ "$STAGE" == "all" || "$STAGE" == "events" ]]; then
    echo "[EVENTS] Validating data/events/ ..."
    echo ""

    EVENTS_DIR="$ROOT/data/events"
    FOUND=0

    for subdir in universal tech biotech stocks; do
        DIR="$EVENTS_DIR/$subdir"
        if [ -d "$DIR" ]; then
            COUNT=$(ls "$DIR"/*.parquet 2>/dev/null | wc -l || true)
            if [ "$COUNT" -gt 0 ]; then
                echo "  [PASS] events/$subdir/: $COUNT parquet file(s)"
                FOUND=1
            else
                echo "  [INFO] events/$subdir/: empty (not yet collected)"
            fi
        fi
    done

    if [ "$FOUND" -eq 0 ]; then
        echo "  [INFO] No event data collected yet — run 05_events.py to populate."
    fi
    echo ""
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo "============================================================"
if [ "$FAIL" -eq 0 ]; then
    echo "  RESULT: PASS — all data validated successfully"
else
    echo "  RESULT: FAIL — fix errors above before proceeding"
fi
echo "============================================================"
echo ""

exit $FAIL

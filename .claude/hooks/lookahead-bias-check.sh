#!/usr/bin/env bash
# ============================================================
# lookahead-bias-check.sh
# Scans feature parquets and pipeline scripts for common
# lookahead bias patterns.
#
# Usage:
#   bash .claude/hooks/lookahead-bias-check.sh           # check all tickers
#   bash .claude/hooks/lookahead-bias-check.sh AAPL      # check one ticker
# ============================================================

set -euo pipefail

PYTHON="C:/Users/borra/anaconda3/python.exe"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROCESSED="$ROOT/data/processed"
SRC="$ROOT/src"

TICKER="${1:-ALL}"

echo ""
echo "============================================================"
echo "  LOOKAHEAD BIAS CHECK"
echo "  market_ml — $(date '+%Y-%m-%d %H:%M')"
echo "============================================================"
echo ""

FAIL=0

# ── 1. Scan source files for forward-looking column names ────────────────────
echo "[1/3] Scanning src/ for forward-looking keywords in column definitions..."
echo ""

FWD_KEYWORDS=("future_" "fwd_" "_forward" "next_price" "tomorrow" "_lead" "lookahead")
FOUND_CODE=0

for kw in "${FWD_KEYWORDS[@]}"; do
    # Search for keyword in feature assignment patterns (df["..."] or df[...] =)
    matches=$(grep -rn --include="*.py" "$kw" "$SRC" 2>/dev/null | \
              grep -v "days_to_next_earnings\|days_to_next_product\|PDUFA\|next_earnings" | \
              grep -v "^\s*#" | \
              grep -v "fwd_keywords" || true)
    if [ -n "$matches" ]; then
        echo "  [WARN] Keyword '$kw' found:"
        echo "$matches" | sed 's/^/    /'
        FOUND_CODE=1
        FAIL=1
    fi
done

if [ "$FOUND_CODE" -eq 0 ]; then
    echo "  [PASS] No forward-looking keywords found in source files."
fi
echo ""

# ── 2. Scan source files for illegal pandas shift patterns ───────────────────
echo "[2/3] Scanning src/ for illegal shift() patterns (negative = forward-looking)..."
echo ""

# shift(-n) on non-label columns is suspicious
# Labels intentionally use shift(-n); feature scripts must not
SHIFT_MATCHES=$(grep -rn --include="*.py" 'shift(-' "$SRC" 2>/dev/null || true)

if [ -n "$SHIFT_MATCHES" ]; then
    echo "  [WARN] Negative shift() detected — verify these are label columns only:"
    echo "$SHIFT_MATCHES" | sed 's/^/    /'
    echo ""
    echo "  RULE: shift(-n) is ONLY allowed in 03_labels.py for computing forward returns."
    echo "  If this appears in 02_features.py or later scripts, it IS lookahead bias."
    # Don't auto-fail on shift (labels need it) — warn for human review
    echo ""
    echo "  [ACTION REQUIRED] Manually confirm each shift(-n) is in a labels script."
else
    echo "  [PASS] No negative shift() patterns found in source files."
fi
echo ""

# ── 3. Validate feature parquet column names ─────────────────────────────────
echo "[3/3] Scanning feature parquets for suspicious column names..."
echo ""

if [ ! -d "$PROCESSED" ]; then
    echo "  [SKIP] data/processed/ does not exist yet — run 02_features.py first."
else
    "$PYTHON" - <<'PYEOF'
import sys
from pathlib import Path
import pandas as pd

processed = Path("data/processed")
fwd_keywords = ["future", "fwd", "forward", "next_price", "tomorrow", "lead", "lookahead"]
# Allowed "next_" columns (these are event calendar lookups, not price lookahead)
allowed_next = {"days_to_next_earnings", "days_to_next_product_event", "days_to_next_pdufa"}

fail = False
ticker_filter = sys.argv[1] if len(sys.argv) > 1 else "ALL"

for f in sorted(processed.glob("*_features.parquet")):
    ticker = f.stem.split("_")[0]
    if ticker_filter != "ALL" and ticker != ticker_filter.upper():
        continue

    df = pd.read_parquet(f)
    bad_cols = []
    for col in df.columns:
        col_lower = col.lower()
        if col in allowed_next:
            continue
        for kw in fwd_keywords:
            if kw in col_lower:
                bad_cols.append(col)
                break

    if bad_cols:
        print(f"  [FAIL] {ticker}: suspicious columns detected: {bad_cols}")
        fail = True
    else:
        print(f"  [PASS] {ticker}: {len(df.columns)} columns, no lookahead names found")

if fail:
    sys.exit(1)
PYEOF
    STATUS=$?
    if [ $STATUS -ne 0 ]; then
        FAIL=1
    fi
fi

echo ""
echo "============================================================"
if [ "$FAIL" -eq 0 ]; then
    echo "  RESULT: PASS — no lookahead bias patterns detected"
else
    echo "  RESULT: FAIL — review warnings above before training"
fi
echo "============================================================"
echo ""

exit $FAIL

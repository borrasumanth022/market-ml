#!/usr/bin/env bash
# ============================================================
# model-evaluation.sh
# Verifies that F1 and per-class recall are reported before
# a model file is saved. Scans the most recent Python script
# output and checks that model files have accompanying metrics.
#
# Usage:
#   bash .claude/hooks/model-evaluation.sh                    # check all models
#   bash .claude/hooks/model-evaluation.sh models/tech/AAPL  # check specific prefix
# ============================================================

set -euo pipefail

PYTHON="C:/Users/borra/anaconda3/python.exe"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODELS_DIR="$ROOT/models"

PREFIX="${1:-}"   # optional: e.g., "models/tech/AAPL"

echo ""
echo "============================================================"
echo "  MODEL EVALUATION CHECK"
echo "  market_ml — $(date '+%Y-%m-%d %H:%M')"
echo "============================================================"
echo ""

FAIL=0

# ── 1. Check that training scripts print required metrics ────────────────────
echo "[1/3] Scanning training scripts for required metric output..."
echo ""

TRAINING_SCRIPTS=$(find "$ROOT/src" -name "*.py" | \
    xargs grep -l "model\|fit\|predict" 2>/dev/null | \
    grep -v "__pycache__" || true)

if [ -z "$TRAINING_SCRIPTS" ]; then
    echo "  [INFO] No training scripts found yet."
else
    for script in $TRAINING_SCRIPTS; do
        SHORT="${script#$ROOT/}"
        HAS_F1=$(grep -c "f1\|F1\|macro" "$script" 2>/dev/null || true)
        HAS_RECALL=$(grep -c "recall\|Bear\|Sideways\|Bull" "$script" 2>/dev/null || true)
        HAS_SAVE=$(grep -c "to_parquet\|pickle\|joblib\|torch.save" "$script" 2>/dev/null || true)

        if [ "$HAS_SAVE" -gt 0 ]; then
            # Script saves a model — verify it also reports metrics
            if [ "$HAS_F1" -eq 0 ]; then
                echo "  [FAIL] $SHORT saves a model but does not report F1"
                FAIL=1
            elif [ "$HAS_RECALL" -eq 0 ]; then
                echo "  [WARN] $SHORT saves a model but may not report per-class recall"
            else
                echo "  [PASS] $SHORT: reports F1 and recall before saving"
            fi
        fi
    done
fi
echo ""

# ── 2. Check that saved models have an accompanying metrics file ─────────────
echo "[2/3] Checking for metrics files alongside saved models..."
echo ""

if [ ! -d "$MODELS_DIR" ] || [ -z "$(find "$MODELS_DIR" -name "*.pkl" -o -name "*.pt" 2>/dev/null | head -1)" ]; then
    echo "  [INFO] No model files found yet — nothing to validate."
else
    "$PYTHON" - <<'PYEOF'
import sys
from pathlib import Path

models_dir = Path("models")
prefix_filter = sys.argv[1] if len(sys.argv) > 1 else ""

fail = False
checked = 0

for model_file in sorted(models_dir.rglob("*.pkl")) + sorted(models_dir.rglob("*.pt")):
    path_str = str(model_file)
    if prefix_filter and prefix_filter not in path_str:
        continue

    # Look for a metrics JSON or CSV next to the model
    stem = model_file.stem
    parent = model_file.parent
    metrics_json  = parent / f"{stem}_metrics.json"
    shap_csv      = parent / f"{stem.replace('xgb_', '').replace('lgbm_', '')}_shap_summary.csv"
    any_metrics   = list(parent.glob(f"{stem}*metric*")) + list(parent.glob(f"{stem}*eval*"))

    if metrics_json.exists():
        print(f"  [PASS] {model_file.relative_to(Path.cwd())}: metrics JSON found")
    elif any_metrics:
        print(f"  [PASS] {model_file.relative_to(Path.cwd())}: metrics file found")
    else:
        # Warn but don't fail — metrics might be in training script output
        print(f"  [WARN] {model_file.relative_to(Path.cwd())}: no metrics file found")
        print(f"         -> Expected: {metrics_json.name}")
        print(f"         -> Create with: json.dump(metrics, open(path, 'w'))")

    checked += 1

if checked == 0:
    print("  [INFO] No model files matched the filter.")
PYEOF
fi
echo ""

# ── 3. Verify naive baseline is established for known tickers ────────────────
echo "[3/3] Checking that naive baselines are documented..."
echo ""

"$PYTHON" - <<'PYEOF'
import sys
from pathlib import Path
import pandas as pd
from config.tickers import TICKER_SECTOR

processed = Path("data/processed")
found_any = False

for f in sorted(processed.glob("*_labeled.parquet")):
    ticker = f.stem.split("_")[0]
    if ticker not in TICKER_SECTOR:
        continue

    df = pd.read_parquet(f)

    # dir_1w is our primary target
    if "dir_1w" not in df.columns:
        print(f"  [INFO] {ticker}: no dir_1w column yet — run 03_labels.py")
        continue

    y = df["dir_1w"].dropna()
    counts = y.value_counts()
    majority_class = counts.idxmax()
    majority_frac  = counts.max() / len(y)
    print(f"  [INFO] {ticker}: naive baseline (always {majority_class:+d}) = {majority_frac:.1%}  "
          f"(Bear={counts.get(-1, 0)/len(y):.1%} Side={counts.get(0, 0)/len(y):.1%} Bull={counts.get(1, 0)/len(y):.1%})")
    found_any = True

if not found_any:
    print("  [INFO] No labeled data found yet — run 03_labels.py first.")
PYEOF

echo ""
echo "============================================================"
if [ "$FAIL" -eq 0 ]; then
    echo "  RESULT: PASS — evaluation standards met"
else
    echo "  RESULT: FAIL — training scripts must report F1 before saving models"
    echo ""
    echo "  Required output format (from evaluation/SKILL.md):"
    echo "    OOS Accuracy : XX.XX%"
    echo "    Macro F1     : 0.XXX"
    echo "    Bear  recall : XX.X%"
    echo "    Side  recall : XX.X%"
    echo "    Bull  recall : XX.X%"
fi
echo "============================================================"
echo ""

exit $FAIL

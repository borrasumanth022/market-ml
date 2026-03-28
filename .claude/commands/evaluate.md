# /project:evaluate -- Per-ticker model evaluation

**Usage:**
- `/project:evaluate` -- all tickers
- `/project:evaluate AAPL` -- single ticker
- `/project:evaluate tech` -- all tech tickers

## Instructions

1. Read `CLAUDE.local.md` for PYTHON_EXE.

2. Run: `{PYTHON_EXE} src/pipeline/07_evaluate.py [ticker|sector]`

3. Print per-ticker F1 table sorted by descending F1:
   ```
   Ticker  Sector    Acc       F1     Bear     Side     Bull   Rows
   AAPL    tech    xx.xx%   0.xxx   xx.x%   xx.x%   xx.x%   7856
   ```

4. Flag outliers:
   - F1 more than 0.05 below sector mean -> UNDERPERFORMER
   - Bear recall < 10% or Sideways recall < 25% -> CLASS COLLAPSE

5. Calibration check: flag if model is overconfident (prob > 0.7 but accuracy ~0.5).

6. Directional accuracy:
   - % of Bull predictions followed by positive 1-week return
   - % of Bear predictions followed by negative 1-week return

7. If profit simulation available, print annualised return and Sharpe per ticker.

8. Compare to sector aggregates in `CLAUDE.md` Step 6. Flag systematic differences.

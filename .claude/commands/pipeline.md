# /project:pipeline -- Run the market_ml pipeline

**Usage:**
- `/project:pipeline` -- run all steps 1-6 for all tickers
- `/project:pipeline 2` -- run step 2 (features) for all tickers
- `/project:pipeline 2 NVDA` -- run step 2 for NVDA only
- `/project:pipeline 6 tech` -- train tech sector only
- `/project:pipeline 6 biotech` -- train biotech sector only

## Instructions

1. Read `CLAUDE.local.md` to get PYTHON_EXE.

2. Step-to-script mapping:
   - 1 = src/pipeline/01_fetch_data.py
   - 2 = src/pipeline/02_features.py
   - 3 = src/pipeline/03_labels.py
   - 4 = src/pipeline/04_events.py
   - 5 = src/pipeline/05_event_features.py
   - 6 = src/pipeline/06_train.py
   - 7 = src/pipeline/07_evaluate.py

3. If no step given, run steps 1-6 in order.

4. Pass CLI args: ticker (steps 1-5) or sector (step 6) as positional arg.

5. After each step, confirm expected output file exists:
   - Step 1: data/raw/{TICKER}_daily_raw.parquet
   - Step 2: data/processed/{TICKER}_features.parquet
   - Step 3: data/processed/{TICKER}_labeled.parquet
   - Step 4: data/events/universal/macro_events.parquet
   - Step 5: data/processed/{TICKER}_with_events.parquet
   - Step 6: models/tech/xgb_tech_shared_v1.pkl and/or models/biotech/xgb_biotech_shared_v1.pkl

6. If any step exits non-zero, stop and report the error. Do not continue.

7. After step 6, extract OOS F1 from stdout and compare to baseline in CLAUDE.md.

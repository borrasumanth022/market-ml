# /project:train -- Train sector XGBoost models

**Usage:**
- `/project:train` -- train both sectors
- `/project:train tech` -- tech only
- `/project:train biotech` -- biotech only

## Instructions

1. Read `CLAUDE.local.md` for PYTHON_EXE.

2. Run: `{PYTHON_EXE} src/pipeline/06_train.py [tech|biotech]`

3. Parse stdout for: walk-forward F1, OOS F1, holdout F1, per-class recall, top SHAP features.

4. Compare to Step 6 baseline from `CLAUDE.md`:
   - Tech: OOS F1=0.402, Holdout F1=0.414
   - Biotech: OOS F1=0.403, Holdout F1=0.386

   Print:
   ```
   Sector    OOS F1   Holdout F1   vs baseline
   tech      0.xxx    0.xxx        +/-0.xxx
   biotech   0.xxx    0.xxx        +/-0.xxx
   ```

5. If Sideways recall drops below 0.40, flag it explicitly.

6. Confirm model artifacts written:
   - models/tech/xgb_tech_shared_v1.pkl
   - models/biotech/xgb_biotech_shared_v1.pkl

7. If both OOS and holdout F1 improve, prompt to update `CLAUDE.md` with new numbers.

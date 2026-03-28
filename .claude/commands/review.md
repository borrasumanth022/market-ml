# /project:review -- Code review for ML correctness and lookahead bias

**Usage:**
- `/project:review` -- review files changed in last commit
- `/project:review src/pipeline/07_evaluate.py` -- specific file
- `/project:review src/pipeline/` -- all .py files in directory

## Instructions

Adopt the persona from `.claude/agents/code-reviewer.md`.

1. Determine files to review:
   - If path given, read that file/directory
   - If no path, run `git diff HEAD~1 --name-only` and review those files

2. For each Python file, run the full checklist:

   **Lookahead bias:**
   - [ ] No shift(-N) on price/return columns as features
   - [ ] No rolling(..., center=True) on feature columns
   - [ ] No target column or derivative used as a feature
   - [ ] No diff(-1) on price data in feature columns

   **Walk-forward validation:**
   - [ ] Uses TimeSeriesSplit, not KFold(shuffle=True)
   - [ ] Train rows always earlier in time than test rows
   - [ ] No test-period data used for scaler fitting or encoding

   **Holdout:**
   - [ ] Cutoff is exactly 2024-01-01
   - [ ] Holdout not touched until final evaluation block

   **Coding standards:**
   - [ ] All print() ASCII-only (no Unicode arrows/box chars)
   - [ ] Ticker lists from config.tickers, not hardcoded
   - [ ] Skip-if-exists pattern on all parquet writes
   - [ ] sys.exit(1) on bad data, not silent pass

3. For each FAIL: show file path, line number, offending code, why wrong, correct fix.

4. Final verdict: CLEAN or ISSUES FOUND (N issues).

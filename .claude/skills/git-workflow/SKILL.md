# Skill: git-workflow

Branch naming, commit format, and push policy for market_ml.

## What is tracked in git

**Tracked (committed):**
- `src/` — all pipeline scripts
- `config/` — ticker registry and settings
- `docs/` — architecture notes and ADRs
- `.claude/` — skills and hooks (but NOT settings.local.json)
- `README.md`
- `.gitignore`
- `tests/`

**Not tracked (gitignored):**
- `data/raw/` and `data/processed/` — parquet files are large binary blobs; regenerate from scripts
- `data/events/` — same reason
- `models/` — trained model binaries (.pkl, .pt); too large and non-diffable
- `CLAUDE.md` — local context for Claude Code, not project documentation
- `.claude/settings.local.json` — contains machine-specific paths and local permissions
- `notebooks/` — Jupyter checkpoints and outputs are noisy

---

## Branch naming convention

```
{type}/{short-description}
```

Types:
- `feat/` — new pipeline script or feature group
- `fix/` — bug fix
- `exp/` — experiment (new model, hyperparameter search)
- `data/` — new data source or event collection
- `docs/` — documentation only

Examples:
```
feat/02-feature-engineering
feat/03-labels
exp/xgb-baseline-tech-sector
data/biotech-fda-events
fix/nvda-warmup-rows
docs/add-evaluation-adr
```

---

## Commit message format

Short subject line (imperative mood, ≤72 chars), optional body with context.

```
Add ticker-agnostic feature engineering (02_features.py)

Reproduces the 36-feature set from aapl_ml for all 11 tickers.
Skips already-processed files; delete to recompute.
MRNA drops to 1,630 rows after 200-day warm-up (expected).
```

**Good subjects:**
- `Add walk-forward XGBoost baseline for tech sector`
- `Fix MRNA warm-up row count (200 not 252)`
- `Add FDA PDUFA events for biotech sector`

**Bad subjects:**
- `update stuff`
- `WIP`
- `fix bug` (which bug?)

---

## When to commit

- After each pipeline step is tested and producing correct output
- Before starting a new experiment that touches existing files
- After fixing a confirmed bug (not mid-investigation)
- Never commit broken code to main

---

## Push policy

**Never push to main directly.** Always:
1. Create a feature/experiment branch
2. Do the work, commit locally
3. Run validation hooks before pushing
4. Push the branch, review the diff mentally
5. Merge to main only when the step is confirmed working

```bash
git checkout -b feat/02-feature-engineering
# ... do work ...
bash .claude/hooks/data-validation.sh
git add src/pipeline/02_features.py
git commit -m "Add ticker-agnostic feature engineering (02_features.py)"
git push origin feat/02-feature-engineering
# then merge to main
```

---

## Data reproducibility note

Since data files are gitignored, the git history must make it possible to regenerate
everything from scratch. This means:

1. Commit the fetch and feature scripts before committing any model
2. Each pipeline script must be runnable standalone: `python src/pipeline/02_features.py`
3. Config must be complete: `config/tickers.py` fully defines the universe
4. Document any external data sources (FRED URLs, FDA calendar URLs) in the event
   collection scripts so they can be re-fetched

---

## Tagging champion models

When a new overall champion is established, tag the commit:

```bash
git tag -a "champion/tech/xgb-phase1-baseline" -m "XGB tech baseline F1=0.xxx"
git push origin --tags
```

This creates a permanent reference point even though the model file itself isn't tracked.

---

## .gitignore maintenance

If a new type of output file appears that shouldn't be tracked, add it to `.gitignore`
before `git add .` ever picks it up. Common additions needed:

```
# If we add SHAP plots
docs/shap_*.png

# If we add hyperparameter search outputs
experiments/

# If we add API keys or credentials
*.key
credentials.json
.env
```

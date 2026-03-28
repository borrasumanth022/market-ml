Commit and push the current changes to GitHub for $ARGUMENTS (default: current branch).

Steps:
1. Run `git status` — show what has changed.
2. Block if any of these paths are staged (never commit these):
   - `data/` (any parquet, any raw or processed file)
   - `models/` (any .pkl file)
   - `CLAUDE.md` or `CLAUDE.local.md`
   - `.claude/settings.local.json`
   - `*.log`, `*.ipynb`
3. Stage only safe files:
   - `src/` — pipeline scripts and utilities
   - `config/` — tickers.py and other config
   - `.claude/` — except settings.local.json (already gitignored)
   - `docs/` — any documentation
   - `requirements.txt`, `.gitignore`, `README.md`
4. Show a `git diff --stat` of what will be committed.
5. Propose a commit message in the project convention:
   - Format: `<type>(<scope>): <description>` where description ≤ 72 chars
   - Types: `feat`, `fix`, `refactor`, `docs`, `chore`, `data`, `model`, `exp`
   - Scopes: `pipeline`, `features`, `events`, `training`, `evaluation`, `config`
   - Include metric delta when relevant: `feat(training): add macro interactions — Tech F1 0.402 (+0.027)`
6. Commit with Co-Authored-By tag.
7. Push to current branch. If on `main`, warn and offer to create a feature branch first — never push directly to main.

If $ARGUMENTS contains a commit message in quotes, use that message directly.
If $ARGUMENTS contains `--tag`, also create a git tag after pushing (for model checkpoints).

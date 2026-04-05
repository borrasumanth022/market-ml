"""
Agent 1 -- Regime and Risk Oversight Agent
==========================================
Runs every Monday morning BEFORE the signal generator.

What it does:
  1. Reads the latest regime state (VIX, yield spread, HMM label) from signal_log.parquet
  2. Fetches top 10 financial headlines from Yahoo Finance RSS (free, no key required)
     Fallback: MarketWatch RSS
  3. Calls Claude API (claude-haiku-4-5-20251001) to explain the regime in 2-3 sentences
  4. Saves the full context report to docs/regime_notes/YYYY-MM-DD.md
  5. Saves the raw agent context (just Claude's response) to docs/regime_notes/YYYY-MM-DD_context.txt
  6. Prints the full report to terminal
  7. Updates the notes column of the most recent week's rows in signal_log.parquet

If the Claude API call fails for any reason, the agent exits cleanly -- it never
blocks or corrupts the signal generator run that follows.

Usage:
    C:\\Users\\borra\\anaconda3\\python.exe src\\agents\\regime_agent.py

Environment:
    ANTHROPIC_API_KEY  -- required; if not set, agent exits with error and signal
                          generator still runs normally.
"""

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import requests

# ── Paths ──────────────────────────────────────────────────────────────────────
SIGNALS_DIR    = ROOT / "data" / "signals"
LOG_PATH       = SIGNALS_DIR / "signal_log.parquet"
REGIME_NOTES   = ROOT / "docs" / "regime_notes"

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL          = "claude-haiku-4-5-20251001"
MAX_TOKENS     = 512
HEADLINE_COUNT = 10

RSS_FEEDS = [
    # Yahoo Finance S&P 500 headlines
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EGSPC&region=US&lang=en-US",
    # MarketWatch top stories
    "https://feeds.marketwatch.com/marketwatch/topstories/",
]

REGIME_LABELS = {0: "range-bound", 1: "trending", 2: "volatile"}


# ── Utilities ──────────────────────────────────────────────────────────────────

def _get_signal_date() -> pd.Timestamp:
    """Return the most recent Monday as signal_date."""
    today = pd.Timestamp.today().normalize()
    days_since_monday = today.weekday()
    return today - pd.Timedelta(days=days_since_monday)


def _read_latest_regime() -> dict:
    """
    Read the most recent row(s) from signal_log.parquet.
    Returns a dict with regime_state, regime_label, vix_close, yield_spread,
    latest_signal_date.
    """
    if not LOG_PATH.exists():
        print(f"  [WARN] signal_log.parquet not found at {LOG_PATH}")
        print(f"  [WARN] Using fallback regime: volatile / VIX unknown")
        return {
            "regime_state":   2,
            "regime_label":   "volatile",
            "vix_close":      None,
            "yield_spread":   None,
            "signal_date":    _get_signal_date(),
        }

    df = pd.read_parquet(LOG_PATH, engine="pyarrow")
    df.index = pd.to_datetime(df.index)
    latest_date = df.index.max()
    row = df[df.index == latest_date].iloc[0]

    vix    = float(row["vix_close"])    if pd.notna(row.get("vix_close"))    else None
    spread = float(row["yield_spread"]) if pd.notna(row.get("yield_spread")) else None
    state  = int(row["regime_state"])   if pd.notna(row.get("regime_state")) else 2
    label  = str(row["regime_label"])   if pd.notna(row.get("regime_label")) else "volatile"

    return {
        "regime_state":   state,
        "regime_label":   label,
        "vix_close":      vix,
        "yield_spread":   spread,
        "signal_date":    latest_date,
    }


def _fetch_headlines(n: int = HEADLINE_COUNT) -> list[str]:
    """
    Fetch top N financial headlines from RSS feed.
    Tries each feed in RSS_FEEDS in order; returns empty list if all fail.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    for url in RSS_FEEDS:
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code != 200:
                print(f"  [WARN] RSS feed returned HTTP {resp.status_code}: {url}")
                continue

            root = ET.fromstring(resp.content)

            # RSS 2.0: items are under channel/item
            titles = []
            for item in root.iter("item"):
                title_el = item.find("title")
                if title_el is not None and title_el.text:
                    titles.append(title_el.text.strip())
                if len(titles) >= n:
                    break

            if titles:
                print(f"  [OK]  Fetched {len(titles)} headlines from: {url[:60]}...")
                return titles[:n]

        except requests.exceptions.Timeout:
            print(f"  [WARN] RSS feed timed out: {url[:60]}...")
        except ET.ParseError as e:
            print(f"  [WARN] RSS XML parse error ({url[:60]}...): {e}")
        except Exception as e:
            print(f"  [WARN] RSS fetch error ({url[:60]}...): {e}")

    print("  [WARN] All RSS feeds failed -- proceeding with no headlines")
    return []


def _build_prompt(regime: dict, headlines: list[str]) -> str:
    """Build the Claude prompt with regime context and headlines."""
    vix_str    = f"{regime['vix_close']:.2f}" if regime["vix_close"] else "unknown"
    spread_str = f"{regime['yield_spread']:.2f}%" if regime["yield_spread"] else "unknown"
    state_name = regime["regime_label"].upper()
    date_str   = regime["signal_date"].strftime("%A, %B %d, %Y")

    headlines_str = "\n".join(
        f"  {i+1}. {h}" for i, h in enumerate(headlines)
    ) if headlines else "  (no headlines available)"

    return f"""You are a quantitative market analyst providing a brief regime context note for an options trader.

Current market conditions as of {date_str}:
  HMM Regime State: {state_name} (3-state HMM: range-bound / trending / volatile)
  VIX: {vix_str}
  10Y-2Y Yield Spread: {spread_str}

Top financial headlines this morning:
{headlines_str}

Strategy context: This trader sells iron condors (delta-neutral, short volatility) when the model
predicts a Sideways week with >= 60% confidence. Iron condors profit when the underlying stays
within a 2% range. They lose when volatility spikes cause a breakout in either direction.

In 2-3 concise sentences, explain:
1. Why is the market in this regime right now -- what is the specific driver from the headlines?
2. Is this volatility structural (likely to persist weeks) or transient (event-driven, likely to resolve)?
3. What should this iron condor trader specifically watch for or be cautious about this week?

Be specific and reference the actual headlines. Do not use bullet points -- write in plain prose."""


def _call_claude(prompt: str) -> str | None:
    """
    Call Claude API with the given prompt.
    Returns the response text, or None if the call fails.
    Never raises -- all exceptions are caught and logged.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("  [FAIL] ANTHROPIC_API_KEY not set in environment.")
        print("  [FAIL] Set it with: set ANTHROPIC_API_KEY=sk-ant-...")
        return None

    try:
        import anthropic
    except ImportError:
        print("  [FAIL] anthropic package not installed.")
        print("  [FAIL] Install with: pip install anthropic")
        return None

    try:
        client   = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model      = MODEL,
            max_tokens = MAX_TOKENS,
            messages   = [{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        print(f"  [OK]  Claude response received ({len(text)} chars, model={MODEL})")
        return text

    except anthropic.AuthenticationError:
        print("  [FAIL] Claude API authentication failed -- check ANTHROPIC_API_KEY")
        return None
    except anthropic.RateLimitError:
        print("  [FAIL] Claude API rate limit hit -- try again in a few minutes")
        return None
    except Exception as e:
        print(f"  [FAIL] Claude API call failed: {type(e).__name__}: {e}")
        return None


def _update_signal_log(signal_date: pd.Timestamp, context: str) -> None:
    """
    Append regime context to the notes column of all rows with signal_date
    in signal_log.parquet. Uses the date passed in (the latest date in the log).
    If the notes column already has per-ticker content, prepends regime context
    with a separator so both are preserved.
    """
    if not LOG_PATH.exists():
        print("  [SKIP] signal_log.parquet not found -- cannot update notes column")
        return

    try:
        df = pd.read_parquet(LOG_PATH, engine="pyarrow")
        df.index = pd.to_datetime(df.index)

        mask = df.index == signal_date
        if mask.sum() == 0:
            print(f"  [SKIP] No rows found for signal_date {signal_date.date()} in log")
            return

        prefix = f"[REGIME] {context}"

        def _merge_notes(existing: str) -> str:
            existing = str(existing).strip() if pd.notna(existing) else ""
            if not existing:
                return prefix
            # Don't double-write regime context if already present
            if "[REGIME]" in existing:
                return existing
            return f"{prefix} | {existing}"

        df.loc[mask, "notes"] = df.loc[mask, "notes"].apply(_merge_notes)

        df.to_parquet(LOG_PATH, engine="pyarrow", index=True)
        print(f"  [OK]  Updated notes for {mask.sum()} rows (signal_date {signal_date.date()})")

    except Exception as e:
        print(f"  [WARN] Could not update signal_log notes: {type(e).__name__}: {e}")
        print(f"  [WARN] Context saved to docs/regime_notes/ -- signal_log unchanged")


def _save_notes(signal_date: pd.Timestamp, regime: dict, headlines: list[str],
                context: str) -> None:
    """
    Save full regime context report to docs/regime_notes/YYYY-MM-DD.md
    and raw Claude context to docs/regime_notes/YYYY-MM-DD_context.txt.
    """
    REGIME_NOTES.mkdir(parents=True, exist_ok=True)
    date_str = signal_date.strftime("%Y-%m-%d")

    vix_str    = f"{regime['vix_close']:.2f}" if regime["vix_close"] else "N/A"
    spread_str = f"{regime['yield_spread']:.2f}%" if regime["yield_spread"] else "N/A"

    headlines_md = "\n".join(
        f"{i+1}. {h}" for i, h in enumerate(headlines)
    ) if headlines else "_No headlines fetched._"

    md_content = f"""# Regime Context -- {date_str}

**Signal Date:** {date_str} (Monday)
**Regime:** {regime['regime_label'].title()} (state {regime['regime_state']})
**VIX:** {vix_str}
**10Y-2Y Yield Spread:** {spread_str}
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Headlines
{headlines_md}

## Agent Context
{context}
"""

    md_path  = REGIME_NOTES / f"{date_str}.md"
    txt_path = REGIME_NOTES / f"{date_str}_context.txt"

    md_path.write_text(md_content, encoding="utf-8")
    txt_path.write_text(context, encoding="utf-8")

    print(f"  [OK]  Saved full report  : {md_path.relative_to(ROOT)}")
    print(f"  [OK]  Saved context text : {txt_path.relative_to(ROOT)}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    bar = "=" * 62
    print(f"\n{bar}")
    print(f"  Regime and Risk Oversight Agent")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{bar}")

    # 1. Read latest regime state
    print("\n[1/4] Reading latest regime from signal_log...")
    regime = _read_latest_regime()
    print(f"  Regime : {regime['regime_label']} (state {regime['regime_state']})")
    print(f"  VIX    : {regime['vix_close']}")
    print(f"  Spread : {regime['yield_spread']}")
    print(f"  Date   : {regime['signal_date'].date()}")

    # 2. Fetch headlines
    print("\n[2/4] Fetching financial headlines...")
    headlines = _fetch_headlines(HEADLINE_COUNT)
    if headlines:
        for i, h in enumerate(headlines, 1):
            safe = h.encode("ascii", errors="replace").decode("ascii")
            print(f"  {i:2}. {safe[:90]}")
    else:
        print("  (no headlines -- will proceed with regime data only)")

    # 3. Call Claude API
    print(f"\n[3/4] Calling Claude API ({MODEL})...")
    prompt  = _build_prompt(regime, headlines)
    context = _call_claude(prompt)

    if context is None:
        print("\n  [FAIL] Agent context unavailable -- exiting cleanly.")
        print("  [FAIL] Signal generator will run normally without regime notes.")
        sys.exit(0)   # exit 0 so bat file continues to signal_generator

    # 4. Save and display
    print(f"\n[4/4] Saving outputs...")

    signal_date = regime["signal_date"]
    _save_notes(signal_date, regime, headlines, context)
    _update_signal_log(signal_date, context)

    # Print full report to terminal
    print(f"\n{bar}")
    print(f"  REGIME CONTEXT -- {signal_date.strftime('%Y-%m-%d')}")
    print(f"  {regime['regime_label'].upper()}  |  VIX {regime['vix_close']}  |  Spread {regime['yield_spread']}%")
    print(f"{bar}")
    print()
    # Wrap context lines to 60 chars for terminal readability (ASCII only)
    safe_context = context.encode("ascii", errors="replace").decode("ascii")
    words = safe_context.split()
    line, lines = [], []
    for w in words:
        if sum(len(x) + 1 for x in line) + len(w) > 60:
            lines.append(" ".join(line))
            line = [w]
        else:
            line.append(w)
    if line:
        lines.append(" ".join(line))
    for l in lines:
        print(f"  {l}")
    print(f"\n{bar}\n")


if __name__ == "__main__":
    main()

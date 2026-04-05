"""
Step 4 -- Event data collection
================================
Collects event/commodity data and saves them to data/events/.

Part A -- Universal macro events (FRED, no API key needed)
  FEDFUNDS  : Fed funds rate (monthly)     -> fed_rate_change
  CPIAUCSL  : CPI (monthly)               -> cpi_change
  UNRATE    : Unemployment rate (monthly)  -> unemployment_change
  GDP       : GDP (quarterly)             -> gdp_change
  Output: data/events/universal/macro_events.parquet

Part B -- Biotech FDA events (openFDA public API, no key needed)
  For each biotech ticker: LLY, MRNA, BIIB, REGN, VRTX
    - Drug approval actions (AP)
    - Complete Response Letters (CR = not approved)
    - Tentative approvals (TA)
  Sources: openFDA drugsfda endpoint + FDA press releases (via openFDA event endpoint)
  Output: data/events/biotech/fda_events.parquet

Part C -- Credit Spread (BAMLH0A0HYM2) for Financials sector
  BAMLH0A0HYM2 : ICE BofA HY Credit Spread (daily, 1996-12-31+)
  Output: data/events/financials/credit_spreads.parquet

Part D -- Energy commodity data (FRED) for Energy sector
  DCOILWTICO : WTI crude oil price (daily, 1986+)
  DHHNGSP    : Henry Hub natural gas spot price (daily, 1997+)
  Output: data/events/energy/energy_events.parquet
  Wide-format DataFrame (one column per series); feature engineering done in step 5.
  NaN sentinels: natgas pre-1997 -> 0.0 (applied in step 5)

Part E -- Consumer staples event data (FRED)
  RSXFS  : Advance Retail Sales ex food services (monthly, 1992+)
  UMCSENT: University of Michigan Consumer Sentiment (monthly, 1978+)
  Output: data/events/consumer_staples/consumer_staples_events.parquet
  Wide-format daily DataFrame; feature engineering done in step 5.
  NaN sentinels: pre-1992 rows -> 0.0 (applied in step 5)

Schema for event files (Parts A and B):
  date           DatetimeIndex
  event_type     str   (macro / fda_action)
  event_subtype  str   (e.g. "fed_rate_change", "drug_approval")
  ticker         str   (ticker symbol, or "ALL" for universal events)
  magnitude      float (numeric size of event; NaN if not applicable)
  direction      int   (+1 positive, -1 negative, 0 neutral)
  source         str   ("fred", "openfda")
  description    str   (human-readable one-liner)

Usage:
    C:\\Users\\borra\\anaconda3\\python.exe src\\pipeline\\04_events.py
"""

import io
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests

from config.tickers import SECTORS, TICKER_SECTOR

# ── Output paths ───────────────────────────────────────────────────────────────
UNIVERSAL_DIR      = ROOT / "data" / "events" / "universal"
BIOTECH_DIR        = ROOT / "data" / "events" / "biotech"
FINANCIALS_DIR     = ROOT / "data" / "events" / "financials"
ENERGY_DIR         = ROOT / "data" / "events" / "energy"
CONSUMER_STAPLES_DIR = ROOT / "data" / "events" / "consumer_staples"

# ── API constants ──────────────────────────────────────────────────────────────
FRED_BASE       = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
FRED_START      = "1993-01-01"
OPENFDA_BASE    = "https://api.fda.gov/drug/drugsfda.json"
OPENFDA_LIMIT   = 100       # results per page (max 100 for nested arrays)
OPENFDA_DELAY   = 0.35      # seconds between requests (240 req/min limit)

# ── Biotech company -> openFDA sponsor_name mapping ───────────────────────────
# Exact strings as they appear in the openFDA drugsfda CDER database.
# Validated by querying known application numbers (see pipeline dev notes).
# NOTE: MRNA's products (COVID/RSV vaccines) are CBER biologics, not CDER drugs,
# so they don't appear in the drugsfda endpoint. MRNA uses hardcoded events below.
BIOTECH_SPONSORS = {
    "LLY":  {
        "name"    : "Eli Lilly",
        "terms"   : ["ELI LILLY AND CO"],          # validated: NDA215866 (Mounjaro)
    },
    "MRNA": {
        "name"    : "Moderna",
        "terms"   : [],                             # CBER products — use hardcoded events
    },
    "BIIB": {
        "name"    : "Biogen",
        "terms"   : ["BIOGEN INC", "BIOGEN IDEC", "BIOGEN"],  # multiple historical names
    },
    "REGN": {
        "name"    : "Regeneron",
        "terms"   : ["REGENERON PHARMACEUTICALS"],  # validated: BLA125387 (Eylea)
    },
    "VRTX": {
        "name"    : "Vertex Pharmaceuticals",
        "terms"   : ["VERTEX PHARMS"],              # validated: NDA203188 (Kalydeco)
    },
    # Phase 1 expansion tickers
    "ABBV": {
        "name"    : "AbbVie",
        "terms"   : ["ABBVIE INC", "ABBVIE"],       # Humira (NDA020715), Skyrizi, Rinvoq
    },
    "BMY":  {
        "name"    : "Bristol-Myers Squibb",
        "terms"   : ["BRISTOL-MYERS SQUIBB CO", "BRISTOL-MYERS SQUIBB"],  # Opdivo, Eliquis
    },
    "GILD": {
        "name"    : "Gilead Sciences",
        "terms"   : ["GILEAD SCIENCES INC", "GILEAD SCIENCES"],  # Sovaldi, Harvoni, Biktarvy
    },
    "AMGN": {
        "name"    : "Amgen",
        "terms"   : ["AMGEN INC", "AMGEN"],         # Enbrel, Neupogen, Prolia, Repatha
    },
    "PFE":  {
        "name"    : "Pfizer",
        "terms"   : ["PFIZER INC", "PFIZER"],       # Eliquis (co-mktg), Ibrance, Xeljanz
        # NOTE: COVID vaccines (Comirnaty) are CBER biologics, not in CDER drugsfda
    },
}

# ── Hardcoded FDA events for tickers not in CDER (MRNA = CBER biologics) ──────
# Moderna's products are vaccines/biologics reviewed by CBER, not available in
# the openFDA drugsfda (CDER) endpoint. Key FDA actions hardcoded from public records.
MRNA_HARDCODED_EVENTS = [
    # (date, event_subtype, direction, description)
    ("2021-08-23", "drug_approval",           1,  "MRNA FDA full approval: Spikevax (COVID-19 vaccine, BLA 125753)"),
    ("2022-10-20", "drug_approval",           1,  "MRNA FDA approval: Spikevax bivalent booster (BA.4/BA.5)"),
    ("2023-09-11", "drug_approval",           1,  "MRNA FDA approval: Spikevax updated XBB.1.5 formulation"),
    ("2024-05-31", "drug_approval",           1,  "MRNA FDA approval: mRESVIA (RSV vaccine, adults 60+)"),
    ("2024-08-22", "drug_approval",           1,  "MRNA FDA approval: Spikevax 2024-2025 formulation (JN.1)"),
    ("2024-06-12", "complete_response_letter", -1, "MRNA FDA Complete Response: mRNA-1283 next-gen COVID vaccine"),
]

# FDA submission status codes -> our direction / label mapping
FDA_STATUS_MAP = {
    "AP" : {"label": "drug_approval",           "direction":  1, "desc": "FDA approved"},
    "TA" : {"label": "tentative_approval",       "direction":  0, "desc": "Tentative approval"},
    "CR" : {"label": "complete_response_letter", "direction": -1, "desc": "Complete Response Letter (not approved)"},
    "CRL": {"label": "complete_response_letter", "direction": -1, "desc": "Complete Response Letter (not approved)"},
    "RF" : {"label": "refuse_to_file",           "direction": -1, "desc": "Refuse to File"},
    "WD" : {"label": "withdrawn",                "direction": -1, "desc": "Application withdrawn"},
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    print(f"\n{'=' * 64}")
    print(f"  {title}")
    print(f"{'=' * 64}")


def make_row(date, event_type, event_subtype, ticker, magnitude, direction, source, description):
    ts = pd.Timestamp(date)
    if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return {
        "date"         : ts.normalize(),
        "event_type"   : event_type,
        "event_subtype": event_subtype,
        "ticker"       : ticker,
        "magnitude"    : float(magnitude) if pd.notna(magnitude) else np.nan,
        "direction"    : int(direction),
        "source"       : source,
        "description"  : str(description),
    }


def to_parquet(rows: list, path: Path, label: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.set_index("date")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    size_kb = path.stat().st_size / 1024
    print(f"\n  Saved {label}: {len(df):,} events -> {path.relative_to(ROOT)} ({size_kb:.0f} KB)")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# PART A — FRED Universal Macro Events
# ══════════════════════════════════════════════════════════════════════════════

section("PART A — FRED Universal Macro Events")

FRED_SERIES = {
    "FEDFUNDS": {
        "subtype"          : "fed_rate_change",
        "change"           : "mom_bps",    # level diff * 100 -> basis points
        "direction_sign"   : -1,           # rate hike = negative for stocks
        "desc_tmpl"        : "Fed Funds Rate: {val:.2f}% (MoM {chg:+.0f} bps)",
    },
    "CPIAUCSL": {
        "subtype"          : "cpi_change",
        "change"           : "mom_pct",
        "direction_sign"   : -1,
        "desc_tmpl"        : "CPI: {val:.3f} (MoM {chg:+.3f}%)",
    },
    "UNRATE": {
        "subtype"          : "unemployment_change",
        "change"           : "mom_pp",     # percentage-point diff
        "direction_sign"   : -1,
        "desc_tmpl"        : "Unemployment: {val:.1f}% (MoM {chg:+.2f} pp)",
    },
    "GDP": {
        "subtype"          : "gdp_change",
        "change"           : "qoq_pct",
        "direction_sign"   : +1,
        "desc_tmpl"        : "GDP: ${val:.0f}B (QoQ {chg:+.2f}%)",
    },
}


def fetch_fred(series_id: str) -> pd.Series:
    url = FRED_BASE.format(series_id)
    r   = requests.get(url, timeout=20)
    r.raise_for_status()
    df  = pd.read_csv(
        io.StringIO(r.text),
        parse_dates=["observation_date"],
        index_col="observation_date",
    )
    df.columns = [series_id]
    df = df[df[series_id] != "."]
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
    return df[series_id].dropna().sort_index()


macro_rows = []

for series_id, cfg in FRED_SERIES.items():
    print(f"\n  Fetching {series_id} ...")
    try:
        s = fetch_fred(series_id)
        s = s[s.index >= FRED_START]

        if cfg["change"] == "mom_bps":
            chg = (s - s.shift(1)) * 100
        elif cfg["change"] == "mom_pct":
            chg = s.pct_change() * 100
        elif cfg["change"] == "mom_pp":
            chg = s - s.shift(1)
        elif cfg["change"] == "qoq_pct":
            chg = s.pct_change() * 100
        else:
            chg = s.diff()

        n = 0
        for date, val in s.items():
            c = chg.get(date, np.nan)
            if pd.isna(c) and date == s.index[0]:
                continue
            mag  = round(float(c), 6) if pd.notna(c) else np.nan
            dirn = int(np.sign(c) * cfg["direction_sign"]) if pd.notna(c) and c != 0 else 0
            try:
                desc = cfg["desc_tmpl"].format(val=val, chg=c if pd.notna(c) else 0)
            except Exception:
                desc = f"{series_id}: {val}"

            macro_rows.append(make_row(
                date=date, event_type="macro",
                event_subtype=cfg["subtype"], ticker="ALL",
                magnitude=mag, direction=dirn,
                source="fred", description=desc,
            ))
            n += 1

        print(f"  {series_id}: {n} observations  "
              f"({s.index.min().date()} to {s.index.max().date()})")
    except Exception as exc:
        print(f"  WARNING: {series_id} failed: {exc}")

macro_df = to_parquet(macro_rows, UNIVERSAL_DIR / "macro_events.parquet", "macro events")


# ══════════════════════════════════════════════════════════════════════════════
# PART B — FDA Biotech Events (openFDA drugsfda)
# ══════════════════════════════════════════════════════════════════════════════

section("PART B — FDA Biotech Events (openFDA)")

print("""
  Source: openFDA drug/drugsfda endpoint (public, no API key required)
  Collecting: drug approval (AP), complete response letters (CR),
              tentative approvals (TA), withdrawals (WD)
  Focus: original NDA/BLA submissions and major efficacy supplements
""")

BIOTECH_TICKERS = [t for t in TICKER_SECTOR if TICKER_SECTOR[t] == "biotech"]


def openfda_get(params: dict, retries: int = 3) -> dict:
    """GET openFDA with retry on transient errors."""
    for attempt in range(retries):
        try:
            r = requests.get(OPENFDA_BASE, params=params, timeout=30)
            if r.status_code == 404:
                return {"results": [], "meta": {"results": {"total": 0}}}
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as exc:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise exc


def fetch_company_applications(ticker: str, sponsor_terms: list) -> list:
    """
    Fetch all drug applications for a company from openFDA.
    Returns a flat list of (date, drug_name, status, submission_type) tuples.
    """
    all_apps = []
    seen_apps = set()

    for term in sponsor_terms:
        skip = 0
        while True:
            params = {
                "search" : f'sponsor_name:"{term}"',
                "limit"  : OPENFDA_LIMIT,
                "skip"   : skip,
            }
            try:
                data  = openfda_get(params)
                total = data.get("meta", {}).get("results", {}).get("total", 0)
                batch = data.get("results", [])

                if not batch:
                    break

                for app in batch:
                    app_num = app.get("application_number", "")
                    if app_num in seen_apps:
                        continue
                    seen_apps.add(app_num)

                    # Extract product name (first product listed)
                    products     = app.get("products", [])
                    drug_name    = products[0].get("brand_name", "Unknown") if products else "Unknown"
                    if not drug_name or drug_name == "Unknown":
                        drug_name = products[0].get("active_ingredients", [{}])[0].get("name", "Unknown") \
                                    if products else "Unknown"

                    sponsor      = app.get("sponsor_name", term)
                    app_type     = app.get("application_number", "")[:3]  # NDA, BLA, ANDA

                    # Walk through submissions looking for meaningful actions
                    for sub in app.get("submissions", []):
                        status      = sub.get("submission_status", "")
                        status_date = sub.get("submission_status_date", "")
                        sub_type    = sub.get("submission_type", "")    # ORIG, SUPPL
                        sub_class   = sub.get("submission_class_code_description", "")
                        pdufa_date  = sub.get("pdufa_date", "")

                        if status not in FDA_STATUS_MAP:
                            continue

                        # Only keep original submissions and efficacy/safety supplements
                        # (skip labeling-only, manufacturing supplements)
                        is_efficacy_suppl = any(
                            kw in (sub_class or "").upper()
                            for kw in ["EFFICACY", "SAFETY", "NEW INDICATION",
                                       "NEW PATIENT POPULATION", "PRIORITY"]
                        )
                        if sub_type not in ("ORIG",) and not is_efficacy_suppl:
                            continue

                        if not status_date or len(status_date) < 8:
                            continue

                        try:
                            action_date = pd.to_datetime(status_date, format="%Y%m%d")
                        except Exception:
                            continue

                        all_apps.append({
                            "ticker"     : ticker,
                            "date"       : action_date,
                            "drug_name"  : drug_name,
                            "app_number" : app_num,
                            "app_type"   : app_type,   # NDA/BLA/ANDA
                            "sub_type"   : sub_type,   # ORIG/SUPPL
                            "status"     : status,
                            "sponsor"    : sponsor,
                            "pdufa_date" : pdufa_date,
                        })

                skip += len(batch)
                time.sleep(OPENFDA_DELAY)

                if skip >= total or skip >= 5000:   # cap at 5000 apps per company
                    break

            except Exception as exc:
                print(f"    WARNING: openFDA error for '{term}' (skip={skip}): {exc}")
                break

    return all_apps


fda_rows  = []
fda_stats = {}   # ticker -> {approved: n, rejected: n, total: n}

for ticker in BIOTECH_TICKERS:
    cfg      = BIOTECH_SPONSORS.get(ticker, {})
    terms    = cfg.get("terms", [])
    name     = cfg.get("name", ticker)

    print(f"\n  [{ticker}] {name}")

    ticker_rows_before = len(fda_rows)

    # ── openFDA (CDER) ────────────────────────────────────────────────────────
    if terms:
        print(f"    Searching openFDA: {terms}")
        try:
            apps = fetch_company_applications(ticker, terms)

            approved  = sum(1 for a in apps if a["status"] == "AP")
            crl       = sum(1 for a in apps if a["status"] in ("CR", "CRL"))
            tentative = sum(1 for a in apps if a["status"] == "TA")
            other     = len(apps) - approved - crl - tentative

            print(f"    Applications found  : {len(apps)}")
            print(f"    Approvals (AP)      : {approved}")
            print(f"    Complete resp (CR)  : {crl}")
            print(f"    Tentative (TA)      : {tentative}")
            print(f"    Other               : {other}")

            if apps:
                dates = [a["date"] for a in apps]
                print(f"    Date range          : {min(dates).date()} to {max(dates).date()}")

            for app in apps:
                status     = app["status"]
                status_cfg = FDA_STATUS_MAP[status]
                drug       = app["drug_name"]
                app_num    = app["app_number"]
                sub_type   = app["sub_type"]
                pdufa      = app.get("pdufa_date", "")
                pdufa_note = f" (PDUFA: {pdufa})" if pdufa else ""
                label_type = "original" if sub_type == "ORIG" else "supplement"
                fda_rows.append(make_row(
                    date          = app["date"],
                    event_type    = "fda_action",
                    event_subtype = status_cfg["label"],
                    ticker        = ticker,
                    magnitude     = np.nan,
                    direction     = status_cfg["direction"],
                    source        = "openfda",
                    description   = (
                        f"{ticker} FDA {status_cfg['label'].replace('_',' ')} | "
                        f"{drug} ({app_num}, {label_type}){pdufa_note}"
                    ),
                ))
        except Exception as exc:
            print(f"    ERROR (openFDA): {exc}")
    else:
        print(f"    Skipping openFDA (CBER products — not in CDER database)")

    # ── Hardcoded events fallback (MRNA and any future CBER tickers) ──────────
    hc_key = f"{ticker}_HARDCODED_EVENTS"
    hc_events = {
        "MRNA": MRNA_HARDCODED_EVENTS,
    }.get(ticker, [])

    if hc_events:
        print(f"    Adding {len(hc_events)} hardcoded FDA actions ...")
        for date_str, subtype, dirn, desc in hc_events:
            fda_rows.append(make_row(
                date=date_str, event_type="fda_action",
                event_subtype=subtype, ticker=ticker,
                magnitude=np.nan, direction=dirn,
                source="hardcoded", description=desc,
            ))

    # ── Per-ticker stats ──────────────────────────────────────────────────────
    ticker_events = fda_rows[ticker_rows_before:]
    n_approved = sum(1 for r in ticker_events if r["direction"] ==  1)
    n_crl      = sum(1 for r in ticker_events if r["direction"] == -1)
    fda_stats[ticker] = {"total": len(ticker_events), "approved": n_approved, "crl": n_crl}

fda_df = to_parquet(fda_rows, BIOTECH_DIR / "fda_events.parquet", "FDA biotech events")


# ══════════════════════════════════════════════════════════════════════════════
# PART C — Credit Spread (BAMLH0A0HYM2) for Financials sector
# ══════════════════════════════════════════════════════════════════════════════

section("PART C -- BAMLH0A0HYM2 Credit Spread (Financials)")

print("""
  Source: FRED BAMLH0A0HYM2 -- ICE BofA US High Yield Index Option-Adjusted Spread
  Units:  percent (OAS, effectively basis points / 100)
  Coverage: 1996-12-31 onwards. Pre-1996 dates filled with sentinel 0.0 in 05_event_features.py.
  Saved to: data/events/financials/credit_spreads.parquet
""")

cs_path = FINANCIALS_DIR / "credit_spreads.parquet"

try:
    print("  Fetching BAMLH0A0HYM2 ...", end=" ", flush=True)
    cs_raw = fetch_fred("BAMLH0A0HYM2")
    cs_raw = cs_raw[cs_raw.index >= "1993-01-01"]

    # Forward-fill weekends/holidays to get a contiguous daily series
    full_idx = pd.date_range(cs_raw.index[0], cs_raw.index[-1], freq="D")
    cs_daily = cs_raw.reindex(full_idx).ffill().dropna()
    cs_daily.index.name = "date"

    # Build a single-column DataFrame that 05_event_features.py can read
    cs_df = pd.DataFrame({"credit_spread_level": cs_daily})
    cs_df.index = pd.to_datetime(cs_df.index).normalize()
    cs_df.index.name = "date"

    FINANCIALS_DIR.mkdir(parents=True, exist_ok=True)
    cs_df.to_parquet(cs_path, engine="pyarrow")
    size_kb = cs_path.stat().st_size / 1024
    print(f"OK  ({len(cs_df):,} obs, {cs_df.index[0].date()} to {cs_df.index[-1].date()})")
    print(f"\n  Saved credit_spreads: {len(cs_df):,} daily rows -> "
          f"{cs_path.relative_to(ROOT)} ({size_kb:.0f} KB)")
    print(f"  First valid date: {cs_df.index[0].date()}  "
          f"(rows before 1996-12-31 are pre-coverage, sentinel applied in step 5)")
except Exception as exc:
    print(f"FAILED: {exc}")
    print("  [WARN] credit_spreads.parquet not saved -- financials features will use sentinel 0.0")


# ══════════════════════════════════════════════════════════════════════════════
# PART D -- Energy commodity data (WTI, Natural Gas, Rig Count)
# ══════════════════════════════════════════════════════════════════════════════

section("PART D -- Energy Commodities (WTI, Natural Gas, Rig Count)")

print("""
  Sources (FRED):
    DCOILWTICO : WTI crude oil price     -- daily, 1986-01-02+
    DHHNGSP    : Henry Hub natural gas   -- daily, 1997-01-07+
  Coverage notes:
    natgas pre-1997 : NaN sentinel 0.0 applied in 05_event_features.py
  Saved to: data/events/energy/energy_events.parquet  (wide-format daily DataFrame)
""")

energy_path = ENERGY_DIR / "energy_events.parquet"
energy_ok = False

try:
    # ── WTI crude oil (DCOILWTICO) ─────────────────────────────────────────
    print("  Fetching DCOILWTICO (WTI crude) ...", end=" ", flush=True)
    wti_raw = fetch_fred("DCOILWTICO")
    wti_raw = wti_raw[wti_raw.index >= "1986-01-01"]
    print(f"OK  ({len(wti_raw):,} obs, {wti_raw.index[0].date()} to {wti_raw.index[-1].date()})")

    # ── Natural gas (DHHNGSP) ──────────────────────────────────────────────
    print("  Fetching DHHNGSP (natural gas) ...", end=" ", flush=True)
    natgas_raw = fetch_fred("DHHNGSP")
    natgas_raw = natgas_raw[natgas_raw.index >= "1993-01-01"]
    print(f"OK  ({len(natgas_raw):,} obs, {natgas_raw.index[0].date()} to {natgas_raw.index[-1].date()})")

    # ── Build contiguous daily index covering both series ─────────────────
    date_start = min(wti_raw.index[0], natgas_raw.index[0])
    date_end   = max(wti_raw.index[-1], natgas_raw.index[-1])
    full_idx   = pd.date_range(date_start, date_end, freq="D")
    full_idx.name = "date"

    # Reindex and forward-fill weekends/holidays
    wti_daily    = wti_raw.reindex(full_idx).ffill()
    natgas_daily = natgas_raw.reindex(full_idx).ffill()

    # ── Assemble wide-format DataFrame ─────────────────────────────────────
    energy_df = pd.DataFrame({
        "wti_price":    wti_daily,
        "natgas_price": natgas_daily,
    }, index=full_idx)
    energy_df.index = pd.to_datetime(energy_df.index).normalize()
    energy_df.index.name = "date"

    # Derived features computed here so step 5 just does a merge + sentinel fill
    # WTI changes (NaN for early rows; step 5 fills with 0.0 where price is NaN)
    energy_df["wti_change_1w"]  = energy_df["wti_price"].pct_change(5)   * 100
    energy_df["wti_change_1m"]  = energy_df["wti_price"].pct_change(21)  * 100

    # Natgas changes (pre-1997 rows will be NaN; sentinel applied in step 5)
    energy_df["natgas_change_1w"] = energy_df["natgas_price"].pct_change(5)  * 100
    energy_df["natgas_change_1m"] = energy_df["natgas_price"].pct_change(21) * 100

    # WTI 63-day z-score (rolling)
    wti_roll_mean = energy_df["wti_price"].rolling(63, min_periods=20).mean()
    wti_roll_std  = energy_df["wti_price"].rolling(63, min_periods=20).std()
    energy_df["wti_zscore_63d"] = (energy_df["wti_price"] - wti_roll_mean) / wti_roll_std.replace(0, np.nan)

    ENERGY_DIR.mkdir(parents=True, exist_ok=True)
    energy_df.to_parquet(energy_path, engine="pyarrow")
    size_kb = energy_path.stat().st_size / 1024
    n_valid_wti    = energy_df["wti_price"].notna().sum()
    n_valid_natgas = energy_df["natgas_price"].notna().sum()
    print(f"\n  Saved energy_events: {len(energy_df):,} daily rows -> "
          f"{energy_path.relative_to(ROOT)} ({size_kb:.0f} KB)")
    print(f"  WTI valid rows    : {n_valid_wti:,}  (from {wti_raw.index[0].date()})")
    print(f"  NatGas valid rows : {n_valid_natgas:,}  (from {natgas_raw.index[0].date()})")
    print(f"  Columns: {list(energy_df.columns)}")
    energy_ok = True
except Exception as exc:
    print(f"FAILED: {exc}")
    print("  [WARN] energy_events.parquet not saved -- energy features will use sentinel 0.0")


# ══════════════════════════════════════════════════════════════════════════════
# PART E -- Consumer Staples event data (Retail Sales + Consumer Sentiment)
# ══════════════════════════════════════════════════════════════════════════════

section("PART E -- Consumer Staples Events (Retail Sales + Consumer Sentiment)")

print("""
  Sources (FRED):
    RSXFS   : Advance Retail Sales ex food services -- monthly, 1992-01+
    UMCSENT : Univ. of Michigan Consumer Sentiment  -- monthly, 1978-01+
  Coverage notes:
    retail_sales pre-1992 : NaN sentinel 0.0 applied in 05_event_features.py
    UMCSENT has full history back to 1978; no sentinel needed
  Saved to: data/events/consumer_staples/consumer_staples_events.parquet  (wide-format daily)
""")

cs_staples_path = CONSUMER_STAPLES_DIR / "consumer_staples_events.parquet"
cs_staples_ok   = False

try:
    # ── Retail Sales (RSXFS) ───────────────────────────────────────────────
    print("  Fetching RSXFS (advance retail sales ex food services) ...", end=" ", flush=True)
    rsxfs_raw = fetch_fred("RSXFS")
    rsxfs_raw = rsxfs_raw[rsxfs_raw.index >= "1992-01-01"]
    print(f"OK  ({len(rsxfs_raw):,} obs, {rsxfs_raw.index[0].date()} to {rsxfs_raw.index[-1].date()})")

    # ── Consumer Sentiment (UMCSENT) ───────────────────────────────────────
    print("  Fetching UMCSENT (consumer sentiment) ...", end=" ", flush=True)
    umcsent_raw = fetch_fred("UMCSENT")
    umcsent_raw = umcsent_raw[umcsent_raw.index >= "1978-01-01"]
    print(f"OK  ({len(umcsent_raw):,} obs, {umcsent_raw.index[0].date()} to {umcsent_raw.index[-1].date()})")

    # ── Build contiguous daily index covering both series ─────────────────
    date_start = min(rsxfs_raw.index[0], umcsent_raw.index[0])
    date_end   = max(rsxfs_raw.index[-1], umcsent_raw.index[-1])
    full_idx   = pd.date_range(date_start, date_end, freq="D")
    full_idx.name = "date"

    # Monthly series forward-filled to daily (each day sees most recent release)
    rsxfs_daily  = rsxfs_raw.reindex(full_idx).ffill()
    umcsent_daily = umcsent_raw.reindex(full_idx).ffill()

    # ── Derived features ───────────────────────────────────────────────────
    # Retail sales mom change (month-over-month pct change, 21-day approx)
    rsxfs_mom = rsxfs_daily.pct_change(21) * 100

    # Retail sales 3-month z-score (63-day rolling)
    rs_roll_mean = rsxfs_daily.rolling(63, min_periods=21).mean()
    rs_roll_std  = rsxfs_daily.rolling(63, min_periods=21).std()
    rsxfs_zscore = (rsxfs_daily - rs_roll_mean) / rs_roll_std.replace(0, np.nan)

    # Consumer sentiment 3-month change (63-day diff of monthly-ffilled level)
    umcsent_chg3m = umcsent_daily.diff(63)

    # ── Assemble wide-format DataFrame ─────────────────────────────────────
    cs_staples_df = pd.DataFrame({
        "retail_sales_level":     rsxfs_daily,
        "retail_sales_mom_change": rsxfs_mom,
        "retail_sales_zscore_3m": rsxfs_zscore,
        "consumer_sentiment_level":   umcsent_daily,
        "consumer_sentiment_change_3m": umcsent_chg3m,
    }, index=full_idx)
    cs_staples_df.index = pd.to_datetime(cs_staples_df.index).normalize()
    cs_staples_df.index.name = "date"

    CONSUMER_STAPLES_DIR.mkdir(parents=True, exist_ok=True)
    cs_staples_df.to_parquet(cs_staples_path, engine="pyarrow")
    size_kb = cs_staples_path.stat().st_size / 1024
    print(f"\n  Saved consumer_staples_events: {len(cs_staples_df):,} daily rows -> "
          f"{cs_staples_path.relative_to(ROOT)} ({size_kb:.0f} KB)")
    print(f"  Retail sales valid rows   : {rsxfs_daily.notna().sum():,}  "
          f"(from {rsxfs_raw.index[0].date()})")
    print(f"  Sentiment valid rows      : {umcsent_daily.notna().sum():,}  "
          f"(from {umcsent_raw.index[0].date()})")
    print(f"  Columns: {list(cs_staples_df.columns)}")
    cs_staples_ok = True
except Exception as exc:
    print(f"FAILED: {exc}")
    print("  [WARN] consumer_staples_events.parquet not saved -- "
          "consumer staples features will use sentinel 0.0")


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════

section("SUMMARY")

print(f"\n  PART A — Universal macro events")
print(f"  {'Series':<20} {'Count':>6}  Date range")
print(f"  {'-'*55}")
for subtype, grp in macro_df.groupby("event_subtype"):
    print(f"  {subtype:<20} {len(grp):>6}  "
          f"{grp.index.min().date()} to {grp.index.max().date()}")

print(f"\n  PART B — FDA biotech events")
print(f"  {'Ticker':<8} {'Name':<28} {'Total':>6}  {'Approved':>9}  {'CRL':>5}")
print(f"  {'-'*60}")
for ticker in BIOTECH_TICKERS:
    st   = fda_stats.get(ticker, {})
    name = BIOTECH_SPONSORS.get(ticker, {}).get("name", ticker)
    print(f"  {ticker:<8} {name:<28} {st.get('total',0):>6}  "
          f"{st.get('approved',0):>9}  {st.get('crl',0):>5}")

if not fda_df.empty:
    print(f"\n  FDA events by subtype:")
    for subtype, grp in fda_df.groupby("event_subtype"):
        print(f"    {subtype:<35} {len(grp):>4} events")

    print(f"\n  FDA events by ticker:")
    for ticker, grp in fda_df.groupby("ticker"):
        date_range = f"{grp.index.min().date()} to {grp.index.max().date()}"
        print(f"    {ticker:<6} {len(grp):>4} events  {date_range}")

print(f"\n  Files written:")
print(f"    {(UNIVERSAL_DIR / 'macro_events.parquet').relative_to(ROOT)}")
print(f"    {(BIOTECH_DIR / 'fda_events.parquet').relative_to(ROOT)}")
if cs_path.exists():
    print(f"    {cs_path.relative_to(ROOT)}")
if energy_ok and energy_path.exists():
    print(f"    {energy_path.relative_to(ROOT)}")
if cs_staples_ok and cs_staples_path.exists():
    print(f"    {cs_staples_path.relative_to(ROOT)}")
print(f"\nStep 4 complete. Next step: src/pipeline/05_event_features.py\n")

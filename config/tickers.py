"""
Ticker registry — single source of truth for all sectors and tickers.
Add new sectors/tickers here; the pipeline reads this at runtime.
"""

SECTORS = {
    "tech": {
        "tickers": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META",
                    "AMD", "TSLA", "CRM", "ADBE", "INTC", "ORCL"],
        "description": "Large cap US technology",
        "universal_events": ["fed_rate", "cpi", "gdp", "unemployment"],
        "sector_events": ["ai_narrative", "semiconductor_cycle", "cloud_growth", "antitrust"],
    },
    "biotech": {
        "tickers": ["LLY", "MRNA", "BIIB", "REGN", "VRTX",
                    "ABBV", "BMY", "GILD", "AMGN", "PFE"],
        "description": "Large cap biotech and pharma",
        "universal_events": ["fed_rate", "cpi"],
        "sector_events": ["fda_pdufa", "clinical_trials", "drug_approvals"],
    }
}

# Sector each ticker belongs to
TICKER_SECTOR = {t: s for s, v in SECTORS.items() for t in v["tickers"]}

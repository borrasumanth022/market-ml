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
    },
    "financials": {
        "tickers": ["JPM", "GS", "BAC", "MS", "WFC"],
        "description": "Large cap US banks and investment banks",
        "universal_events": ["fed_rate", "yield_curve", "credit_spreads"],
        "sector_events": [],
    },
    "energy": {
        "tickers": ["XOM", "CVX", "COP", "SLB", "EOG"],
        "description": "Large cap US energy companies",
        "universal_events": ["fed_rate", "oil_price", "natural_gas", "rig_count"],
        "sector_events": ["oil_price", "natural_gas", "rig_count"],
    },
    "consumer_staples": {
        "tickers": ["KO", "PG", "WMT", "COST", "CL"],
        "description": "Large cap US consumer staples",
        "universal_events": ["fed_rate", "cpi", "retail_sales", "consumer_confidence"],
        "sector_events": ["retail_sales", "consumer_confidence"],
    },
}

# Sector each ticker belongs to
TICKER_SECTOR = {t: s for s, v in SECTORS.items() for t in v["tickers"]}

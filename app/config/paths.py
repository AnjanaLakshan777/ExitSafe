"""Filesystem locations used across the application."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"              # files exactly as obtained; never edited
PROCESSED_DIR = DATA_DIR / "processed"  # derived / canonical outputs
SAMPLE_DIR = DATA_DIR / "sample"        # small synthetic fixtures (tracked in git)

# Market data
CSE_RAW_DIR = RAW_DIR / "cse"
CSE_MARKET_RAW_DIR = CSE_RAW_DIR / "market"
CSE_COMPANY_RAW_DIR = CSE_RAW_DIR / "company"
CSE_CORPORATE_ACTIONS_RAW_DIR = CSE_RAW_DIR / "corporate_actions"
CSE_ANNOUNCEMENTS_RAW_DIR = CSE_RAW_DIR / "announcements"
EXTERNAL_RAW_DIR = RAW_DIR / "external"  # secondary / third-party datasets
CSE_PROCESSED_DIR = PROCESSED_DIR / "cse"
AUDIT_DIR = PROCESSED_DIR / "audit"      # data-source audit reports and manifests

# Intelligence data
INTELLIGENCE_RAW_DIR = RAW_DIR / "intelligence"                        # collected items, as received
INTELLIGENCE_PROCESSED_DIR = PROCESSED_DIR / "intelligence"            # normalized MarketEvents
INTELLIGENCE_HISTORICAL_DIR = INTELLIGENCE_PROCESSED_DIR / "historical"  # archived events for backtesting

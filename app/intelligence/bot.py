"""ExitSafe market-threat bot.

    python -m app.intelligence.bot              # run forever: scan news every BOT_INTERVAL_MINUTES,
                                                # update the tracked CSV once a day after the CSE close
    python -m app.intelligence.bot --once       # one news scan + one price update, then exit
    python -m app.intelligence.bot --news-only  # skip price updates
    python -m app.intelligence.bot --prices-only --once

Settings come from the project's .env file (see .env.example).
"""

import argparse
import logging
import time
from datetime import datetime

from app.intelligence.collectors.price_collector import COLOMBO
from app.intelligence.price_updater import TrackedCsvError, update_tracked_csv
from app.intelligence.settings import load_settings
from app.intelligence.threat_scan import run_threat_scan

log = logging.getLogger("exitsafe.bot")
# CSE closes at 14:30 Colombo time; prices are fetched from this hour on.
PRICE_UPDATE_HOUR = 15


def scan_news(settings):
    result = run_threat_scan(settings)
    log.info("News scan: %d items, %d new threats, %d emailed",
             result.items_collected, len(result.new_threats), len(result.alerted))
    for record in result.new_threats:
        log.info("  [%s] %s (%s)", record.event.severity.value, record.event.title,
                 record.event.source_name)
    for message in result.errors:
        log.warning("  %s", message)
    for message in result.notes:
        log.info("  %s", message)


def update_prices(settings):
    """Returns True if the update ran (whether or not there were new rows)."""
    try:
        result = update_tracked_csv(settings)
    except TrackedCsvError as exc:
        log.warning("Price update skipped: %s", exc)
        return False
    log.info("Price update: %d row(s) added to %s", len(result.added), result.csv_path)
    for quote in result.added:
        log.info("  %s %s close=%s (%s)", quote.symbol, quote.day, quote.close, quote.source)
    for symbol, reason in sorted(result.skipped.items()):
        log.info("  %s skipped: %s", symbol, reason)
    for message in result.errors:
        log.warning("  %s", message)
    return not result.errors


def main(argv=None):
    parser = argparse.ArgumentParser(description="ExitSafe world-market threat bot")
    parser.add_argument("--once", action="store_true", help="run one cycle and exit")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--news-only", action="store_true", help="don't update prices")
    group.add_argument("--prices-only", action="store_true", help="don't scan news")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    settings = load_settings()
    log.info("Gemini: %s | Email alerts: %s | Alert level: %s+ | Tracked CSV: %s",
             "on" if settings.gemini_configured else "off (no GEMINI_API_KEY)",
             "on" if settings.email_configured else "off (SMTP settings missing)",
             settings.alert_min_severity.value, settings.tracked_csv)

    prices_done_for = None
    while True:
        if not args.prices_only:
            scan_news(settings)
        if not args.news_only:
            today = datetime.now(COLOMBO).date()
            after_close = datetime.now(COLOMBO).hour >= PRICE_UPDATE_HOUR
            if args.once or (after_close and prices_done_for != today):
                if update_prices(settings):
                    prices_done_for = today
        if args.once:
            return
        log.info("Next check in %d minutes (Ctrl+C to stop)", settings.interval_minutes)
        time.sleep(settings.interval_minutes * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Stopped")

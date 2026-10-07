"""Run one tracked-market-data update, then exit. Meant for cron or a platform's scheduled job.

    python -m app.scheduler.update_market_data

Safe to run as often as you like (for example every 30 minutes from 15:00 to 18:00 Sri
Lanka time on weekdays): nothing is collected while the CSE session is open, only one
update runs at a time, and stored rows are never added twice. Settings and database
connection come from the environment / .env, like the dashboard.

Exit code 0: finished, nothing to do, or still waiting for the close.
Exit code 1: a source or the database failed; the next run will try again.
"""

import logging
import sys

from app.intelligence.market_tracking import (ALREADY_RUNNING, WAITING_FOR_MARKET_CLOSE,
                                              run_scheduled_update)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    result = run_scheduled_update()
    for message in result.messages:
        logging.getLogger("exitsafe.tracking").info("%s", message)
    waiting = result.status in (WAITING_FOR_MARKET_CLOSE, ALREADY_RUNNING)
    return 1 if result.retry_later and not waiting else 0


if __name__ == "__main__":
    sys.exit(main())

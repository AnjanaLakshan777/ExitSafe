"""Create the client database (if missing) and its tables (accounts and tracked market data).

Usage: python scripts/init_client_db.py
Reads DB_NAME/DB_USER/DB_PASSWORD (or dbname/user/password) from .env.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import create_engine, inspect  # noqa: E402

from app.data.repositories.client_repository import (ClientRepository, database_url,  # noqa: E402
                                                     ensure_database)
from app.data.repositories.tracking_repository import TrackingRepository  # noqa: E402


def main():
    url = database_url()
    created = ensure_database(url)
    print(f"Database {url.database!r} {'created' if created else 'already exists'}")
    engine = create_engine(url)
    ClientRepository(engine).create_tables()
    TrackingRepository(engine).create_tables()
    print("Tables:", ", ".join(inspect(engine).get_table_names()))
    engine.dispose()


if __name__ == "__main__":
    main()

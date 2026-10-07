"""Tracked market data, stored in the same PostgreSQL database as the client accounts.

- ``tracked_datasets``: a client's tracked dataset (one per uploaded file) and its update status.
- ``market_data_uploads``: the original uploaded file, byte for byte. Never changed.
- ``tracked_symbols``: the companies a dataset follows and how current each one is.
- ``market_observations``: daily prices collected after each CSE session. Shared by all
  clients (one row per symbol, date and source), so the same CSE price isn't stored twice.
- ``market_data_update_runs`` / ``tracked_dataset_updates``: what each update did.

Everything a client sees goes through a method that takes their ``client_id``;
the update job uses the unscoped methods at the bottom.
"""

import threading
from contextlib import contextmanager
from datetime import date, datetime, timezone

from sqlalchemy import (JSON, Boolean, Date, DateTime, Float, ForeignKey, Integer, LargeBinary,
                        String, Text, UniqueConstraint, select, text, update)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Mapped, Session, mapped_column, relationship, selectinload

from app.data.repositories.client_repository import Base, ClientRepository, _utcnow

NOT_YET_UPDATED = "NOT_YET_UPDATED"
RUNNING = "RUNNING"
INTERRUPTED = "INTERRUPTED"
# Any number will do; it only has to be the same for every process sharing the database.
UPDATE_LOCK_KEY = 4_455_120_901
_local_lock = threading.Lock()


def aware(value):
    """SQLite drops the time zone; stored times are always UTC."""
    if isinstance(value, datetime) and value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


class TrackedDataset(Base):
    __tablename__ = "tracked_datasets"

    id: Mapped[int] = mapped_column(primary_key=True)
    client_id: Mapped[int] = mapped_column(ForeignKey("clients.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(120))
    purpose: Mapped[str | None] = mapped_column(String(32))      # START_INVESTING / ALREADY_INVESTED
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    price_sources: Mapped[str] = mapped_column(String(64))       # e.g. "cse,gemini"
    symbol_override: Mapped[str | None] = mapped_column(String(32))  # file had no Symbol column
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    last_attempted_update: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_successful_update: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_status: Mapped[str] = mapped_column(String(32), default=NOT_YET_UPDATED)
    last_message: Mapped[str | None] = mapped_column(Text)
    last_run_id: Mapped[int | None] = mapped_column(Integer)

    upload: Mapped["MarketDataUpload"] = relationship(
        back_populates="dataset", cascade="all, delete-orphan", uselist=False)
    symbols: Mapped[list["TrackedSymbol"]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan", order_by="TrackedSymbol.symbol")


class MarketDataUpload(Base):
    __tablename__ = "market_data_uploads"

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("tracked_datasets.id", ondelete="CASCADE"), unique=True)
    file_name: Mapped[str] = mapped_column(String(255))
    content: Mapped[bytes] = mapped_column(LargeBinary, deferred=True)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    size_bytes: Mapped[int] = mapped_column(Integer)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    rows_read: Mapped[int] = mapped_column(Integer)
    valid_rows: Mapped[int] = mapped_column(Integer)
    date_min: Mapped[date | None] = mapped_column(Date)
    date_max: Mapped[date | None] = mapped_column(Date)
    validation_status: Mapped[str] = mapped_column(String(16))
    validation_summary: Mapped[dict] = mapped_column(JSON)
    provenance: Mapped[dict] = mapped_column(JSON)

    dataset: Mapped[TrackedDataset] = relationship(back_populates="upload")


class TrackedSymbol(Base):
    __tablename__ = "tracked_symbols"
    __table_args__ = (UniqueConstraint("dataset_id", "symbol"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("tracked_datasets.id", ondelete="CASCADE"), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    upload_last_date: Mapped[date | None] = mapped_column(Date)   # newest day in the upload
    last_market_date: Mapped[date | None] = mapped_column(Date)
    last_source: Mapped[str | None] = mapped_column(String(64))
    last_update_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_status: Mapped[str | None] = mapped_column(String(32))
    last_message: Mapped[str | None] = mapped_column(Text)

    dataset: Mapped[TrackedDataset] = relationship(back_populates="symbols")


class MarketObservation(Base):
    """One collected daily price, with the canonical market-data fields and its source."""
    __tablename__ = "market_observations"
    __table_args__ = (UniqueConstraint("symbol", "date", "source"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    date: Mapped[date] = mapped_column(Date)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)
    change_pct: Mapped[float | None] = mapped_column(Float)
    turnover: Mapped[float | None] = mapped_column(Float)
    estimated_traded_value: Mapped[float | None] = mapped_column(Float)
    trades: Mapped[float | None] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(64))
    source_priority: Mapped[int] = mapped_column(Integer)
    source_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    source_url: Mapped[str | None] = mapped_column(Text)
    validation_status: Mapped[str] = mapped_column(String(16))
    validation_warnings: Mapped[str] = mapped_column(Text, default="")
    collected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    update_run_id: Mapped[int | None] = mapped_column(Integer)


class MarketDataUpdateRun(Base):
    __tablename__ = "market_data_update_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    trigger: Mapped[str] = mapped_column(String(16))          # scheduled / manual
    client_id: Mapped[int | None] = mapped_column(Integer)    # set for a client's own update
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(32), default=RUNNING)
    session_date: Mapped[date | None] = mapped_column(Date)
    symbols_processed: Mapped[int] = mapped_column(Integer, default=0)
    rows_added: Mapped[int] = mapped_column(Integer, default=0)
    rows_skipped: Mapped[int] = mapped_column(Integer, default=0)
    gemini_used: Mapped[bool] = mapped_column(Boolean, default=False)
    details: Mapped[dict | None] = mapped_column(JSON)


class TrackedDatasetUpdate(Base):
    __tablename__ = "tracked_dataset_updates"

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("tracked_datasets.id", ondelete="CASCADE"), index=True)
    run_id: Mapped[int | None] = mapped_column(Integer)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    trigger: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(32))
    rows_added: Mapped[int] = mapped_column(Integer, default=0)
    message: Mapped[str | None] = mapped_column(Text)
    details: Mapped[list | None] = mapped_column(JSON)


class TrackingRepository:
    def __init__(self, engine=None):
        self.engine = engine if engine is not None else ClientRepository().engine

    def create_tables(self):
        Base.metadata.create_all(self.engine)

    def _session(self):
        return Session(self.engine, expire_on_commit=False)

    # --- client-facing: always scoped to the logged-in client ---

    def create_dataset(self, client_id, *, name, purpose, price_sources, symbol_override,
                       upload, symbols):
        """Store a new tracked dataset; ``upload`` holds the MarketDataUpload fields and
        ``symbols`` maps each symbol to the newest date in the upload."""
        with self._session() as session:
            dataset = TrackedDataset(client_id=client_id, name=name, purpose=purpose,
                                     price_sources=price_sources, symbol_override=symbol_override,
                                     upload=MarketDataUpload(**upload),
                                     symbols=[TrackedSymbol(symbol=s, upload_last_date=d,
                                                            last_market_date=d)
                                              for s, d in sorted(symbols.items())])
            session.add(dataset)
            session.commit()
            return dataset.id

    def find_by_upload(self, client_id, sha256):
        with self._session() as session:
            return session.scalar(select(TrackedDataset).join(MarketDataUpload)
                                  .where(TrackedDataset.client_id == client_id,
                                         MarketDataUpload.sha256 == sha256))

    def datasets_for(self, client_id):
        with self._session() as session:
            return list(session.scalars(
                select(TrackedDataset).where(TrackedDataset.client_id == client_id)
                .options(selectinload(TrackedDataset.symbols),
                         selectinload(TrackedDataset.upload))
                .order_by(TrackedDataset.id)))

    def dataset_for(self, client_id, dataset_id):
        """The client's dataset, or None if it doesn't exist or belongs to someone else."""
        with self._session() as session:
            return session.scalar(
                select(TrackedDataset).where(TrackedDataset.id == dataset_id,
                                             TrackedDataset.client_id == client_id)
                .options(selectinload(TrackedDataset.symbols),
                         selectinload(TrackedDataset.upload)))

    def upload_content(self, client_id, dataset_id):
        """The original uploaded bytes, or None if the dataset isn't the client's."""
        with self._session() as session:
            return session.scalar(
                select(MarketDataUpload.content).join(TrackedDataset)
                .where(TrackedDataset.id == dataset_id, TrackedDataset.client_id == client_id))

    def set_active(self, client_id, dataset_id, active):
        """Pause or resume a dataset. False if it isn't the client's."""
        with self._session() as session:
            changed = session.execute(
                update(TrackedDataset).where(TrackedDataset.id == dataset_id,
                                             TrackedDataset.client_id == client_id)
                .values(active=bool(active))).rowcount
            session.commit()
            return bool(changed)

    def history_for(self, client_id, dataset_id, limit=10):
        with self._session() as session:
            return list(session.scalars(
                select(TrackedDatasetUpdate).join(TrackedDataset)
                .where(TrackedDataset.id == dataset_id, TrackedDataset.client_id == client_id)
                .order_by(TrackedDatasetUpdate.id.desc()).limit(limit)))

    def observations_for(self, symbols):
        """Collected observations for these symbols (shared market data), oldest first."""
        if not symbols:
            return []
        with self._session() as session:
            return list(session.scalars(
                select(MarketObservation).where(MarketObservation.symbol.in_(sorted(symbols)))
                .order_by(MarketObservation.symbol, MarketObservation.date,
                          MarketObservation.source_priority, MarketObservation.id)))

    def last_run(self, trigger=None):
        with self._session() as session:
            query = select(MarketDataUpdateRun).order_by(MarketDataUpdateRun.id.desc()).limit(1)
            if trigger:
                query = query.where(MarketDataUpdateRun.trigger == trigger)
            return session.scalar(query)

    # --- used by the update job ---

    @contextmanager
    def update_lock(self):
        """Yields True if this process may run an update, False if another one is running.

        PostgreSQL advisory locks cover every process and server using the database.
        """
        if self.engine.dialect.name != "postgresql":
            got = _local_lock.acquire(blocking=False)
            try:
                yield got
            finally:
                if got:
                    _local_lock.release()
            return
        with self.engine.connect() as conn:
            got = bool(conn.execute(text("SELECT pg_try_advisory_lock(:key)"),
                                    {"key": UPDATE_LOCK_KEY}).scalar())
            try:
                yield got
            finally:
                if got:
                    conn.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": UPDATE_LOCK_KEY})
                conn.commit()

    def active_tracked_symbols(self, client_id=None):
        """Active symbols of active datasets (one client's, or everyone's)."""
        with self._session() as session:
            query = (select(TrackedSymbol).join(TrackedDataset)
                     .where(TrackedDataset.active.is_(True), TrackedSymbol.active.is_(True))
                     .options(selectinload(TrackedSymbol.dataset))
                     .order_by(TrackedSymbol.dataset_id, TrackedSymbol.symbol))
            if client_id is not None:
                query = query.where(TrackedDataset.client_id == client_id)
            return list(session.scalars(query))

    def start_run(self, trigger, client_id=None):
        """Open a run record. Runs still marked RUNNING were cut off (the lock is free now)."""
        with self._session() as session:
            session.execute(update(MarketDataUpdateRun)
                            .where(MarketDataUpdateRun.status == RUNNING)
                            .values(status=INTERRUPTED))
            run = MarketDataUpdateRun(trigger=trigger, client_id=client_id)
            session.add(run)
            session.commit()
            return run.id

    def finish_run(self, run_id, **values):
        with self._session() as session:
            session.execute(update(MarketDataUpdateRun).where(MarketDataUpdateRun.id == run_id)
                            .values(finished_at=_utcnow(), **values))
            session.commit()

    def has_observation(self, symbol, day, source=None):
        with self._session() as session:
            query = select(MarketObservation.id).where(MarketObservation.symbol == symbol,
                                                       MarketObservation.date == day)
            if source is not None:
                query = query.where(MarketObservation.source == source)
            return session.scalar(query.limit(1)) is not None

    def preferred_source(self, symbol, day):
        """Source of the most trusted stored row for this symbol and day, or None."""
        with self._session() as session:
            return session.scalar(
                select(MarketObservation.source)
                .where(MarketObservation.symbol == symbol, MarketObservation.date == day)
                .order_by(MarketObservation.source_priority, MarketObservation.id).limit(1))

    def add_observation(self, **values):
        """Insert one observation in its own transaction. False if it was already stored."""
        with self._session() as session:
            session.add(MarketObservation(**values))
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                return False
            return True

    def record_dataset_update(self, dataset_id, run_id, trigger, *, status, message, rows_added,
                              details, symbol_updates, successful, now):
        """Save one dataset's outcome (status, per-symbol state, history) in one transaction."""
        with self._session() as session:
            values = {"last_attempted_update": now, "last_status": status,
                      "last_message": message, "last_run_id": run_id}
            if successful:
                values["last_successful_update"] = now
            session.execute(update(TrackedDataset).where(TrackedDataset.id == dataset_id)
                            .values(**values))
            for symbol_id, changes in symbol_updates.items():
                session.execute(update(TrackedSymbol).where(TrackedSymbol.id == symbol_id)
                                .values(last_update_at=now, **changes))
            session.add(TrackedDatasetUpdate(dataset_id=dataset_id, run_id=run_id, at=now,
                                             trigger=trigger, status=status,
                                             rows_added=rows_added, message=message,
                                             details=details))
            session.commit()

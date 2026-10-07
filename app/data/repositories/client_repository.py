"""Client accounts and their investments, stored in PostgreSQL.

Connection settings come from the project ``.env`` file (see ``.env.example``):
``DB_NAME``/``DB_USER``/``DB_PASSWORD`` (or the short ``dbname``/``user``/``password``),
plus optional ``DB_HOST`` and ``DB_PORT``. Passwords are stored only as salted
scrypt hashes, never as plain text.
"""

import hashlib
import hmac
import os
import secrets
from datetime import date, datetime, timezone
from decimal import Decimal

from dotenv import dotenv_values
from sqlalchemy import (Date, DateTime, ForeignKey, Numeric, String, create_engine, select,
                        text)
from sqlalchemy.engine import URL
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship

from app.config.paths import PROJECT_ROOT

ENV_FILE = PROJECT_ROOT / ".env"

_SCRYPT = {"n": 2 ** 14, "r": 8, "p": 1, "dklen": 64}


def _utcnow():
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Client(Base):
    __tablename__ = "clients"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    full_name: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    investments: Mapped[list["Investment"]] = relationship(
        back_populates="client", cascade="all, delete-orphan")


class Investment(Base):
    __tablename__ = "investments"

    id: Mapped[int] = mapped_column(primary_key=True)
    client_id: Mapped[int] = mapped_column(ForeignKey("clients.id", ondelete="CASCADE"), index=True)
    symbol: Mapped[str] = mapped_column(String(32))
    quantity: Mapped[Decimal] = mapped_column(Numeric(18, 4))
    purchase_price: Mapped[Decimal] = mapped_column(Numeric(18, 4))
    purchase_date: Mapped[date | None] = mapped_column(Date)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    client: Mapped[Client] = relationship(back_populates="investments")

    @property
    def amount(self):
        return self.quantity * self.purchase_price


def hash_password(password):
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, **_SCRYPT)
    return f"scrypt${salt.hex()}${digest.hex()}"


def verify_password(password, stored):
    try:
        scheme, salt, digest = stored.split("$")
    except ValueError:
        return False
    if scheme != "scrypt":
        return False
    candidate = hashlib.scrypt(password.encode("utf-8"), salt=bytes.fromhex(salt), **_SCRYPT)
    return hmac.compare_digest(candidate.hex(), digest)


def database_url(env=None, database=None):
    """PostgreSQL URL from ``env`` (defaults to the ``.env`` file, then the process environment).

    The ``.env`` file wins over the process environment so the short ``user`` key
    is not shadowed by the shell's own ``USER`` variable.
    """
    if env is None:
        env = {**os.environ, **{k: v for k, v in dotenv_values(ENV_FILE).items() if v}}

    def pick(*keys, default=None):
        for key in keys:
            if env.get(key) and env[key].strip():
                return env[key].strip()
        return default

    name = database or pick("DB_NAME", "dbname")
    if not name:
        raise ValueError("Set DB_NAME (or dbname) in .env")
    return URL.create("postgresql+psycopg2",
                      username=pick("DB_USER", "user", default="postgres"),
                      password=pick("DB_PASSWORD", "password"),
                      host=pick("DB_HOST", "host", default="localhost"),
                      port=int(pick("DB_PORT", "port", default="5432")),
                      database=name)


def ensure_database(url):
    """Create the database named in ``url`` if it does not exist yet."""
    admin = create_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            exists = conn.execute(text("SELECT 1 FROM pg_database WHERE datname = :n"),
                                  {"n": url.database}).scalar()
            if not exists:
                conn.execute(text(f'CREATE DATABASE "{url.database.replace(chr(34), "")}"'))
    finally:
        admin.dispose()
    return not exists


class ClientRepository:
    def __init__(self, engine=None):
        self.engine = engine if engine is not None else create_engine(database_url())

    def create_tables(self):
        Base.metadata.create_all(self.engine)

    def add_client(self, email, password, full_name=None):
        with Session(self.engine, expire_on_commit=False) as session:
            client = Client(email=email.strip().lower(), password_hash=hash_password(password),
                            full_name=full_name)
            session.add(client)
            session.commit()
            return client

    def get_client(self, email):
        with Session(self.engine) as session:
            return session.scalar(select(Client).where(Client.email == email.strip().lower()))

    def authenticate(self, email, password):
        """The client if the password matches, else None."""
        client = self.get_client(email)
        if client and verify_password(password, client.password_hash):
            return client
        return None

    def add_investment(self, client_id, symbol, quantity, purchase_price, purchase_date=None):
        with Session(self.engine, expire_on_commit=False) as session:
            investment = Investment(client_id=client_id, symbol=symbol.strip().upper(),
                                    quantity=Decimal(str(quantity)),
                                    purchase_price=Decimal(str(purchase_price)),
                                    purchase_date=purchase_date)
            session.add(investment)
            session.commit()
            return investment

    def investments_for(self, client_id):
        with Session(self.engine) as session:
            return list(session.scalars(select(Investment)
                                        .where(Investment.client_id == client_id)
                                        .order_by(Investment.id)))

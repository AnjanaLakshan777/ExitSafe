"""Market-threat bot settings, read from environment variables or the project ``.env`` file.

Secrets (Gemini key, email password) are only ever read from the environment;
they are never written to disk by the bot. See ``.env.example`` for every option.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from app.config.paths import CSE_PROCESSED_DIR, INTELLIGENCE_PROCESSED_DIR, PROJECT_ROOT
from app.data.schemas.event_schema import Severity

ENV_FILE = PROJECT_ROOT / ".env"

THREATS_FILE = INTELLIGENCE_PROCESSED_DIR / "threats.jsonl"
PRICE_LOG_FILE = INTELLIGENCE_PROCESSED_DIR / "price_updates.jsonl"
BOT_STATUS_FILE = INTELLIGENCE_PROCESSED_DIR / "bot_status.json"
# Copy of the user's CSV that the bot appends daily prices to. Kept under
# processed/ because data/raw/ holds files exactly as obtained.
DEFAULT_TRACKED_CSV = CSE_PROCESSED_DIR / "tracked_market_data.csv"

# Free-tier Gemini model; change GEMINI_MODEL if Google retires it.
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
PRICE_SOURCES = ("cse", "gemini")


@dataclass(frozen=True)
class BotSettings:
    gemini_api_key: str | None = None
    gemini_model: str = DEFAULT_GEMINI_MODEL
    use_web_search: bool = True

    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str | None = None
    smtp_password: str | None = None
    alert_email_to: tuple[str, ...] = ()

    interval_minutes: int = 30
    lookback_hours: int = 24
    alert_min_severity: Severity = Severity.HIGH

    tracked_csv: Path = DEFAULT_TRACKED_CSV
    tracked_symbol: str | None = None     # only for a CSV without a Symbol column
    price_sources: tuple[str, ...] = PRICE_SOURCES

    @property
    def email_configured(self):
        return bool(self.smtp_user and self.smtp_password and self.alert_email_to)

    @property
    def gemini_configured(self):
        return bool(self.gemini_api_key)


def _flag(value, default):
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int(value, default, name):
    if value is None or not value.strip():
        return default
    try:
        number = int(value)
    except ValueError:
        raise ValueError(f"{name} must be a whole number, got {value!r}") from None
    if number <= 0:
        raise ValueError(f"{name} must be positive, got {number}")
    return number


def _text(value):
    return value.strip() if value and value.strip() else None


def load_settings(env=None):
    """Build settings from ``env`` (defaults to ``os.environ`` after loading ``.env``)."""
    if env is None:
        load_dotenv(ENV_FILE)
        env = os.environ

    severity = (env.get("ALERT_MIN_SEVERITY") or Severity.HIGH.value).strip().upper()
    sources = tuple(s.strip().lower() for s in (env.get("PRICE_SOURCES") or "cse,gemini").split(",")
                    if s.strip())
    unknown = [s for s in sources if s not in PRICE_SOURCES]
    if unknown:
        raise ValueError(f"Unknown PRICE_SOURCES {unknown}; use any of {list(PRICE_SOURCES)}")
    recipients = tuple(a.strip() for a in (env.get("ALERT_EMAIL_TO") or "").split(",") if a.strip())
    csv_path = _text(env.get("TRACKED_CSV_PATH"))

    return BotSettings(
        gemini_api_key=_text(env.get("GEMINI_API_KEY")),
        gemini_model=_text(env.get("GEMINI_MODEL")) or DEFAULT_GEMINI_MODEL,
        use_web_search=_flag(env.get("USE_WEB_SEARCH"), True),
        smtp_host=_text(env.get("SMTP_HOST")) or "smtp.gmail.com",
        smtp_port=_int(env.get("SMTP_PORT"), 587, "SMTP_PORT"),
        smtp_user=_text(env.get("SMTP_USER")),
        smtp_password=_text(env.get("SMTP_PASSWORD")),
        alert_email_to=recipients,
        interval_minutes=_int(env.get("BOT_INTERVAL_MINUTES"), 30, "BOT_INTERVAL_MINUTES"),
        lookback_hours=_int(env.get("LOOKBACK_HOURS"), 24, "LOOKBACK_HOURS"),
        alert_min_severity=Severity(severity),
        tracked_csv=Path(csv_path) if csv_path else DEFAULT_TRACKED_CSV,
        tracked_symbol=_text(env.get("TRACKED_SYMBOL")),
        price_sources=sources,
    )

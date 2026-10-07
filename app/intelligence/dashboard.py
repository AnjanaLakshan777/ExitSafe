"""Streamlit panels for the market-threat bot: threat alerts and the tracked price CSV."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st

from app.data.schemas.event_schema import Severity
from app.data.source_catalog import DataSourceType, get_source
from app.intelligence.classification.threat_keywords import SEVERITY_ORDER, severity_rank
from app.intelligence.price_updater import TrackedCsvError, save_tracked_csv, update_tracked_csv
from app.intelligence.settings import load_settings
from app.intelligence.threat_scan import read_status, run_threat_scan
from app.intelligence.threat_store import ThreatStore

SEVERITY_ICONS = {Severity.CRITICAL: "🔴", Severity.HIGH: "🟠", Severity.MEDIUM: "🟡",
                  Severity.LOW: "⚪"}
RECENT_HOURS = 24


def source_label(source_name):
    """Readable label for a price row's catalog source."""
    source = get_source(source_name)
    if source.is_ai_generated:
        return f"Secondary AI-sourced ({source_name})"
    if source.source_type is DataSourceType.OFFICIAL_CSE:
        return f"Official CSE ({source_name})"
    return source_name


def threats_table(records):
    return pd.DataFrame([{
        "Severity": f"{SEVERITY_ICONS[r.event.severity]} {r.event.severity.value}",
        "Headline": r.event.title,
        "Source": r.event.source_name,
        "Status": r.event.verification_status.value,
        "Published (UTC)": r.event.published_time.strftime("%Y-%m-%d %H:%M"),
        "Triggered by": ", ".join(r.threat_terms),
        "Link": r.event.source_url,
    } for r in records])


def _show_status(status):
    if status is None:
        st.caption("No scan has run yet. Click **Scan now**, or start the bot with "
                   "`python -m app.intelligence.bot`.")
        return
    when = datetime.fromisoformat(status["last_scan"]).astimezone().strftime("%Y-%m-%d %H:%M")
    st.caption(f"Last scan {when}: {status['items_collected']} stories read, "
               f"{status['new_threats']} new threat(s), {status['emailed']} emailed.")
    for note in status.get("notes", []):
        st.caption(f"ℹ️ {note}")
    if status.get("errors"):
        with st.expander(f"{len(status['errors'])} source(s) could not be read"):
            for error in status["errors"]:
                st.text(error)


def show_threat_panel():
    """Recent world-market threats found by the bot, with a manual scan button."""
    settings = load_settings()
    store = ThreatStore()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=RECENT_HOURS)
    serious = [r for r in store.recent(min_severity=settings.alert_min_severity)
               if r.event.published_time >= cutoff]
    label = (f"🌍 World Market Threats — {len(serious)} at {settings.alert_min_severity.value}+ "
             f"in the last {RECENT_HOURS}h" if serious else "🌍 World Market Threats")

    with st.expander(label, expanded=bool(serious)):
        if st.button("Scan now", key="threat_scan",
                     help="Read the news feeds (and Gemini web search, if a key is set) now"):
            try:
                with st.spinner("Reading news sources..."):
                    result = run_threat_scan(settings, store=store)
                st.success(f"{result.items_collected} stories read, "
                           f"{len(result.new_threats)} new threat(s), "
                           f"{len(result.alerted)} emailed.")
            except Exception as exc:  # noqa: BLE001 - show it instead of a traceback
                st.error(f"The scan could not finish: {type(exc).__name__}: {exc}")
        _show_status(read_status())

        levels = [s.value for s in SEVERITY_ORDER]
        minimum = st.select_slider("Show severity at least", levels,
                                   value=settings.alert_min_severity.value, key="threat_min")
        records = [r for r in store.recent(limit=200)
                   if severity_rank(r.event.severity) >= severity_rank(minimum)]
        if not records:
            st.info("No threats stored at this level.")
        else:
            st.dataframe(threats_table(records), hide_index=True,
                         column_config={"Link": st.column_config.LinkColumn("Link",
                                                                            display_text="open")})
        st.caption("Found by keyword matching on headlines. News and search results are "
                   "unconfirmed claims; rows from Gemini web search are external, "
                   "lower-credibility search results. Check the source before acting.")


def show_tracked_csv_panel(upload_name=None, upload_content=None):
    """Save the current upload as the bot's tracked CSV, and update its prices on demand."""
    settings = load_settings()
    path = settings.tracked_csv
    with st.expander("📈 Daily price updates (tracked CSV)"):
        st.caption(f"The bot appends each trading day's prices to `{path}` — official CSE "
                   "data first, Gemini web search only for symbols CSE doesn't list. Gemini "
                   "prices are secondary AI-sourced data: they're marked in the file's Source "
                   "column and left out of quantitative analysis by default.")
        if upload_content is not None and st.button(f"Track `{upload_name}` for daily updates",
                                                    key="track_csv"):
            save_tracked_csv(upload_content, path)
            st.success("Saved. Choose **Bot-tracked CSV** above to analyse the updated file.")
        if not path.exists():
            st.info("No tracked CSV yet. Upload your market-data CSV above, then click "
                    "**Track ... for daily updates**.")
            return
        if st.button("Update prices now", key="update_prices"):
            with st.spinner("Fetching prices..."):
                try:
                    result = update_tracked_csv(settings)
                except TrackedCsvError as exc:
                    st.error(str(exc))
                    return
                except Exception as exc:  # noqa: BLE001 - show it instead of a traceback
                    st.error(f"The price update could not finish: {type(exc).__name__}: {exc}")
                    return
            st.success(f"{len(result.added)} row(s) added.")
            if result.added:
                st.dataframe(pd.DataFrame([{"Symbol": q.symbol, "Date": q.day, "Close": q.close,
                                            "Volume": q.volume, "Source": source_label(q.source)}
                                           for q in result.added]), hide_index=True)
                ai_rows = sum(1 for q in result.added if get_source(q.source).is_ai_generated)
                if ai_rows:
                    st.warning(f"{ai_rows} row(s) came from Gemini web search. They are secondary "
                               "AI-sourced prices, not exchange data, and are left out of "
                               "quantitative analysis by default.")
            for symbol, reason in sorted(result.skipped.items()):
                st.caption(f"{symbol}: {reason}")
            for error in result.errors:
                st.warning(error)

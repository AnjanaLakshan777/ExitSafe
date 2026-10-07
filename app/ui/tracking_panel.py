"""Tracked market data in the dashboard: start tracking an upload, see how current it is,
update it now, or pause it. Everything is looked up with the logged-in client's id."""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
from sqlalchemy.exc import SQLAlchemyError

from app.data.repositories.tracking_repository import TrackingRepository, aware
from app.intelligence import market_tracking as tracking
from app.intelligence.collectors.price_collector import COLOMBO
from app.intelligence.settings import load_settings
from app.ui import auth

PURPOSE_LABELS = {tracking.START_INVESTING: "Start investing (companies I'm considering)",
                  tracking.ALREADY_INVESTED: "I already invested (companies I hold)"}
DATABASE_ERROR = ("Tracked market data is unavailable because the database can't be reached. "
                  "Please try again later.")
_ready = {}   # id(engine) -> engine whose tables were created


def repository():
    engine = auth._repository().engine
    repo = TrackingRepository(engine)
    if _ready.get(id(engine)) is not engine:
        repo.create_tables()
        _ready[id(engine)] = engine
    return repo


def when(value):
    """A stored UTC time in Sri Lanka time."""
    if value is None:
        return "never"
    return aware(value).astimezone(COLOMBO).strftime("%d %b %Y, %H:%M") + " (Sri Lanka time)"


def _status_label(status):
    return tracking.STATUS_LABELS.get(status, status.replace("_", " ").capitalize())


def has_tracked_data(client):
    """True if the client tracks any market data (False when the database can't be reached)."""
    try:
        return bool(repository().datasets_for(client["id"]))
    except SQLAlchemyError:
        return False


def show_start_tracking(client, file_name, content, symbol, purpose=None):
    """Offer to keep the data just uploaded up to date."""
    with st.expander("📌 Keep this data up to date automatically"):
        st.caption("Upload once. ExitSafe saves this file to your account and adds each new "
                   "trading day from the official CSE after the market closes, so you don't "
                   "need to upload it again.")
        name = st.text_input("Name", value=Path(file_name).stem, key="track_name")
        options = list(PURPOSE_LABELS)
        purpose = st.radio("What is this data for?", options, format_func=PURPOSE_LABELS.get,
                           index=options.index(purpose) if purpose in options else 0,
                           key="track_purpose", horizontal=True)
        if not st.button("Start tracking", key="start_tracking"):
            return
        try:
            repo = repository()
            dataset = tracking.start_tracking(repo, client["id"], file_name, content, name=name,
                                              purpose=purpose, symbol=symbol)
        except tracking.TrackingSetupError as exc:
            st.error(str(exc))
            return
        except SQLAlchemyError:
            st.error(DATABASE_ERROR)
            return
        st.success("Your market data is now being tracked.")
        show_overview(repo, dataset, collected=0)
        st.caption("Next time, choose **My tracked market data** above. There's no need to "
                   "upload the file again.")


def show_overview(repo, dataset, collected):
    symbols = dataset.symbols
    dates = [s.last_market_date for s in symbols if s.last_market_date]
    if dataset.active:
        st.success("Market data is updated automatically.", icon="✅")
    else:
        st.info("Tracking is paused: this data isn't being updated.", icon="⏸️")
    left, middle, right, last = st.columns(4)
    left.metric("Companies tracked", len(symbols))
    middle.metric("Last market date", str(max(dates)) if dates else "n/a")
    right.metric("Daily observations", f"{dataset.upload.valid_rows + collected:,}")
    last.metric("Update status", _status_label(dataset.last_status))
    if dataset.last_message:
        st.write(dataset.last_message)

    sources = sorted({tracking.source_description(s.last_source) for s in symbols})
    st.caption(f"Tracked companies: {', '.join(s.symbol for s in symbols)}. "
               f"Data source: {' and '.join(sources)}. History comes from your file "
               f"“{dataset.upload.file_name}”.")
    st.caption(f"Last successful update: {when(dataset.last_successful_update)}.")
    if dataset.active:
        now = datetime.now(timezone.utc)
        due = tracking.next_update_time(now, aware(dataset.last_successful_update))
        st.caption("Next automatic update: " + ("due now (after today's close)." if due <= now
                                                 else f"{when(due)}, after the CSE close."))
        service = repo.last_run(trigger=tracking.SCHEDULED)
        if service is None:
            st.caption("ℹ️ The automatic update service hasn't run on this server yet. "
                       "You can use **Update now** in the meantime.")
        else:
            st.caption(f"The automatic update service last ran {when(service.started_at)}.")


def show_tracked_data(client):
    """Pick and show one of the client's tracked datasets. Returns its analysis input."""
    try:
        return _show_tracked_data(client)
    except SQLAlchemyError:
        st.error(DATABASE_ERROR)
        return None


def _show_tracked_data(client):
    repo = repository()
    datasets = repo.datasets_for(client["id"])
    if not datasets:
        st.info("You aren't tracking any market data yet. Choose **Upload or paste data**, "
                "upload your CSV, then click **Start tracking**.")
        return None
    names = {d.id: d.name + ("" if d.active else " (paused)") for d in datasets}
    chosen = st.selectbox("Tracked dataset", list(names), format_func=names.get,
                          key="tracked_dataset") if len(datasets) > 1 else datasets[0].id
    dataset = next(d for d in datasets if d.id == chosen)

    st.subheader("Tracked Market Data")
    if dataset.purpose:
        st.caption(PURPOSE_LABELS[dataset.purpose])
    update_now, toggle = st.columns(2)
    if update_now.button("Update now", key="tracking_update_now", disabled=not dataset.active):
        _update_now(repo, client, dataset)
        dataset = repo.dataset_for(client["id"], dataset.id)
    if toggle.button("Pause tracking" if dataset.active else "Resume tracking",
                     key="tracking_toggle"):
        repo.set_active(client["id"], dataset.id, not dataset.active)
        st.rerun()

    data = tracking.analysis_input(repo, client["id"], dataset.id)
    show_overview(repo, dataset, collected=len(data.collected))
    _show_details(repo, client, dataset, data)
    return data


def _update_now(repo, client, dataset):
    with st.spinner("Collecting the latest market data..."):
        result = tracking.run_tracking_update(repo, load_settings(), client_id=client["id"],
                                              trigger=tracking.MANUAL)
    outcome = result.datasets.get(dataset.id)
    if outcome is None:
        (st.error if result.status == tracking.FAILED else st.info)(" ".join(result.messages))
        return
    show = {tracking.UPDATED: st.success, tracking.PARTIALLY_UPDATED: st.warning,
            tracking.UPDATE_FAILED: st.error}.get(outcome.status, st.info)
    show(outcome.message)
    st.dataframe(pd.DataFrame([{"Company": o.symbol,
                                "Result": tracking.SYMBOL_LABELS.get(o.outcome, o.outcome),
                                "Note": o.message} for o in outcome.symbols]), hide_index=True)
    for message in result.messages:
        st.warning(message)


def _show_details(repo, client, dataset, data):
    with st.expander("🔎 Data details"):
        st.markdown("**Companies**")
        st.dataframe(pd.DataFrame([{
            "Company": s.symbol,
            "Last market date": s.last_market_date,
            "Source of latest day": tracking.source_description(s.last_source),
            "Last checked": when(s.last_update_at) if s.last_update_at else "not yet",
            "Result": tracking.SYMBOL_LABELS.get(s.last_status, "") if s.last_status else "",
            "Note": s.last_message or "",
        } for s in dataset.symbols]), hide_index=True)

        st.markdown("**Days added since your upload**")
        if data.collected:
            st.dataframe(pd.DataFrame([{
                "Company": r.symbol, "Date": r.date, "Close": r.close, "Volume": r.volume,
                "Source": tracking.source_description(r.source),
                "Used in analysis": ("No by default (secondary AI-sourced)"
                                     if tracking.is_ai_source(r.source) else "Yes"),
            } for r in data.collected]), hide_index=True)
        else:
            st.caption("None yet.")
        if data.superseded:
            st.caption(f"{len(data.superseded)} secondary price(s) are not used because an "
                       "official CSE price exists for the same day.")

        history = repo.history_for(client["id"], dataset.id)
        if history:
            st.markdown("**Recent updates**")
            st.dataframe(pd.DataFrame([{
                "When": when(h.at), "Started by": "You" if h.trigger == tracking.MANUAL
                else "Automatic", "Result": _status_label(h.status), "Days added": h.rows_added,
                "Note": h.message or ""} for h in history]), hide_index=True)

        upload = dataset.upload
        st.markdown("**Your original file**")
        st.caption(f"“{upload.file_name}”, saved {when(upload.uploaded_at)}: {upload.rows_read:,} "
                   f"rows from {upload.date_min} to {upload.date_max}, validation "
                   f"{upload.validation_status}. It is kept unchanged; new days are added "
                   f"alongside it. SHA-256 {upload.sha256[:12]}…")

"""Client login, registration and investments pages for the Streamlit console."""

import re
from datetime import date

import pandas as pd
import streamlit as st
from sqlalchemy.exc import IntegrityError, OperationalError, SQLAlchemyError

from app.data.repositories.client_repository import ClientRepository

SESSION_KEY = "client"
MIN_PASSWORD_LENGTH = 8
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def registration_errors(email, password, confirm):
    """Problems with a registration form, as messages (empty if it is valid)."""
    errors = []
    if not _EMAIL.match(email.strip()):
        errors.append("Enter a valid email address.")
    if len(password) < MIN_PASSWORD_LENGTH:
        errors.append(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    if password != confirm:
        errors.append("Passwords do not match.")
    return errors


@st.cache_resource
def _repository():
    repository = ClientRepository()
    repository.create_tables()
    return repository


def _session_client(client):
    return {"id": client.id, "email": client.email, "name": client.full_name or client.email}


def _show_login(repository):
    with st.form("login"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")
    if submitted:
        client = repository.authenticate(email, password)
        if client is None:
            st.error("Wrong email or password.")
        else:
            st.session_state[SESSION_KEY] = _session_client(client)
            st.rerun()


def _show_register(repository):
    with st.form("register"):
        name = st.text_input("Full name")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password",
                                 help=f"At least {MIN_PASSWORD_LENGTH} characters")
        confirm = st.text_input("Confirm password", type="password")
        submitted = st.form_submit_button("Create account")
    if not submitted:
        return
    errors = registration_errors(email, password, confirm)
    if errors:
        for error in errors:
            st.error(error)
        return
    try:
        client = repository.add_client(email, password, name.strip() or None)
    except IntegrityError:
        st.error("An account with this email already exists. Log in instead.")
        return
    st.session_state[SESSION_KEY] = _session_client(client)
    st.rerun()


def require_login():
    """The logged-in client (id, email, name), or None after showing the login/register page."""
    if SESSION_KEY in st.session_state:
        return st.session_state[SESSION_KEY]
    try:
        repository = _repository()
    except (OperationalError, ValueError) as error:
        st.error("Cannot connect to the client database. Check DB_NAME / DB_USER / DB_PASSWORD "
                 "in .env and run `python scripts/init_client_db.py`.")
        with st.expander("Technical details"):
            st.code(str(error))
        return None

    st.title("ExitSafe")
    login, register = st.tabs(["Log in", "Register"])
    with login:
        _show_login(repository)
    with register:
        _show_register(repository)
    return None


def show_account_panel(client):
    """Signed-in client's investments, an add-investment form and a log-out button."""
    repository = _repository()
    with st.sidebar:
        st.markdown(f"Signed in as **{client['name']}**")
        st.caption(client["email"])
        if st.button("Log out"):
            del st.session_state[SESSION_KEY]
            st.rerun()

    with st.expander("💼 My Investments"):
        with st.form("add_investment", clear_on_submit=True):
            symbol = st.text_input("Symbol", placeholder="e.g. JKH.N0000")
            quantity = st.number_input("Quantity", min_value=0.0, step=1.0)
            price = st.number_input("Purchase price (LKR)", min_value=0.0, step=0.01)
            bought = st.date_input("Purchase date", value=date.today(), max_value=date.today())
            submitted = st.form_submit_button("Add investment")
        if submitted:
            if not symbol.strip() or quantity <= 0 or price <= 0:
                st.error("Enter a symbol, and a quantity and price above zero.")
            else:
                try:
                    repository.add_investment(client["id"], symbol, quantity, price, bought)
                    st.success(f"Added {symbol.strip().upper()}.")
                except SQLAlchemyError as error:
                    st.error(f"Could not save the investment: {error}")

        investments = repository.investments_for(client["id"])
        if not investments:
            st.caption("No investments yet.")
            return
        table = pd.DataFrame([{
            "Symbol": i.symbol,
            "Quantity": float(i.quantity),
            "Purchase price": float(i.purchase_price),
            "Amount": float(i.amount),
            "Purchase date": i.purchase_date,
        } for i in investments])
        st.dataframe(table, hide_index=True)
        st.metric("Total invested (LKR)", f"{table['Amount'].sum():,.2f}")

import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from streamlit.testing.v1 import AppTest

import app.ui.auth as auth
from app.data.repositories.client_repository import ClientRepository


def test_registration_errors():
    assert auth.registration_errors("a@b.com", "longenough", "longenough") == []
    assert auth.registration_errors("not-an-email", "longenough", "longenough") == \
        ["Enter a valid email address."]
    assert len(auth.registration_errors("a@b.com", "short", "other")) == 2


@pytest.fixture
def repo(monkeypatch):
    engine = create_engine("sqlite://", poolclass=StaticPool,
                           connect_args={"check_same_thread": False})
    repository = ClientRepository(engine)
    repository.create_tables()
    monkeypatch.setattr(auth, "_repository", lambda: repository)
    return repository


def _page():
    def page():
        from app.ui.auth import require_login, show_account_panel
        client = require_login()
        if client is not None:
            show_account_panel(client)
    return AppTest.from_function(page).run()


def _fill(at, form, values):
    inputs = [t for t in at.text_input if t.form_id == form]
    for widget, value in zip(inputs, values):
        widget.input(value)
    next(b for b in at.button if b.form_id == form).click().run()


def test_register_logout_login(repo):
    at = _page()
    _fill(at, "register", ["Ann", "ann@example.com", "password1", "password1"])
    assert not at.exception
    assert at.session_state[auth.SESSION_KEY]["email"] == "ann@example.com"

    at.sidebar.button[0].click().run()
    assert auth.SESSION_KEY not in at.session_state

    _fill(at, "login", ["ann@example.com", "wrongpass"])
    assert "Wrong email or password." in [e.value for e in at.error]
    _fill(at, "login", ["ann@example.com", "password1"])
    assert at.session_state[auth.SESSION_KEY]["name"] == "Ann"


def test_duplicate_registration_is_refused(repo):
    repo.add_client("ann@example.com", "password1")
    at = _page()
    _fill(at, "register", ["Ann", "ann@example.com", "password1", "password1"])
    assert any("already exists" in e.value for e in at.error)
    assert auth.SESSION_KEY not in at.session_state

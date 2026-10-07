from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError

from app.data.repositories.client_repository import (ClientRepository, database_url,
                                                     hash_password, verify_password)


@pytest.fixture
def repo():
    repository = ClientRepository(create_engine("sqlite://"))
    repository.create_tables()
    return repository


def test_password_is_hashed_and_verifies():
    stored = hash_password("s3cret")
    assert "s3cret" not in stored
    assert verify_password("s3cret", stored)
    assert not verify_password("wrong", stored)
    assert hash_password("s3cret") != stored  # salted


def test_add_client_and_authenticate(repo):
    client = repo.add_client(" Ann@Example.com ", "pw123", "Ann")
    assert client.email == "ann@example.com"
    assert repo.authenticate("ann@example.com", "pw123").id == client.id
    assert repo.authenticate("ann@example.com", "nope") is None
    assert repo.authenticate("missing@example.com", "pw123") is None


def test_duplicate_email_rejected(repo):
    repo.add_client("a@b.com", "x")
    with pytest.raises(IntegrityError):
        repo.add_client("A@B.com", "y")


def test_investments(repo):
    client = repo.add_client("a@b.com", "x")
    repo.add_investment(client.id, "jkh.n0000", 100, 20.5, date(2026, 1, 2))
    [inv] = repo.investments_for(client.id)
    assert inv.symbol == "JKH.N0000"
    assert inv.amount == Decimal("2050.0000")


def test_database_url_reads_short_and_long_keys():
    url = database_url({"dbname": "exitsafe", "user": "me", "password": "p"})
    assert (url.database, url.username, url.password, url.host, url.port) == \
        ("exitsafe", "me", "p", "localhost", 5432)
    url = database_url({"DB_NAME": "a", "dbname": "b", "DB_PORT": "5433"})
    assert (url.database, url.port, url.username) == ("a", 5433, "postgres")
    with pytest.raises(ValueError):
        database_url({})

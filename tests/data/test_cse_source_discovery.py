"""Offline tests for the parsing and date rules in scripts/cse_source_discovery.py."""

import importlib.util
from datetime import date
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "cse_source_discovery.py"


@pytest.fixture(scope="module")
def discovery():
    spec = importlib.util.spec_from_file_location("cse_source_discovery", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def word(text, x0, top, width=None):
    return {"text": text, "x0": x0, "x1": x0 + (width or 5 * len(text)), "top": top}


def data_row(top, name_words, type_letter, values, group="BANKS", board="Main"):
    words = [word(group, 9, top), word(board, 90, top)]
    words += [word(text, x, top) for text, x in name_words]
    if type_letter:
        words.append(word(type_letter, 195, top, width=4))
    xs = [230, 272, 306, 358, 395, 450, 506, 569]
    words += [word(v, x, top) for v, x in zip(values, xs)]
    return words


VALUES = ["18.90", "18.90", "9/29/2026", "19.00", "18.80", "7,362,860,271", "91,223,023.1",
          "17,691,081,661"]


def test_listing_and_report_dates(discovery):
    assert discovery.listing_date("29-09-2026 Report") == date(2026, 9, 29)
    assert discovery.listing_date("smd full 13-02-2026") == date(2026, 2, 13)
    assert discovery.listing_date("no date here") is None
    assert discovery.new_report_date("Tuesday, 29 September, 2026") == date(2026, 9, 29)
    assert discovery.new_report_date("") is None


def test_equity_row_is_parsed(discovery):
    bounds = discovery.ColumnBounds(board=86.8, name=122.1, type=195.0)
    rows = discovery.equity_rows_from_words(data_row(170, [("JKH", 121)], "N", VALUES), bounds)

    assert rows == [{
        "industry_group": "BANKS", "board": "Main", "company_name": "JKH", "type": "N",
        "close_price": "18.90", "last_traded_price": "18.90", "date_last_traded": "9/29/2026",
        "high": "19.00", "low": "18.80", "foreign_holding": "7,362,860,271",
        "turnover_rs": "91,223,023.1", "quantity_in_cds": "17,691,081,661",
    }]


def test_wrapped_company_name_is_reassembled(discovery):
    bounds = discovery.ColumnBounds(board=86.8, name=122.1, type=195.0)
    words = [word("COMMERCIAL", 121, 163)] + data_row(170, [], "N", VALUES) + [word("BANK", 121, 177)]

    rows = discovery.equity_rows_from_words(words, bounds)

    assert rows[0]["company_name"] == "COMMERCIAL BANK"


def test_type_letter_fused_onto_name_is_split_only_inside_type_column(discovery):
    bounds = discovery.ColumnBounds(board=86.8, name=122.1, type=195.0)
    fused = data_row(170, [("OFFICE", 121)], None, VALUES) + [word("EQUIPMENTN", 148, 170, width=52)]
    rows = discovery.equity_rows_from_words(fused, bounds)
    assert (rows[0]["company_name"], rows[0]["type"]) == ("OFFICE EQUIPMENT", "N")

    # A name that merely ends in N but stays left of the type column is untouched.
    normal = data_row(170, [("DIALOGN", 121)], None, VALUES)
    assert "parse_error" in discovery.equity_rows_from_words(normal, bounds)[0]


def test_negative_values_and_page_header_date(discovery):
    bounds = discovery.ColumnBounds(board=86.8, name=122.1, type=195.0)
    values = VALUES[:5] + ["-89,343"] + VALUES[6:]
    words = [word("9/29/2026", 560, 58)] + data_row(170, [("HAYLEYS", 121)], "N", values)

    rows = discovery.equity_rows_from_words(words, bounds)

    assert len(rows) == 1  # the lone page-header date is not a data row
    assert rows[0]["foreign_holding"] == "-89,343"


def test_stale_rows_are_excluded_not_redated(discovery):
    parsed = {"2026-09-29": {"rows": [
        {"company_name": "A", "type": "N", "close_price": "1", "high": "1", "low": "1",
         "turnover_rs": "10", "date_last_traded": "9/29/2026"},
        {"company_name": "B", "type": "N", "close_price": "2", "high": "2", "low": "2",
         "turnover_rs": "20", "date_last_traded": "9/25/2026"},
    ]}}

    frame, stale = discovery.report_rows_to_canonical(parsed)

    assert stale == 1
    assert list(frame["symbol"]) == ["A [N]"]
    assert list(frame["date"]) == ["2026-09-29"]
    assert frame.loc[0, "open"] == "" and frame.loc[0, "volume"] == ""  # not provided, not invented


def test_chart_points_use_payload_timestamps(discovery):
    charts = {"JKH.N0000": {"points": [
        {"t": 1790620200000, "h": 19.0, "l": 18.8, "p": 18.9, "q": 4848018, "o": None}]}}

    frame = discovery.chart_points_to_canonical(charts)

    assert frame.loc[0, "date"] == "2026-09-29"  # 00:00 Asia/Colombo
    assert frame.loc[0, "open"] == ""
    assert frame.loc[0, "volume"] == "4848018"


def test_cross_check_compares_matching_rows(discovery):
    charts = {"JKH.N0000": {"points": [
        {"t": 1790620200000, "h": 19.0, "l": 18.8, "p": 18.9, "q": 4848018}]}}
    parsed = {"2026-09-29": {"rows": [
        {"company_name": "JKH", "type": "N", "close_price": "18.90", "high": "19.00",
         "low": "18.80", "turnover_rs": "91,223,023.1", "date_last_traded": "9/29/2026"}]}}

    [check] = discovery.cross_check_charts(charts, parsed)

    assert check["close_matches"] and check["high_matches"] and check["low_matches"]
    assert check["vwap_within_high_low"]


def test_listing_titles_with_underscores(discovery):
    assert discovery.listing_date("24_09_2026 Report") == date(2026, 9, 24)

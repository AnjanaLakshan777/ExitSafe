"""Smoke tests for scripts/data_source_audit.py (runs only on local files)."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "data_source_audit.py"


@pytest.fixture(scope="module")
def audit():
    spec = importlib.util.spec_from_file_location("data_source_audit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_audit_of_sample_file_writes_report_and_manifest(audit, tmp_path, capsys):
    sample = Path(__file__).resolve().parents[2] / "data" / "sample" / "sample_market_data.csv"

    exit_code = audit.main(["--file", str(sample), "--source", "exitsafe_sample",
                            "--output-dir", str(tmp_path)])

    report = (tmp_path / audit.REPORT_NAME).read_text(encoding="utf-8")
    assert exit_code == 0
    assert "DATA SOURCE AUDIT" in report
    assert "Rows:            75" in report
    assert "Start date:      2026-01-02" in report
    assert "Status:          WARNING" in report  # the synthetic sample has zero-volume price moves
    assert "NOT an official CSE source" in report
    assert "SOURCES NOT YET AVAILABLE LOCALLY" in report
    assert report in capsys.readouterr().out

    manifest = json.loads((tmp_path / "exitsafe_sample__sample_market_data.manifest.json")
                          .read_text(encoding="utf-8"))
    assert manifest["provenance"]["original_file_name"] == "sample_market_data.csv"
    assert manifest["validation"]["total_rows"] == 75


def test_audit_fails_on_unusable_file(audit, tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("symbol,date,close\nABC,2026-01-05,1\n", encoding="utf-8")

    exit_code = audit.main(["--file", str(bad), "--source", "hf_tharu_jwd_cse_market_data",
                            "--output-dir", str(tmp_path / "out")])

    report = (tmp_path / "out" / audit.REPORT_NAME).read_text(encoding="utf-8")
    assert exit_code == 1
    assert "Missing columns: open, high, low, volume" in report
    assert "Status:          FAIL" in report

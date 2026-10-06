"""Tests for app.analytics.covariance. Expected values come from hand calculations or the statistics module."""

import math
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.analytics.covariance import (
    calculate_aligned_return_matrix,
    calculate_annualized_covariance_matrix,
    calculate_correlation_matrix,
    calculate_covariance_matrix,
    calculate_return_matrix,
    describe_return_alignment,
)
from app.analytics.volatility import calculate_annualized_volatility
from app.data.loaders.csv_market_loader import load_csv_market_data

X = [0.01, 0.02, 0.03]
Y = [0.03, 0.01, 0.02]
SAMPLE_3_SYMBOLS = Path(__file__).resolve().parents[2] / "data" / "sample" / "sample_market_data.csv"


def prices(symbol, returns, start=100.0, dates=None, status=None):
    """Canonical rows whose consecutive closes produce exactly ``returns``."""
    closes = [start]
    for r in returns:
        closes.append(closes[-1] * (1 + r))
    frame = pd.DataFrame({
        "date": dates if dates is not None else pd.bdate_range("2025-01-01", periods=len(closes)),
        "symbol": symbol,
        "close": closes,
    })
    if status is not None:
        frame["validation_status"] = status
    return frame


def two_symbols(x=X, y=Y):
    return pd.concat([prices("ABC", x), prices("XYZ", y)], ignore_index=True)


# Known covariance, independent check

def test_known_returns_give_hand_calculated_covariance():
    cov = calculate_covariance_matrix(two_symbols())

    assert list(cov.index) == list(cov.columns) == ["ABC", "XYZ"]
    assert cov.loc["ABC", "XYZ"] == pytest.approx(-0.00005, rel=1e-9)
    assert cov.loc["ABC", "ABC"] == pytest.approx(0.0001, rel=1e-9)
    assert cov.loc["XYZ", "XYZ"] == pytest.approx(0.0001, rel=1e-9)


def test_covariance_matches_independent_sample_covariance():
    series = {"AAA": [0.012, -0.004, 0.007, 0.001, -0.009, 0.015],
              "BBB": [0.003, 0.002, -0.006, 0.011, -0.002, 0.004],
              "CCC": [-0.010, 0.006, 0.009, -0.003, 0.008, -0.001]}
    data = pd.concat([prices(s, r) for s, r in series.items()], ignore_index=True)

    cov = calculate_covariance_matrix(data)

    for a in series:
        for b in series:
            assert cov.loc[a, b] == pytest.approx(statistics.covariance(series[a], series[b]),
                                                  rel=1e-9, abs=1e-15)


def test_sample_not_population_covariance():
    cov = calculate_covariance_matrix(two_symbols()).loc["ABC", "XYZ"]
    population = -0.0001 / 3
    assert cov != pytest.approx(population)
    assert cov == pytest.approx(population * 3 / 2)


# Annualization

def test_annualized_covariance_is_daily_times_252():
    annual = calculate_annualized_covariance_matrix(two_symbols())
    assert annual.loc["ABC", "XYZ"] == pytest.approx(-0.0126, rel=1e-9)
    assert annual.loc["ABC", "ABC"] == pytest.approx(0.0252, rel=1e-9)


def test_covariance_is_not_annualized_with_sqrt_252():
    daily = calculate_covariance_matrix(two_symbols()).loc["ABC", "XYZ"]
    annual = calculate_annualized_covariance_matrix(two_symbols()).loc["ABC", "XYZ"]

    assert annual == pytest.approx(daily * 252)
    assert annual != pytest.approx(daily * math.sqrt(252))


def test_volatility_uses_sqrt_252_while_covariance_uses_252():
    # Annualized variance (covariance diagonal * 252) must equal the square of
    # annualized volatility (daily volatility * sqrt(252)) for the same returns.
    data = prices("ABC", X)
    annual_variance = calculate_annualized_covariance_matrix(data).loc["ABC", "ABC"]
    annual_volatility = calculate_annualized_volatility(data).iloc[0]["annualized_volatility"]

    assert annual_volatility == pytest.approx(math.sqrt(statistics.variance(X)) * math.sqrt(252))
    assert annual_variance == pytest.approx(annual_volatility ** 2, rel=1e-12)


def test_custom_periods_per_year():
    annual = calculate_annualized_covariance_matrix(two_symbols(), periods_per_year=52)
    assert annual.loc["ABC", "XYZ"] == pytest.approx(-0.00005 * 52, rel=1e-9)


@pytest.mark.parametrize("bad", [0, -252, "252", None, True, float("nan")])
def test_invalid_periods_per_year_fails_clearly(bad):
    with pytest.raises(ValueError, match="periods_per_year"):
        calculate_annualized_covariance_matrix(two_symbols(), periods_per_year=bad)


# Correlation

def test_hand_calculated_correlation():
    corr = calculate_correlation_matrix(two_symbols())
    assert corr.loc["ABC", "XYZ"] == pytest.approx(-0.5, rel=1e-9)
    assert corr.loc["ABC", "XYZ"] == pytest.approx(statistics.correlation(X, Y), rel=1e-9)


def test_perfect_positive_correlation():
    x = [0.01, -0.02, 0.015, 0.004, -0.007]
    y = [2 * r + 0.001 for r in x]                  # exact positive linear relation
    corr = calculate_correlation_matrix(two_symbols(x, y))
    assert corr.loc["ABC", "XYZ"] == pytest.approx(1.0, abs=1e-9)


def test_perfect_negative_correlation():
    x = [0.01, -0.02, 0.015, 0.004, -0.007]
    y = [-r for r in x]
    corr = calculate_correlation_matrix(two_symbols(x, y))
    assert corr.loc["ABC", "XYZ"] == pytest.approx(-1.0, abs=1e-9)


def test_uncorrelated_series_give_zero():
    # Deviations are orthogonal: sum((x - mean)(y - mean)) = 0 exactly.
    x = [0.01, -0.01, 0.01, -0.01]
    y = [0.01, 0.01, -0.01, -0.01]
    assert statistics.correlation(x, y) == pytest.approx(0.0, abs=1e-12)
    assert calculate_correlation_matrix(two_symbols(x, y)).loc["ABC", "XYZ"] == pytest.approx(
        0.0, abs=1e-9)


def test_correlation_diagonal_is_one_and_values_in_range():
    rng = np.random.default_rng(11)
    data = pd.concat([prices(s, list(rng.normal(0, 0.02, 30))) for s in ["A", "B", "C", "D"]],
                     ignore_index=True)
    corr = calculate_correlation_matrix(data)

    assert list(np.diag(corr)) == [1.0, 1.0, 1.0, 1.0]
    assert ((corr >= -1.0) & (corr <= 1.0)).all().all()


# Symmetry and diagonal

def test_matrices_are_symmetric_and_diagonal_is_variance():
    series = {"AAA": [0.012, -0.004, 0.007, 0.001], "BBB": [0.003, 0.002, -0.006, 0.011],
              "CCC": [-0.010, 0.006, 0.009, -0.003]}
    data = pd.concat([prices(s, r) for s, r in series.items()], ignore_index=True)

    cov = calculate_covariance_matrix(data)
    corr = calculate_correlation_matrix(data)

    pd.testing.assert_frame_equal(cov, cov.T)
    pd.testing.assert_frame_equal(corr, corr.T)
    for symbol, returns in series.items():
        assert cov.loc[symbol, symbol] == pytest.approx(statistics.variance(returns), rel=1e-9)


def test_single_symbol_gives_one_by_one_matrices():
    cov = calculate_covariance_matrix(prices("ABC", X))
    corr = calculate_correlation_matrix(prices("ABC", X))
    assert cov.shape == corr.shape == (1, 1)
    assert cov.loc["ABC", "ABC"] == pytest.approx(statistics.variance(X), rel=1e-9)
    assert corr.loc["ABC", "ABC"] == 1.0


# No mixing, alignment, ordering

def test_symbols_are_not_mixed():
    # Very different price levels, rows shuffled together. Mixing series would
    # produce returns like 10/100 - 1 = -90%.
    abc, xyz = [0.01, -0.005, 0.02, 0.003], [0.004, 0.006, -0.01, 0.002]
    data = pd.concat([prices("ABC", abc, start=100), prices("XYZ", xyz, start=10)])
    data = data.sample(frac=1, random_state=3)

    cov = calculate_covariance_matrix(data)

    assert cov.loc["ABC", "XYZ"] == pytest.approx(statistics.covariance(abc, xyz), rel=1e-9)
    assert cov.loc["XYZ", "XYZ"] == pytest.approx(statistics.variance(xyz), rel=1e-9)


def test_unsorted_input_is_handled():
    data = two_symbols().iloc[[5, 2, 7, 0, 3, 6, 1, 4]]
    assert calculate_covariance_matrix(data).loc["ABC", "XYZ"] == pytest.approx(-0.00005, rel=1e-9)


def test_dates_are_aligned_and_periods_not_mixed():
    days = pd.bdate_range("2025-01-01", periods=6)          # d0 .. d5
    abc = pd.DataFrame({"date": days, "symbol": "ABC",
                        "close": [100, 101, 99, 102, 103, 101.0]})
    # XYZ has no row on d3, so its d4 return covers two days (d2 -> d4).
    xyz = pd.DataFrame({"date": days.delete(3), "symbol": "XYZ",
                        "close": [50, 51, 50.5, 52, 51.0]})

    info = describe_return_alignment(pd.concat([abc, xyz]))
    aligned = calculate_aligned_return_matrix(pd.concat([abc, xyz]))

    assert list(aligned.index) == [days[1], days[2], days[5]]     # d1, d2, d5
    assert info["observations"] == 3
    assert info["excluded_missing"] == 1        # d3: XYZ has no return
    assert info["excluded_misaligned"] == 1     # d4: XYZ's return is a two-day return
    expected_abc = [101 / 100 - 1, 99 / 101 - 1, 101 / 103 - 1]
    expected_xyz = [51 / 50 - 1, 50.5 / 51 - 1, 51 / 52 - 1]
    cov = calculate_covariance_matrix(pd.concat([abc, xyz]))
    assert cov.loc["ABC", "XYZ"] == pytest.approx(statistics.covariance(expected_abc, expected_xyz),
                                                  rel=1e-9)


def test_return_matrix_layout():
    matrix = calculate_return_matrix(two_symbols())
    assert list(matrix.columns) == ["ABC", "XYZ"]
    assert matrix.index.is_monotonic_increasing
    assert len(matrix) == 3                       # first close of each symbol gives no return
    assert matrix.iloc[0].tolist() == pytest.approx([0.01, 0.03])


# Invalid rows and missing observations

def test_invalid_rows_are_excluded_and_not_bridged():
    days = pd.bdate_range("2025-01-01", periods=6)
    abc = pd.DataFrame({"date": days, "symbol": "ABC", "close": [100, 101, 99, 5000, 103, 104.0],
                        "validation_status": ["VALID", "VALID", "VALID", "INVALID", "VALID",
                                              "WARNING"]})
    xyz = pd.DataFrame({"date": days, "symbol": "XYZ", "close": [50, 51, 50.5, 52, 51, 52.0],
                        "validation_status": "VALID"})
    data = pd.concat([abc, xyz])

    aligned = calculate_aligned_return_matrix(data)

    # d3 (into the INVALID row) and d4 (out of it) are both unusable for ABC;
    # no 99 -> 103 "bridged" return is created.
    assert list(aligned.index) == [days[1], days[2], days[5]]
    assert aligned["ABC"].tolist() == pytest.approx([0.01, 99 / 101 - 1, 104 / 103 - 1])
    assert 103 / 99 - 1 not in aligned["ABC"].tolist()
    assert aligned.abs().max().max() < 0.05                  # the 5000 close never leaks in


def test_missing_observations_are_reported():
    data = pd.concat([prices("ABC", [0.01, 0.02, -0.01, 0.005]), prices("NEW", [0.02])])
    info = describe_return_alignment(data)

    assert info["returns_per_symbol"] == {"ABC": 4, "NEW": 1}
    assert info["observations"] == 1
    assert info["excluded_missing"] == 3
    assert calculate_return_matrix(data)["NEW"].isna().sum() == 3


# Insufficient data and constant series

def test_insufficient_common_observations_give_nan_not_zero():
    data = pd.concat([prices("ABC", [0.01, 0.02, -0.01, 0.005]), prices("NEW", [0.02])])

    cov = calculate_covariance_matrix(data)
    corr = calculate_correlation_matrix(data)

    assert cov.isna().all().all()
    assert corr.isna().all().all()
    assert list(cov.index) == ["ABC", "NEW"]


def test_symbol_without_any_returns_gives_nan_matrix():
    data = pd.concat([prices("ABC", X), prices("ONE", [])])     # ONE has a single close
    assert describe_return_alignment(data)["observations"] == 0
    assert calculate_covariance_matrix(data).isna().all().all()


def test_constant_returns_are_handled_safely():
    flat = [0.0, 0.0, 0.0]
    data = pd.concat([prices("ABC", X), prices("FLAT", flat)], ignore_index=True)

    cov = calculate_covariance_matrix(data)
    corr = calculate_correlation_matrix(data)

    assert cov.loc["FLAT", "FLAT"] == 0.0              # variance of a constant is genuinely 0
    assert cov.loc["ABC", "FLAT"] == pytest.approx(0.0, abs=1e-18)
    assert math.isnan(corr.loc["FLAT", "FLAT"])        # correlation undefined
    assert math.isnan(corr.loc["ABC", "FLAT"])
    assert corr.loc["ABC", "ABC"] == 1.0


def test_constant_growth_with_float_noise_is_treated_as_constant():
    # 100 -> 110 -> 121 -> 133.1: returns are 0.1 up to floating-point noise.
    data = pd.concat([prices("ABC", X), pd.DataFrame({
        "date": pd.bdate_range("2025-01-01", periods=4), "symbol": "GROW",
        "close": [100, 110, 121, 133.1]})], ignore_index=True)
    corr = calculate_correlation_matrix(data)
    assert math.isnan(corr.loc["GROW", "GROW"])
    assert math.isnan(corr.loc["ABC", "GROW"])


# Empty input, missing columns, no mutation

def test_empty_input_is_handled_clearly():
    empty = two_symbols().iloc[0:0]
    assert calculate_covariance_matrix(empty).empty
    assert calculate_correlation_matrix(empty).empty
    assert calculate_annualized_covariance_matrix(empty).empty
    info = describe_return_alignment(empty)
    assert info["observations"] == 0 and info["symbols"] == [] and info["start_date"] is None


@pytest.mark.parametrize("dropped", ["date", "symbol", "close"])
def test_missing_required_columns_fail_clearly(dropped):
    with pytest.raises(ValueError, match=f"missing column\\(s\\): {dropped}"):
        calculate_covariance_matrix(two_symbols().drop(columns=dropped))


def test_duplicate_rows_without_validation_fail_clearly():
    data = pd.concat([two_symbols(), two_symbols().iloc[[1]]])
    with pytest.raises(ValueError, match="duplicate symbol/date"):
        calculate_covariance_matrix(data)


def test_input_is_not_mutated():
    data = two_symbols()
    data["validation_status"] = "VALID"
    before = data.copy()

    calculate_covariance_matrix(data)
    calculate_annualized_covariance_matrix(data)
    calculate_correlation_matrix(data)
    describe_return_alignment(data)

    pd.testing.assert_frame_equal(data, before)


# End-to-end on the synthetic 3-symbol sample via the CSV importer

def test_three_symbol_sample_through_csv_importer():
    canonical = load_csv_market_data(SAMPLE_3_SYMBOLS).data
    info = describe_return_alignment(canonical)
    aligned = calculate_aligned_return_matrix(canonical)
    cov = calculate_covariance_matrix(canonical)
    corr = calculate_correlation_matrix(canonical)

    assert info["symbols"] == ["ABC", "LMN", "XYZ"]
    assert info["observations"] == 24
    for a in info["symbols"]:
        for b in info["symbols"]:
            assert cov.loc[a, b] == pytest.approx(
                statistics.covariance(list(aligned[a]), list(aligned[b])), rel=1e-9)
            assert corr.loc[a, b] == pytest.approx(
                statistics.correlation(list(aligned[a]), list(aligned[b])), rel=1e-9)

import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    annualised_return,
    annualised_volatility,
    build_summary_table,
    correlation_matrix,
    daily_returns,
    drawdown_series,
    max_drawdown,
    portfolio_diversification_score,
    rolling_beta,
    sharpe_ratio,
    value_at_risk,
)


def test_daily_returns_values():
    prices = pd.DataFrame({"A": [100.0, 110.0, 99.0]})
    r = daily_returns(prices)
    assert len(r) == 2  # first row has no previous day and is dropped
    assert r["A"].tolist() == pytest.approx([0.10, -0.10])


def test_annualised_return_constant():
    r = pd.Series([0.001] * 252)
    assert annualised_return(r) == pytest.approx(1.001 ** 252 - 1)


def test_annualised_volatility_zero_for_constant_returns():
    assert annualised_volatility(pd.Series([0.01] * 50)) == pytest.approx(0.0)


def test_annualised_volatility_scales_by_sqrt_252():
    r = pd.Series([0.01, -0.01] * 50)
    assert annualised_volatility(r) == pytest.approx(r.std() * np.sqrt(252))


def test_sharpe_zero_volatility_returns_float_zero():
    result = sharpe_ratio(pd.Series([0.001] * 20))
    assert result == 0.0
    assert isinstance(result, float)


def test_sharpe_sign():
    rng = np.random.default_rng(0)
    good = pd.Series(rng.normal(0.002, 0.01, 500))
    bad = pd.Series(rng.normal(-0.002, 0.01, 500))
    assert sharpe_ratio(good) > 0
    assert sharpe_ratio(bad) < 0


def test_max_drawdown_known_series():
    prices = pd.Series([100.0, 150.0, 75.0, 120.0])
    assert max_drawdown(prices) == pytest.approx(-0.5)


def test_max_drawdown_rising_series_is_zero():
    assert max_drawdown(pd.Series([1.0, 2.0, 3.0])) == 0


def test_drawdown_series_never_positive(prices):
    dd = drawdown_series(prices["AAPL"])
    assert (dd <= 0).all()
    assert len(dd) == len(prices)


def test_value_at_risk_percentile():
    r = pd.Series(np.arange(-50, 50) / 1000)  # -0.050 ... 0.049
    assert value_at_risk(r, 0.95) == pytest.approx(np.percentile(r, 5))
    assert value_at_risk(r, 0.95) < 0


def test_rolling_beta_of_levered_stock(market_returns):
    stock = (2 * market_returns).rename("LEV")
    beta = rolling_beta(stock, market_returns, window=30)
    assert len(beta) == len(market_returns) - 30
    assert beta.name == "LEV"
    # np.cov uses ddof=1 and np.var ddof=0, so the ratio is 2 * n/(n-1)
    assert beta.to_numpy() == pytest.approx(2 * 30 / 29)


def test_rolling_beta_nan_when_market_flat(dates):
    flat = pd.Series(0.0, index=dates)
    stock = pd.Series(np.linspace(-0.01, 0.01, len(dates)), index=dates)
    beta = rolling_beta(stock, flat, window=20)
    assert beta.isna().all()


def test_build_summary_table(prices, market_returns):
    summary = build_summary_table(prices, market_returns)
    assert list(summary.index) == list(prices.columns)
    assert list(summary.columns) == [
        "Annual Return", "Volatility", "Sharpe Ratio",
        "Max Drawdown", "VaR (95%)", "Current Beta",
    ]
    assert (summary["Max Drawdown"] <= 0).all()
    assert (summary["Volatility"] > 0).all()
    assert summary["Current Beta"].notna().all()


def test_correlation_matrix_shape(prices):
    corr = correlation_matrix(prices)
    assert corr.shape == (3, 3)
    assert np.allclose(corr.values, corr.values.T)
    assert np.allclose(np.diag(corr.values), 1.0)


def test_diversification_score_identical_series():
    corr = pd.DataFrame(np.ones((3, 3)))
    assert portfolio_diversification_score(corr) == pytest.approx(1.0)


def test_diversification_score_uncorrelated():
    corr = pd.DataFrame(np.eye(4))
    assert portfolio_diversification_score(corr) == pytest.approx(0.0)

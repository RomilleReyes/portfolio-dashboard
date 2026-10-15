import pandas as pd
import pytest

from src.fetch import PORTFOLIO, get_benchmark, get_portfolio_value, get_prices


def test_get_prices_converts_vuag_to_gbp(fake_download, tmp_path):
    prices = get_prices(out_dir=tmp_path)
    assert prices["VUAG.L"].eq(90.0).all()  # 9000 pence -> £90


def test_get_prices_drops_partial_leading_rows(fake_download, tmp_path, dates):
    prices = get_prices(out_dir=tmp_path)
    assert prices.notna().all().all()
    assert prices.index[0] == dates[1]  # the VUAG.L-only first row is gone
    assert set(prices.columns) == set(PORTFOLIO)


def test_first_portfolio_value_is_complete(fake_download, tmp_path):
    """Regression: the first value used to only count VUAG.L (others were NaN -> 0)."""
    prices = get_prices(out_dir=tmp_path)
    value = get_portfolio_value(prices, out_dir=tmp_path)
    assert value.iloc[0] > 0.5 * value.iloc[1]


def test_get_prices_skips_ticker_that_failed(monkeypatch, tmp_path, dates):
    def _download(tickers, *args, **kwargs):
        close = pd.DataFrame({t: 100.0 for t in tickers}, index=dates)
        close["MRNA"] = float("nan")
        close.columns = pd.MultiIndex.from_product([["Close"], close.columns])
        return close

    monkeypatch.setattr("src.fetch.yf.download", _download)
    prices = get_prices(out_dir=tmp_path)
    assert "MRNA" not in prices.columns
    assert len(prices) == len(dates)


def test_get_prices_writes_csv(fake_download, tmp_path):
    get_prices(out_dir=tmp_path)
    saved = pd.read_csv(tmp_path / "prices.csv", index_col=0)
    assert set(saved.columns) == set(PORTFOLIO)


def test_portfolio_value_is_price_times_shares(tmp_path):
    prices = pd.DataFrame({"AAPL": [10.0, 20.0], "NVDA": [1.0, 2.0]})
    value = get_portfolio_value(prices, out_dir=tmp_path)
    expected = prices["AAPL"] * PORTFOLIO["AAPL"] + prices["NVDA"] * PORTFOLIO["NVDA"]
    assert value.tolist() == pytest.approx(expected.tolist())
    assert (tmp_path / "portfolio_value.csv").exists()


def test_portfolio_value_ignores_unknown_columns(tmp_path):
    prices = pd.DataFrame({"AAPL": [10.0], "NOT_HELD": [1_000_000.0]})
    value = get_portfolio_value(prices, out_dir=tmp_path)
    assert value.iloc[0] == pytest.approx(10.0 * PORTFOLIO["AAPL"])


def test_get_benchmark_returns_series(fake_download):
    spy = get_benchmark()
    assert isinstance(spy, pd.Series)
    assert spy.name == "SPY"
    assert spy.iloc[0] == pytest.approx(500.0)

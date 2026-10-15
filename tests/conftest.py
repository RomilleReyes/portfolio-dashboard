import numpy as np
import pandas as pd
import pytest

from src.fetch import PORTFOLIO


@pytest.fixture
def dates():
    return pd.bdate_range("2025-01-01", periods=120, name="Date")


@pytest.fixture
def prices(dates):
    """Deterministic random-walk prices for three tickers."""
    rng = np.random.default_rng(42)
    data = {
        t: 100 * np.cumprod(1 + rng.normal(0.0005, 0.02, len(dates)))
        for t in ["AAPL", "NVDA", "BRK-B"]
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def market_returns(dates):
    rng = np.random.default_rng(7)
    return pd.Series(rng.normal(0.0004, 0.01, len(dates)), index=dates, name="SPY")


def make_download_frame(dates, tickers):
    """
    Mimic yf.download(...) for several tickers: MultiIndex columns ("Close", ticker).
    The first row only has a VUAG.L price, like a London session opening before New York.
    """
    close = pd.DataFrame(
        {t: np.linspace(100, 120, len(dates)) + i for i, t in enumerate(tickers)},
        index=dates,
    )
    if "VUAG.L" in close.columns:
        close["VUAG.L"] = 9000.0  # priced in pence
    close.iloc[0, [i for i, t in enumerate(tickers) if t != "VUAG.L"]] = np.nan
    close.columns = pd.MultiIndex.from_product([["Close"], close.columns])
    return close


@pytest.fixture
def fake_download(monkeypatch, dates):
    """Replace yfinance.download so no test touches the network."""
    calls = []

    def _download(tickers, *args, **kwargs):
        calls.append(tickers)
        if isinstance(tickers, str):  # single ticker, e.g. the SPY benchmark
            spy = pd.DataFrame({"SPY": np.linspace(500, 550, len(dates))}, index=dates)
            spy.columns = pd.MultiIndex.from_product([["Close"], spy.columns])
            return spy
        return make_download_frame(dates, list(tickers))

    monkeypatch.setattr("src.fetch.yf.download", _download)
    return calls


@pytest.fixture
def all_tickers():
    return list(PORTFOLIO.keys())

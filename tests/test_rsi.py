import numpy as np
import pandas as pd
import pytest

from src.rsi import calc_bollinger_bands, calc_ema, calc_rsi, calc_sma, rsi_signal


def test_rsi_bounded(prices):
    rsi = calc_rsi(prices["AAPL"]).dropna()
    assert ((rsi >= 0) & (rsi <= 100)).all()


def test_rsi_nan_before_period():
    rsi = calc_rsi(pd.Series(np.arange(1.0, 31.0)), period=14)
    assert rsi.iloc[:14].isna().all()  # diff() costs one row, then 14 to warm up
    assert rsi.iloc[14:].notna().all()


def test_rsi_strictly_rising_is_100():
    rsi = calc_rsi(pd.Series(np.arange(1.0, 41.0)))
    assert rsi.iloc[-1] == pytest.approx(100.0)


def test_rsi_strictly_falling_is_0():
    rsi = calc_rsi(pd.Series(np.arange(40.0, 0.0, -1.0)))
    assert rsi.iloc[-1] == pytest.approx(0.0)


def test_sma():
    sma = calc_sma(pd.Series([1.0, 2.0, 3.0, 4.0]), window=2)
    assert sma.isna().iloc[0]
    assert sma.iloc[1:].tolist() == [1.5, 2.5, 3.5]


def test_ema_first_value_and_direction():
    ema = calc_ema(pd.Series([10.0, 20.0, 20.0, 20.0]), span=3)
    assert ema.iloc[0] == 10.0
    assert ema.iloc[1] == pytest.approx(15.0)  # alpha = 2 / (3 + 1) = 0.5
    assert ema.is_monotonic_increasing


def test_bollinger_band_ordering(prices):
    upper, mid, lower = calc_bollinger_bands(prices["NVDA"])
    valid = upper.notna()
    assert (upper[valid] >= mid[valid]).all()
    assert (mid[valid] >= lower[valid]).all()
    assert mid.equals(calc_sma(prices["NVDA"], 20))


@pytest.mark.parametrize("value, expected", [
    (85, "Overbought ⚠️"),
    (70, "Overbought ⚠️"),
    (69.9, "Neutral"),
    (50, "Neutral"),
    (30.1, "Neutral"),
    (30, "Oversold 🟢"),
    (10, "Oversold 🟢"),
])
def test_rsi_signal(value, expected):
    assert rsi_signal(value) == expected

import pandas as pd
import numpy as np


def calc_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index.
    > 70 = potentially overbought
    < 30 = potentially oversold
    """
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calc_sma(prices: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average"""
    return prices.rolling(window=window).mean()


def calc_ema(prices: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average — more weight on recent prices"""
    return prices.ewm(span=span, adjust=False).mean()


def calc_bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2.0):
    """
    Bollinger Bands: middle band = SMA20, upper/lower = ±2 std deviations.
    When price touches the upper band → potentially overbought.
    When price touches the lower band → potentially oversold.
    """
    sma = calc_sma(prices, window)
    std = prices.rolling(window=window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper, sma, lower


def rsi_signal(rsi_value: float) -> str:
    """Human-readable RSI interpretation"""
    if rsi_value >= 70:
        return "Overbought ⚠️"
    elif rsi_value <= 30:
        return "Oversold 🟢"
    else:
        return "Neutral"


if __name__ == "__main__":
    from src.fetch import get_prices
    prices = get_prices()
    for ticker in prices.columns:
        rsi = calc_rsi(prices[ticker]).iloc[-1]
        print(f"{ticker:10} RSI: {rsi:.1f}  →  {rsi_signal(rsi)}")
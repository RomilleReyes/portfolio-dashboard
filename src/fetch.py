import yfinance as yf
import pandas as pd
import os

# Portfolio: ticker -> number of shares held
PORTFOLIO = {
    "AAPL":  10,   # Apple
    "NVDA":  5,    # NVIDIA
    "AMD":   8,    # AMD
    "UNH":   3,    # UnitedHealth
    "MRNA":  6,    # Moderna
    "NFLX":  2,    # Netflix
    "VUAG.L": 15,  # Vanguard S&P 500 ETF (London-listed, priced in GBX)
    "DBSDY": 4,    # DBS Group (US OTC ADR)
    "BRK-B": 7,    # Berkshire Hathaway B
}

RAW_DIR = os.path.join("data", "raw")

# Ticker and name
TICKER_NAMES = {
    "AAPL":   "Apple",
    "NVDA":   "NVIDIA",
    "AMD":    "AMD",
    "UNH":    "UnitedHealth",
    "MRNA":   "Moderna",
    "NFLX":   "Netflix",
    "VUAG.L": "Vanguard S&P 500 ETF",
    "DBSDY":  "DBS Group",
    "BRK-B":  "Berkshire B",
}

def get_prices(period="1y", interval="1d", out_dir=RAW_DIR):
    """
    Download adjusted closing prices for all tickers.
    Returns a cleaned DataFrame with forward-fill for any missing days
    (e.g. VUAG.L has UK market holidays that US tickers don't).
    """
    tickers = list(PORTFOLIO.keys())
    raw = yf.download(tickers, period=period, interval=interval, auto_adjust=True)
    prices = raw["Close"].copy()

    # A ticker that failed to download entirely would otherwise wipe out every row below
    prices = prices.dropna(axis=1, how="all")

    # Forward-fill gaps (different market holidays across US/UK/SG)
    prices = prices.ffill()

    # Drop leading rows where some tickers have no price yet (e.g. the London
    # session opens before New York), otherwise they'd count as 0 in the total
    prices = prices.dropna(how="any")

    # VUAG.L is priced in GBX (pence), convert to GBP for consistency
    if "VUAG.L" in prices.columns:
        prices["VUAG.L"] = prices["VUAG.L"] / 100

    os.makedirs(out_dir, exist_ok=True)
    prices.to_csv(os.path.join(out_dir, "prices.csv"))
    print(f"✓ Downloaded {len(prices)} trading days for {len(tickers)} tickers")
    return prices


def get_portfolio_value(prices, out_dir=RAW_DIR):
    """
    Multiply each stock's price by your share count
    and sum to get total portfolio value per day.
    """
    shares = pd.Series(PORTFOLIO)
    # Align columns — some tickers might have failed to download
    available = [t for t in shares.index if t in prices.columns]
    value = (prices[available] * shares[available]).sum(axis=1)
    os.makedirs(out_dir, exist_ok=True)
    value.to_csv(os.path.join(out_dir, "portfolio_value.csv"))
    return value


def get_benchmark(period="1y"):
    """
    Download SPY as a market benchmark for beta calculations.
    Returns a Series (yfinance gives a one-column DataFrame for a single ticker).
    """
    spy = yf.download("SPY", period=period, auto_adjust=True)
    close = spy["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.rename("SPY")


if __name__ == "__main__":
    prices = get_prices()
    portfolio = get_portfolio_value(prices)
    benchmark = get_benchmark()
    print(f"\nPortfolio value today: £/$ {portfolio.iloc[-1]:,.2f}")
    print(f"\nSample prices (last 3 days):\n{prices.tail(3).to_string()}")
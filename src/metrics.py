import numpy as np
import pandas as pd


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Percentage change day-over-day.
    This is the raw material for almost every risk metric.
    """
    return prices.pct_change().dropna()


def annualised_return(returns: pd.Series) -> float:
    """
    Compound annual growth rate of a return series.
    252 = approximate trading days in a year.
    """
    return (1 + returns.mean()) ** 252 - 1


def annualised_volatility(returns: pd.Series) -> float:
    """
    Standard deviation of daily returns, scaled to annual.
    Higher = riskier.
    """
    return returns.std() * np.sqrt(252)


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.05) -> float:
    """
    Return per unit of risk taken.
    > 1.0 = good, > 2.0 = excellent, < 0 = better in cash.
    UK risk-free rate (gilts) is around 4-5% as of 2024.
    """
    ann_return = annualised_return(returns)
    ann_vol = annualised_volatility(returns)
    if ann_vol == 0:
        return 0
    return (ann_return - risk_free_rate) / ann_vol


def max_drawdown(prices: pd.Series) -> float:
    """
    The largest peak-to-trough decline.
    e.g. -0.45 means the stock fell 45% from its highest point.
    This is what keeps investors up at night.
    """
    rolling_peak = prices.cummax()
    drawdown = (prices - rolling_peak) / rolling_peak
    return drawdown.min()


def drawdown_series(prices: pd.Series) -> pd.Series:
    """Full drawdown series for plotting, not just the minimum."""
    rolling_peak = prices.cummax()
    return (prices - rolling_peak) / rolling_peak


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    VaR: on a bad day (top 5% worst outcomes), how much do you lose?
    e.g. -0.034 means you could lose 3.4% in a single bad day.
    Used by every bank and fund manager in the world.
    """
    return np.percentile(returns.dropna(), (1 - confidence) * 100)


def rolling_beta(stock_returns: pd.Series,
                 market_returns: pd.Series,
                 window: int = 60) -> pd.Series:
    """
    Beta vs the market (SPY).
    Beta = 1.0: moves with the market.
    Beta > 1.0: amplifies market moves (NVDA typically ~1.5-2.0).
    Beta < 1.0: more defensive (BRK-B typically ~0.8, UNH ~0.6).
    """
    betas = []
    idx = []

    # Align both series on the same dates
    aligned = pd.concat([stock_returns, market_returns], axis=1).dropna()
    s = aligned.iloc[:, 0]
    m = aligned.iloc[:, 1]

    for i in range(window, len(s)):
        s_window = s.iloc[i - window:i]
        m_window = m.iloc[i - window:i]
        cov = np.cov(s_window, m_window)[0][1]
        var = np.var(m_window)
        betas.append(cov / var if var != 0 else np.nan)
        idx.append(s.index[i])

    return pd.Series(betas, index=idx, name=stock_returns.name)


def build_summary_table(prices: pd.DataFrame,
                        market_returns: pd.Series) -> pd.DataFrame:
    """
    Generate one clean summary table for all tickers.
    This is what goes into your weekly report.
    """
    returns = daily_returns(prices)
    rows = []

    for ticker in prices.columns:
        r = returns[ticker].dropna()
        p = prices[ticker].dropna()
        beta_series = rolling_beta(r, market_returns)

        rows.append({
            "Ticker":           ticker,
            "Annual Return":    annualised_return(r),
            "Volatility":       annualised_volatility(r),
            "Sharpe Ratio":     sharpe_ratio(r),
            "Max Drawdown":     max_drawdown(p),
            "VaR (95%)":        value_at_risk(r),
            "Current Beta":     beta_series.iloc[-1] if len(beta_series) > 0 else np.nan,
        })

    df = pd.DataFrame(rows).set_index("Ticker")
    return df


def correlation_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Pairwise correlations between all stock returns.
    Ranges from -1 (perfect inverse) to +1 (perfect together).
    """
    returns = daily_returns(prices)
    return returns.corr()


def portfolio_diversification_score(corr_matrix: pd.DataFrame) -> float:
    """
    Average pairwise correlation across the whole portfolio.
    Lower = better diversified.
    < 0.5 = well diversified
    > 0.8 = dangerously concentrated
    """
    n = len(corr_matrix)
    total = corr_matrix.values.sum() - n
    pairs = n * (n - 1)
    return total / pairs


if __name__ == "__main__":
    from fetch import get_prices, get_benchmark

    prices = get_prices()
    benchmark = get_benchmark()
    market_returns = benchmark.pct_change().dropna()

    summary = build_summary_table(prices, market_returns)

    # Format for readability
    fmt = summary.copy()
    for col in ["Annual Return", "Volatility", "Max Drawdown", "VaR (95%)"]:
        fmt[col] = fmt[col].map(lambda x: f"{x:.1%}")
    fmt["Sharpe Ratio"] = fmt["Sharpe Ratio"].map(lambda x: f"{x:.2f}")
    fmt["Current Beta"] = fmt["Current Beta"].map(lambda x: f"{x:.2f}")

    print("Portfolio Risk Summary")
    print("=" * 70)
    print(fmt.to_string())
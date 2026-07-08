import pandas as pd
import os
from datetime import datetime
from fetch import get_prices, get_portfolio_value, get_benchmark, TICKER_NAMES
from metrics import (daily_returns, build_summary_table,
                     correlation_matrix, portfolio_diversification_score)
from rsi import calc_rsi, rsi_signal


def generate_weekly_report():
    print("📥 Fetching data...")
    prices = get_prices(period="1y")
    portfolio = get_portfolio_value(prices)
    benchmark = get_benchmark(period="1y")

    # Extract SPY as a clean Series
    spy = benchmark["SPY"] if "SPY" in benchmark.columns else benchmark.iloc[:, 0]
    market_returns = spy.pct_change().dropna()
    returns = daily_returns(prices)

    today = datetime.today().strftime("%Y-%m-%d")
    os.makedirs("data/reports", exist_ok=True)

    # ── Section 1: Summary Table ──────────────────────────────────────────
    print("\n📊 Building risk summary...")
    summary = build_summary_table(prices, market_returns)
    summary.to_csv(f"data/reports/summary_{today}.csv")

    # ── Section 2: RSI Signals ────────────────────────────────────────────
    print("\n📡 Calculating RSI signals...")
    rsi_data = {}
    for ticker in prices.columns:
        rsi_val = calc_rsi(prices[ticker]).iloc[-1]
        rsi_data[ticker] = {
            "RSI": round(rsi_val, 1),
            "Signal": rsi_signal(rsi_val),
            "Name": TICKER_NAMES.get(ticker, ticker)
        }
    rsi_df = pd.DataFrame(rsi_data).T
    rsi_df.to_csv(f"data/reports/rsi_{today}.csv")

    # ── Section 3: Portfolio Performance ─────────────────────────────────
    portfolio_return = float((portfolio.iloc[-1] / portfolio.iloc[0] - 1))
    spy_return = float((spy.iloc[-1] / spy.iloc[0] - 1))
    outperformance = portfolio_return - spy_return

    # ── Section 4: Weekly change for each stock ───────────────────────────
    weekly_changes = {}
    for ticker in prices.columns:
        if len(prices[ticker].dropna()) >= 5:
            week_ago = prices[ticker].dropna().iloc[-6]
            today_price = prices[ticker].dropna().iloc[-1]
            weekly_changes[ticker] = (today_price - week_ago) / week_ago

    weekly_df = pd.Series(weekly_changes, name="Weekly Change").sort_values()
    weekly_df.to_csv(f"data/reports/weekly_change_{today}.csv")

    # ── Print the report ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  PORTFOLIO WEEKLY REPORT — {today}")
    print("=" * 60)

    print(f"\n📈 Portfolio Return (1 year): {portfolio_return:.1%}")
    print(f"📊 S&P 500 Return (1 year):   {spy_return:.1%}")
    print(f"{'⬆️' if outperformance > 0 else '⬇️'} vs Benchmark:             {outperformance:+.1%}")

    print("\n── RSI Signals ──────────────────────────────────────────")
    for ticker, row in rsi_df.iterrows():
        print(f"  {ticker:10} {row['RSI']:5.1f}   {row['Signal']}")

    print("\n── Weekly Movers ────────────────────────────────────────")
    for ticker, chg in weekly_df.items():
        bar = "▲" if chg > 0 else "▼"
        print(f"  {ticker:10} {bar} {chg:+.2%}")

    print("\n── Risk Metrics (Top 3 Sharpe) ──────────────────────────")
    top3 = summary["Sharpe Ratio"].sort_values(ascending=False).head(3)
    for ticker, sharpe in top3.items():
        print(f"  {ticker:10} Sharpe: {sharpe:.2f}")

    print(f"\n✅ Report saved to data/reports/ ({today})")
    return summary, rsi_df, weekly_df


if __name__ == "__main__":
    generate_weekly_report()
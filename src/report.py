import pandas as pd
import os
from datetime import datetime
from src.fetch import get_prices, get_portfolio_value, get_benchmark, TICKER_NAMES
from src.metrics import build_summary_table
from src.rsi import calc_rsi, rsi_signal


def weekly_change(prices: pd.DataFrame, days: int = 5) -> pd.Series:
    """
    % change over the last `days` trading days for each ticker,
    sorted from worst to best. Tickers with too little history are skipped.
    """
    changes = {}
    for ticker in prices.columns:
        p = prices[ticker].dropna()
        if len(p) > days:
            week_ago = p.iloc[-(days + 1)]
            changes[ticker] = (p.iloc[-1] - week_ago) / week_ago
    return pd.Series(changes, name="Weekly Change", dtype=float).sort_values()


def generate_weekly_report():
    print("📥 Fetching data...")
    prices = get_prices(period="1y")
    portfolio = get_portfolio_value(prices)
    spy = get_benchmark(period="1y")
    market_returns = spy.pct_change().dropna()

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
    weekly_df = weekly_change(prices)
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
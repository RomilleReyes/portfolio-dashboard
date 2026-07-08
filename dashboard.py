import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from src.fetch import get_prices, get_portfolio_value, get_benchmark, TICKER_NAMES
from src.metrics import (daily_returns, drawdown_series, rolling_beta,
                          correlation_matrix, build_summary_table)
from src.rsi import calc_rsi, calc_sma, calc_bollinger_bands

# ── Colours per ticker ────────────────────────────────────────────────────
COLOURS = {
    "AAPL":   "#378ADD",
    "NVDA":   "#639922",
    "AMD":    "#D85A30",
    "UNH":    "#533AB7",
    "MRNA":   "#1D9E75",
    "NFLX":   "#E24B4A",
    "VUAG.L": "#F0A500",
    "DBSDY":  "#E84393",
    "BRK-B":  "#9B59B6",
}

print("📥 Loading data...")
prices     = get_prices(period="1y")
portfolio  = get_portfolio_value(prices)
benchmark  = get_benchmark(period="1y")
returns    = daily_returns(prices)
mkt_ret    = benchmark.pct_change().dropna()

fig = make_subplots(
    rows=4, cols=2,
    subplot_titles=[
        "📈 Portfolio Value (1 Year)",
        "📊 Cumulative Returns vs S&P 500",
        "🌡️ RSI — All Tickers",
        "📉 Drawdown Chart",
        "🔗 Correlation Heatmap",
        "⚡ 30-Day Rolling Volatility",
        "🎯 Sharpe Ratios",
        "🧮 Rolling Beta vs S&P 500",
    ],
    vertical_spacing=0.1,
    horizontal_spacing=0.08,
)

# Panel 1: Portfolio value
fig.add_trace(go.Scatter(
    x=portfolio.index, y=portfolio,
    fill="tozeroy", fillcolor="rgba(55,138,221,0.1)",
    line=dict(color="#378ADD", width=2),
    name="Portfolio Value"
), row=1, col=1)

# Panel 2: Cumulative returns vs SPY
spy_cum = (1 + mkt_ret).cumprod()
fig.add_trace(go.Scatter(
    x=spy_cum.index, y=spy_cum,
    line=dict(color="#888", width=1.5, dash="dash"),
    name="S&P 500"
), row=1, col=2)

for ticker in prices.columns:
    r = returns[ticker].dropna()
    cum = (1 + r).cumprod()
    fig.add_trace(go.Scatter(
        x=cum.index, y=cum,
        line=dict(color=COLOURS.get(ticker, "#aaa"), width=1.5),
        name=TICKER_NAMES.get(ticker, ticker)
    ), row=1, col=2)

# Panel 3: RSI for all tickers
for ticker in prices.columns:
    rsi = calc_rsi(prices[ticker])
    fig.add_trace(go.Scatter(
        x=rsi.index, y=rsi,
        line=dict(color=COLOURS.get(ticker, "#aaa"), width=1.3),
        name=ticker, legendgroup=ticker, showlegend=False
    ), row=2, col=1)

# RSI reference lines
for level, colour in [(70, "rgba(226,75,74,0.4)"), (30, "rgba(29,158,117,0.4)"), (50, "rgba(128,128,128,0.2)")]:
    fig.add_hline(y=level, line_dash="dash", line_color=colour, row=2, col=1)

# Panel 4: Drawdown
for ticker in prices.columns:
    dd = drawdown_series(prices[ticker].dropna())
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd,
        fill="tozeroy", fillcolor="rgba(128,128,128,0.1)",
        line=dict(color=COLOURS.get(ticker, "#aaa"), width=1),
        name=ticker, legendgroup=ticker, showlegend=False
    ), row=2, col=2)

# Panel 5: Correlation heatmap
corr = correlation_matrix(prices)
fig.add_trace(go.Heatmap(
    z=corr.values,
    x=corr.columns.tolist(),
    y=corr.index.tolist(),
    colorscale="RdBu_r",
    zmin=-1, zmax=1,
    text=[[f"{v:.2f}" for v in row] for row in corr.values],
    texttemplate="%{text}",
    showscale=True,
    name="Correlation"
), row=3, col=1)

# Panel 6: Rolling 30-day volatility
for ticker in prices.columns:
    vol = returns[ticker].rolling(window=30).std() * (252 ** 0.5)
    fig.add_trace(go.Scatter(
        x=vol.index, y=vol,
        line=dict(color=COLOURS.get(ticker, "#aaa"), width=1.3),
        name=ticker, legendgroup=ticker, showlegend=False
    ), row=3, col=2)

# Panel 7: Sharpe ratios (bar chart)
summary = build_summary_table(prices, mkt_ret)
sharpe = summary["Sharpe Ratio"].sort_values(ascending=True)
bar_colours = [COLOURS.get(t, "#aaa") for t in sharpe.index]
fig.add_trace(go.Bar(
    x=sharpe.values,
    y=[TICKER_NAMES.get(t, t) for t in sharpe.index],
    orientation="h",
    marker_color=bar_colours,
    name="Sharpe Ratio"
), row=4, col=1)
fig.add_vline(x=0, line_color="rgba(128,128,128,0.5)", row=4, col=1)

# Panel 8: Rolling beta for each ticker
for ticker in prices.columns:
    r = returns[ticker].dropna()
    beta = rolling_beta(r, mkt_ret, window=60)
    fig.add_trace(go.Scatter(
        x=beta.index, y=beta,
        line=dict(color=COLOURS.get(ticker, "#aaa"), width=1.3),
        name=ticker, legendgroup=ticker, showlegend=False
    ), row=4, col=2)
fig.add_hline(y=1.0, line_dash="dash",
              line_color="rgba(128,128,128,0.4)", row=4, col=2)

# ── Layout ────────────────────────────────────────────────────────────────
fig.update_layout(
    height=1400,
    title_text="Portfolio Intelligence Dashboard",
    title_font_size=18,
    template="plotly_dark",
    legend=dict(orientation="h", y=-0.02, x=0),
    hovermode="x unified",
)

fig.update_yaxes(tickformat=".0%", row=2, col=2)  # Drawdown as %
fig.update_yaxes(tickformat=".0%", row=3, col=2)  # Volatility as %
fig.update_xaxes(tickformat=".2f", row=4, col=1)  # Sharpe values

fig.write_html("data/dashboard.html")
print("✅ Dashboard saved to data/dashboard.html")
fig.show()


#Run with dashboard.py

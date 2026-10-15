# Portfolio Intelligence Dashboard

A Python toolkit that downloads a year of daily prices for a stock portfolio, calculates risk metrics and technical signals, and produces a weekly report and an interactive 8-panel Plotly dashboard.

## Features

- **Data fetching**: adjusted daily closes from Yahoo Finance via `yfinance`. Prices are forward-filled across US and UK market holidays, and the London-listed ETF is converted from pence to pounds.
- **Risk metrics**: annualised return and volatility, Sharpe ratio, max drawdown, 95% Value at Risk, rolling beta against the S&P 500, the correlation matrix and a diversification score.
- **Technical indicators**: RSI with overbought and oversold signals, SMA, EMA and Bollinger Bands.
- **Weekly report**: a console summary plus CSVs in `data/reports/`, covering the risk summary, RSI signals and weekly movers.
- **Dashboard**: an interactive HTML page (`data/dashboard.html`) with 8 panels: portfolio value, cumulative returns vs the S&P 500, RSI, drawdowns, a correlation heatmap, rolling volatility, Sharpe ratios and rolling beta.

## Holdings

| Ticker | Name | Shares |
| --- | --- | --- |
| AAPL | Apple | 10 |
| NVDA | NVIDIA | 5 |
| AMD | AMD | 8 |
| UNH | UnitedHealth | 3 |
| MRNA | Moderna | 6 |
| NFLX | Netflix | 2 |
| VUAG.L | Vanguard S&P 500 ETF (London, GBP) | 15 |
| DBSDY | DBS Group (US ADR) | 4 |
| BRK-B | Berkshire Hathaway B | 7 |

To track your own portfolio, edit `PORTFOLIO` and `TICKER_NAMES` in [`src/fetch.py`](src/fetch.py), and optionally `COLOURS` in [`dashboard.py`](dashboard.py).

## Project structure

```
portfolio-dashboard/
├── dashboard.py          # Builds the 8-panel Plotly dashboard
├── src/
│   ├── fetch.py          # Portfolio config + price download and cleaning
│   ├── metrics.py        # Risk metrics (returns, Sharpe, VaR, beta, ...)
│   ├── rsi.py            # Technical indicators (RSI, SMA, EMA, Bollinger)
│   └── report.py         # Weekly report (console + CSV)
├── tests/                # pytest suite (offline, yfinance is mocked)
├── data/
│   ├── raw/              # prices.csv, portfolio_value.csv
│   ├── reports/          # dated report CSVs
│   └── dashboard.html    # generated dashboard
├── requirements.txt
└── requirements-dev.txt
```

## Setup

Requires Python 3.11 or later (developed on 3.14).

```bash
git clone https://github.com/RomilleReyes/portfolio-dashboard.git
cd portfolio-dashboard
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run every command from the project root:

```bash
python dashboard.py        # build data/dashboard.html and open it in the browser
python -m src.report       # print the weekly report and save CSVs to data/reports/
python -m src.metrics      # print the risk summary table
python -m src.rsi          # print the current RSI for each ticker
python -m src.fetch        # download prices and print the latest portfolio value
```

## Metrics at a glance

| Metric | What it tells you |
| --- | --- |
| Annual return | Average daily return compounded over 252 trading days |
| Volatility | Annualised standard deviation of daily returns; higher means riskier |
| Sharpe ratio | Return above the risk-free rate (5%) per unit of volatility; above 1 is good |
| Max drawdown | Largest fall from a peak, e.g. −0.30 means a 30% drop |
| VaR (95%) | The daily loss exceeded on only 5% of days |
| Beta | Sensitivity to the S&P 500 over a rolling 60-day window; 1 means it moves with the market |
| RSI | Momentum oscillator (0 to 100); above 70 may mean overbought, below 30 oversold |

## Tests

The test suite uses synthetic data and a mocked `yfinance`, so it runs offline in under a second.

```bash
pip install -r requirements-dev.txt
python -m pytest -v
```

## Notes and limitations

- Holdings in different currencies (USD and GBP) are added together without FX conversion, so the total portfolio value is indicative only.
- Data comes from Yahoo Finance and may be delayed or occasionally incomplete.

## Disclaimer

This project is for educational purposes only and is not financial advice.

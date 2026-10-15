import plotly.graph_objects as go

from dashboard import build_dashboard
from src.fetch import get_portfolio_value


def test_build_dashboard(prices, market_returns, tmp_path):
    portfolio = get_portfolio_value(prices, out_dir=tmp_path)
    fig = build_dashboard(prices, portfolio, market_returns)

    assert isinstance(fig, go.Figure)
    n = len(prices.columns)
    # portfolio + SPY + per-ticker (cum returns, RSI, drawdown, vol, beta) + heatmap + sharpe bars
    assert len(fig.data) == 2 + 5 * n + 2
    assert fig.layout.title.text == "Portfolio Intelligence Dashboard"
    assert sum(isinstance(t, go.Heatmap) for t in fig.data) == 1


def test_importing_dashboard_has_no_side_effects(fake_download):
    """Importing the module must not download data (fake_download records calls)."""
    import importlib

    import dashboard
    importlib.reload(dashboard)
    assert fake_download == []

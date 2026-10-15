import numpy as np
import pandas as pd
import pytest

from src import report
from src.report import generate_weekly_report, weekly_change


def test_weekly_change_math():
    prices = pd.DataFrame({
        "UP":   [100.0, 1, 1, 1, 1, 110.0],
        "DOWN": [200.0, 1, 1, 1, 1, 150.0],
    })
    changes = weekly_change(prices)
    assert changes["UP"] == pytest.approx(0.10)
    assert changes["DOWN"] == pytest.approx(-0.25)
    assert list(changes.index) == ["DOWN", "UP"]  # sorted worst to best


def test_weekly_change_skips_short_history():
    """Regression: exactly 5 rows used to raise IndexError (iloc[-6])."""
    prices = pd.DataFrame({"A": [1.0, 2, 3, 4, 5]})
    assert weekly_change(prices).empty


def test_weekly_change_ignores_nans():
    prices = pd.DataFrame({"A": [100.0, np.nan, 1, 1, 1, 1, 120.0]})
    assert weekly_change(prices)["A"] == pytest.approx(0.20)


def test_generate_weekly_report_end_to_end(fake_download, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    summary, rsi_df, weekly_df = generate_weekly_report()

    reports = sorted(p.name.split("_")[0] for p in (tmp_path / "data" / "reports").iterdir())
    assert reports == ["rsi", "summary", "weekly"]
    assert (tmp_path / "data" / "raw" / "prices.csv").exists()

    assert len(summary) == len(report.TICKER_NAMES)
    assert set(rsi_df.columns) == {"RSI", "Signal", "Name"}
    assert rsi_df.loc["AAPL", "Name"] == "Apple"
    assert len(weekly_df) == len(report.TICKER_NAMES)

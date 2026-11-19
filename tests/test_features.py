"""Tests for feature engineering.

Run with: pytest
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_signals.features.build_features import (
    add_features,
    add_target,
    build_features,
    rsi,
)


def make_prices(rows: int = 120, ticker: str = "SPY") -> pd.DataFrame:
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.normal(0, 1, rows))
    close = np.clip(close, 20, None)
    dates = pd.bdate_range("2020-01-01", periods=rows).date
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": ticker,
            "open": close * 1.001,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, rows),
        }
    )


def test_rsi_stays_in_range() -> None:
    prices = make_prices()
    values = rsi(prices["close"]).dropna()
    assert not values.empty
    assert values.between(0, 100).all()


def test_returns_are_backward_looking() -> None:
    prices = make_prices()
    features = add_features(prices)
    expected = prices["close"].iloc[1] / prices["close"].iloc[0] - 1
    assert features["ret_1d"].iloc[1] == pytest.approx(expected)


def test_label_uses_forward_return() -> None:
    prices = make_prices()
    labelled = add_target(add_features(prices))
    horizon = 5
    expected = prices["close"].iloc[horizon] / prices["close"].iloc[0] - 1
    assert labelled["fwd_ret"].iloc[0] == pytest.approx(expected)
    assert labelled["fwd_ret"].tail(horizon).isna().all()


def test_build_features_drops_unlabelled_rows() -> None:
    prices = pd.concat([make_prices(ticker=t) for t in ("SPY", "QQQ")], ignore_index=True)
    features = build_features(prices)
    assert features["label"].isin([0, 1]).all()
    assert features["fwd_ret"].notna().all()
    assert features["date"].is_monotonic_increasing

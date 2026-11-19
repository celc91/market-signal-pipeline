"""Feature engineering for daily market data.

Every feature is computed with information available at the close of day *t*.
The label looks five trading days ahead, so the last five rows of each ticker
have no label and are dropped.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HORIZON = 5
TRADING_DAYS = 252


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def average_true_range(frame: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range as a fraction of the close."""
    prev_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add price, volatility and calendar features for one ticker."""
    out = frame.sort_values("date").copy()
    close = out["close"]

    out["ret_1d"] = close.pct_change()
    out["log_ret_1d"] = np.log(close).diff()
    out["sma_10"] = close.rolling(10, min_periods=10).mean()
    out["sma_50"] = close.rolling(50, min_periods=50).mean()
    out["close_to_sma50"] = close / out["sma_50"] - 1
    out["rsi_14"] = rsi(close, 14)
    out["atr_14_pct"] = average_true_range(out, 14) / close
    out["vol_20"] = out["ret_1d"].rolling(20, min_periods=20).std() * np.sqrt(TRADING_DAYS)
    out["volume_z_20"] = (
        out["volume"] - out["volume"].rolling(20, min_periods=20).mean()
    ) / out["volume"].rolling(20, min_periods=20).std()
    out["range_pct"] = (out["high"] - out["low"]) / close
    out["gap_pct"] = out["open"] / close.shift(1) - 1

    dates = pd.to_datetime(out["date"])
    out["day_of_week"] = dates.dt.dayofweek
    out["month"] = dates.dt.month
    return out


def add_target(frame: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Forward return and binary label over the next `horizon` trading days."""
    out = frame.copy()
    out["fwd_ret"] = out["close"].shift(-horizon) / out["close"] - 1
    out["label"] = (out["fwd_ret"] > 0).astype("Int64")
    return out


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Run the full pipeline over a multi-ticker price frame."""
    required = {"date", "ticker", "open", "high", "low", "close", "volume"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    frames = []
    for _, group in prices.groupby("ticker", sort=True):
        if len(group) < 60:
            continue
        enriched = add_target(add_features(group))
        frames.append(enriched)

    if not frames:
        raise ValueError("No ticker had enough rows to build features")

    features = pd.concat(frames, ignore_index=True)
    feature_columns = [
        "ret_1d",
        "log_ret_1d",
        "close_to_sma50",
        "rsi_14",
        "atr_14_pct",
        "vol_20",
        "volume_z_20",
        "range_pct",
        "gap_pct",
        "day_of_week",
        "month",
    ]
    features = features.dropna(subset=feature_columns)
    features = features.dropna(subset=["label"])
    features["label"] = features["label"].astype(int)
    return features.sort_values(["date", "ticker"]).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/prices.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/features.parquet"))
    args = parser.parse_args(argv)

    prices = pd.read_parquet(args.input)
    features = build_features(prices)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(args.output, index=False)

    print(
        f"Wrote {len(features):,} rows and {features.shape[1]} columns to {args.output}\n"
        f"Range: {features['date'].min()} -> {features['date'].max()}\n"
        f"Positive label rate: {features['label'].mean():.1%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

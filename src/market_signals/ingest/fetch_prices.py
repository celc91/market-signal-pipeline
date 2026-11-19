"""Download daily OHLCV prices from Stooq and store them as Parquet.

Stooq serves plain CSV without an API key. The data is used locally only and
is never committed to the repository.
"""

from __future__ import annotations

import argparse
import io
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import requests

STOOQ_URL = "https://stooq.com/q/d/l/"
COLUMNS = ["date", "open", "high", "low", "close", "volume"]
TICKERS = ["SPY", "QQQ", "IWM", "DIA"]


def stooq_symbol(ticker: str) -> str:
    """Map a US ticker to its Stooq symbol (e.g. SPY -> spy.us)."""
    ticker = ticker.lower().strip()
    return ticker if "." in ticker else f"{ticker}.us"


def download_prices(
    ticker: str,
    start: date,
    end: date | None = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """Return a tidy OHLCV frame for one ticker.

    Raises ValueError when Stooq answers with anything other than a CSV table,
    which happens when the symbol does not exist.
    """
    params = {
        "s": stooq_symbol(ticker),
        "i": "d",
        "d1": start.strftime("%Y%m%d"),
        "d2": (end or date.today()).strftime("%Y%m%d"),
    }
    response = requests.get(STOOQ_URL, params=params, timeout=timeout)
    response.raise_for_status()

    text = response.text.strip()
    if not text.lower().startswith("date,"):
        raise ValueError(f"Unexpected Stooq response for {ticker}: {text[:80]!r}")

    frame = pd.read_csv(io.StringIO(text))
    frame.columns = [column.strip().lower() for column in frame.columns]
    missing = set(COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns for {ticker}: {sorted(missing)}")

    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    frame["ticker"] = ticker.upper()
    frame = frame[["date", "ticker", *COLUMNS[1:]]].sort_values("date")

    # Stooq reports zero-volume rows for some holidays; drop them.
    frame = frame[frame["volume"] > 0].reset_index(drop=True)
    return frame


def write_parquet(frame: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=TICKERS)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2010, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=None)
    parser.add_argument("--out", type=Path, default=Path("data/raw/prices.parquet"))
    args = parser.parse_args(argv)

    frames = []
    for ticker in args.tickers:
        try:
            frame = download_prices(ticker, args.start, args.end)
        except (requests.RequestException, ValueError) as exc:
            print(f"[skip] {ticker}: {exc}", file=sys.stderr)
            continue
        print(f"[ok] {ticker}: {len(frame):,} rows ({frame['date'].min()} -> {frame['date'].max()})")
        frames.append(frame)

    if not frames:
        print("No data downloaded.", file=sys.stderr)
        return 1

    combined = pd.concat(frames, ignore_index=True)
    write_parquet(combined, args.out)
    print(f"Wrote {len(combined):,} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

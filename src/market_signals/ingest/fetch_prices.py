"""Download daily OHLCV prices from Yahoo Finance and store them as Parquet.

No API key is required. Data is used locally only and is never committed to
the repository. Yahoo's terms apply; this is an educational project, not a
redistribution service.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import requests

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
COLUMNS = ["date", "open", "high", "low", "close", "volume"]
TICKERS = ["SPY", "QQQ", "IWM", "DIA"]


def _epoch(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp())


def download_prices(
    ticker: str,
    start: date,
    end: date | None = None,
    timeout: int = 30,
    retries: int = 3,
) -> pd.DataFrame:
    """Return a tidy OHLCV frame for one ticker.

    Yahoo answers 429 when it rate-limits; retry with a short backoff before
    giving up. Raises ValueError when the payload has no usable rows.
    """
    params = {
        "period1": _epoch(start),
        "period2": _epoch(end or date.today()),
        "interval": "1d",
        "events": "div,splits",
    }

    last_error: Exception | None = None
    for attempt in range(retries):
        response = requests.get(
            CHART_URL.format(ticker=ticker.upper()),
            params=params,
            headers=HEADERS,
            timeout=timeout,
        )
        if response.status_code == 429:
            last_error = requests.HTTPError("429 Too Many Requests")
            time.sleep(2 * (attempt + 1))
            continue
        response.raise_for_status()
        break
    else:
        raise ValueError(f"Yahoo kept rate-limiting {ticker}: {last_error}")

    payload = response.json()
    results = payload.get("chart", {}).get("result") or []
    if not results:
        error = payload.get("chart", {}).get("error", {})
        raise ValueError(f"No data for {ticker}: {error.get('description', 'empty result')}")

    result = results[0]
    timestamps = result.get("timestamp") or []
    quote = (result.get("indicators", {}).get("quote") or [{}])[0]

    frame = pd.DataFrame(
        {
            "date": [datetime.fromtimestamp(ts, tz=timezone.utc).date() for ts in timestamps],
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "volume": quote.get("volume"),
        }
    )
    frame["ticker"] = ticker.upper()
    frame = frame[COLUMNS + ["ticker"]]
    frame = frame.dropna(subset=["open", "high", "low", "close"])
    frame = frame[frame["volume"].fillna(0) > 0]
    return frame.sort_values("date").reset_index(drop=True)


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
    for position, ticker in enumerate(args.tickers):
        try:
            frame = download_prices(ticker, args.start, args.end)
        except (requests.RequestException, ValueError) as exc:
            print(f"[skip] {ticker}: {exc}", file=sys.stderr)
            continue
        print(
            f"[ok] {ticker}: {len(frame):,} rows "
            f"({frame['date'].min()} -> {frame['date'].max()})"
        )
        frames.append(frame)
        if position < len(args.tickers) - 1:
            time.sleep(1)

    if not frames:
        print("No data downloaded.", file=sys.stderr)
        return 1

    combined = pd.concat(frames, ignore_index=True)
    write_parquet(combined, args.out)
    print(f"Wrote {len(combined):,} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Data directory

Nothing here is committed. The ingest script writes:

- `data/raw/prices.parquet` — daily OHLCV for the selected tickers.
- `data/processed/features.parquet` — model-ready feature table.

Regenerate both with:

```bash
python -m market_signals.ingest.fetch_prices --tickers SPY QQQ IWM DIA
python -m market_signals.features.build_features
```

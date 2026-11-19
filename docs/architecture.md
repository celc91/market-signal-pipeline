# Architecture

## Layers

1. **Ingestion** (`src/market_signals/ingest`) pulls daily OHLCV CSVs from
   Stooq and writes a single Parquet file. Stateless and idempotent: rerunning
   it replaces the file.
2. **Feature engineering** (`src/market_signals/features`) produces one row per
   ticker and trading day with price, volatility, volume and calendar features,
   plus the forward 5-day return used as the label.
3. **Warehouse** (`src/market_signals/load`) loads the feature table into
   BigQuery, partitioned by date and clustered by ticker.
4. **Modeling** (`sql/`) trains a BigQuery ML baseline. Later weeks add a custom
   XGBoost container on Vertex AI so both approaches can be compared on cost,
   latency and accuracy.
5. **Serving and monitoring** (weeks 4-6) adds batch prediction, an endpoint,
   Model Registry versions and drift monitoring.

## Data contract

| Column | Type | Notes |
|---|---|---|
| date | DATE | trading day |
| ticker | STRING | uppercase symbol |
| close | FLOAT | split-adjusted close |
| ret_1d | FLOAT | close / previous close - 1 |
| vol_20 | FLOAT | 20-day realised volatility, annualised |
| rsi_14 | FLOAT | Wilder RSI |
| fwd_ret | FLOAT | close[t+5] / close[t] - 1 |
| label | INT64 | 1 if fwd_ret > 0 |

## Design decisions

- **Temporal split only.** A random split leaks the future and inflates every
  metric. Train ≤ 2022, evaluate 2023 onwards.
- **Features are point-in-time.** Each row uses information known at the close
  of that day. The forward return is the only forward-looking column and it is
  excluded from the feature set.
- **One model per universe, not per ticker.** Pooling gives more rows and a
  single system to monitor. Ticker-specific models are a later experiment.
- **Metrics before profits.** AUC and log loss first, cost-aware backtest
  second. Accuracy alone is misleading when the class balance drifts.

## Open questions

- Does the signal survive transaction costs and slippage?
- Is the 5-day horizon better than 1 or 10 days?
- Do volatility features dominate because they proxy market regimes?

-- Feature table in pure SQL, built from a raw_prices table
-- (date, ticker, open, high, low, close, volume).
--
-- The Python pipeline in src/market_signals/features is the canonical path.
-- This file exists to practise window functions in BigQuery and to keep the
-- whole feature definition reviewable in one place.
--
-- Replace YOUR_PROJECT_ID before running.

CREATE OR REPLACE TABLE `YOUR_PROJECT_ID.market_signals.features_sql`
PARTITION BY date
CLUSTER BY ticker
AS
WITH base AS (
  SELECT
    date,
    ticker,
    open,
    high,
    low,
    close,
    volume,
    LAG(close) OVER w AS prev_close,
    AVG(close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) AS sma_10,
    AVG(close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) AS sma_50,
    AVG(volume) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS vol_mean_20,
    STDDEV_SAMP(volume) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS vol_std_20,
    LEAD(close, 5) OVER w AS close_5d_ahead
  FROM `YOUR_PROJECT_ID.market_signals.raw_prices`
  WINDOW w AS (PARTITION BY ticker ORDER BY date)
),
returns AS (
  SELECT
    *,
    SAFE_DIVIDE(close, prev_close) - 1 AS ret_1d,
    LN(SAFE_DIVIDE(close, prev_close)) AS log_ret_1d
  FROM base
),
features AS (
  SELECT
    date,
    ticker,
    ret_1d,
    log_ret_1d,
    SAFE_DIVIDE(close, sma_50) - 1 AS close_to_sma50,
    SAFE_DIVIDE(close, sma_10) - 1 AS close_to_sma10,
    SAFE_DIVIDE(high - low, close) AS range_pct,
    SAFE_DIVIDE(open, prev_close) - 1 AS gap_pct,
    SAFE_DIVIDE(volume - vol_mean_20, vol_std_20) AS volume_z_20,
    STDDEV_SAMP(ret_1d) OVER (
      PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
    ) * SQRT(252) AS vol_20,
    EXTRACT(DAYOFWEEK FROM date) AS day_of_week,
    EXTRACT(MONTH FROM date) AS month,
    SAFE_DIVIDE(close_5d_ahead, close) - 1 AS fwd_ret
  FROM returns
)
SELECT
  *,
  IF(fwd_ret > 0, 1, 0) AS label
FROM features
WHERE fwd_ret IS NOT NULL
  AND ret_1d IS NOT NULL
  AND close_to_sma50 IS NOT NULL
  AND vol_20 IS NOT NULL;

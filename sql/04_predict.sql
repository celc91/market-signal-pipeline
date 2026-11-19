-- Latest predictions, ready to be written to a serving table.
-- Replace YOUR_PROJECT_ID before running.

WITH latest AS (
  SELECT MAX(date) AS max_date
  FROM `YOUR_PROJECT_ID.market_signals.features`
)
SELECT
  p.date,
  p.ticker,
  p.predicted_label,
  p.predicted_label_probs
FROM ML.PREDICT(
  MODEL `YOUR_PROJECT_ID.market_signals.model_direction_bqml`,
  (
    SELECT * EXCEPT (label, fwd_ret)
    FROM `YOUR_PROJECT_ID.market_signals.features`, latest
    WHERE date = latest.max_date
  )
) AS p
ORDER BY ticker;

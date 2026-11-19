-- Evaluation on the held-out period.
-- Replace YOUR_PROJECT_ID before running.

-- 1. Model metrics on 2023+ data.
SELECT *
FROM ML.EVALUATE(
  MODEL `YOUR_PROJECT_ID.market_signals.model_direction_bqml`,
  (
    SELECT * EXCEPT (date, ticker, fwd_ret)
    FROM `YOUR_PROJECT_ID.market_signals.features`
    WHERE date > '2022-12-31'
  )
);

-- 2. Naive baseline: always predict "up". A model that cannot beat this
--    accuracy (or its AUC) has no edge worth reporting.
SELECT
  AVG(label) AS naive_up_rate,
  COUNT(*) AS rows_evaluated
FROM `YOUR_PROJECT_ID.market_signals.features`
WHERE date > '2022-12-31';

-- 3. Confusion matrix at the default threshold.
SELECT *
FROM ML.CONFUSION_MATRIX(
  MODEL `YOUR_PROJECT_ID.market_signals.model_direction_bqml`,
  (
    SELECT * EXCEPT (date, ticker, fwd_ret)
    FROM `YOUR_PROJECT_ID.market_signals.features`
    WHERE date > '2022-12-31'
  )
);

-- 4. Which features carry the signal (or the noise).
SELECT *
FROM ML.GLOBAL_EXPLAIN(
  MODEL `YOUR_PROJECT_ID.market_signals.model_direction_bqml`
)
ORDER BY attribution DESC;

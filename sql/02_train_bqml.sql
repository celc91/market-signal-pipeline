-- Baseline model: can a tree ensemble beat "the market usually goes up"?
--
-- Split is temporal on purpose: train on everything up to 2022-12-31 and keep
-- 2023+ untouched for evaluation. No random split, no future leakage.
--
-- Replace YOUR_PROJECT_ID before running.

CREATE OR REPLACE MODEL `YOUR_PROJECT_ID.market_signals.model_direction_bqml`
OPTIONS (
  model_type = 'BOOSTED_TREE_CLASSIFIER',
  input_label_cols = ['label'],
  auto_class_weights = TRUE,
  max_iterations = 50,
  early_stop = TRUE,
  data_split_method = 'NO_SPLIT',
  enable_global_explain = TRUE
) AS
SELECT
  * EXCEPT (date, ticker, fwd_ret)
FROM `YOUR_PROJECT_ID.market_signals.features`
WHERE date <= '2022-12-31';

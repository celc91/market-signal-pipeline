# Model card — market direction baseline

## Model details

- **Name:** `model_direction_bqml`
- **Type:** BigQuery ML `BOOSTED_TREE_CLASSIFIER`
- **Version:** baseline v0.1 (2026-11-18)
- **Owner:** Carlos Lugo
- **Training code:** `sql/02_train_bqml.sql`
- **Local equivalent:** `src/market_signals/models/baseline.py`

## Intended use

- **Primary use:** educational. Demonstrate an end-to-end, leakage-free ML
  pipeline on Google Cloud for the PMLE certification.
- **Out of scope:** real trading, capital allocation, or any decision with
  financial consequences. The model has no demonstrated edge.
- **Users:** the author and anyone reviewing the portfolio project.

## Training data

- Daily OHLCV for SPY, QQQ, IWM and DIA from Yahoo Finance (2010-03-16 to 2026-11-10).
- 16,764 rows after feature engineering; 12,892 used for training (≤ 2022-12-31).
- No data is redistributed with the repository.
- Label: 1 when the 5-day forward return is positive (58.4% of rows overall).

## Features

`ret_1d`, `log_ret_1d`, `close_to_sma50`, `rsi_14`, `atr_14_pct`, `vol_20`,
`volume_z_20`, `range_pct`, `gap_pct`, `day_of_week`, `month`.

All features use information available at the close of the same day. The
forward return is excluded from training inputs.

## Evaluation

- **Split:** temporal. Train ≤ 2022-12-31; test 2023-01-03 → 2026-11-10 (3,872 rows).
- **Metrics:** ROC AUC 0.498, log loss 0.714, accuracy 0.494, precision 0.575,
  recall 0.464, F1 0.513.
- **Baseline:** always-up rule, accuracy 0.575.
- **Interpretation:** no measurable edge. The model is statistically
  indistinguishable from a coin flip on this holdout.

## Limitations

- Single market regime family (US large-cap ETFs); no cross-asset validation.
- Dividends and borrow costs are not modelled; the backtest is not implemented yet.
- Survivorship and look-ahead risks are controlled by construction but not audited
  by a third party.
- Class balance drifts with the market; monitoring is not implemented yet.

## Ethical and legal considerations

- Not investment advice. Do not deploy with real capital.
- Yahoo Finance terms apply to the data; the repository ships no market data.
- Predictions should never be presented as guaranteed returns.

## Maintenance

- Retraining: manual for now; Vertex Pipelines will automate it.
- Monitoring: Vertex AI Model Monitoring planned for week 6 of the plan.
- Review cadence: after every modelling change, update this card and the README table.

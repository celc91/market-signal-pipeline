# 0001 — Problem framing

- **Status:** accepted
- **Date:** 2026-11-18

## Context

The project needs a target that is honest, measurable and useful as an ML
engineering exercise. "Predict the price" is neither testable nor defensible.

## Decision

Predict the **probability that the 5-day forward return is positive** for a
small universe of liquid US ETFs (SPY, QQQ, IWM, DIA).

- Train on data up to 2022-12-31.
- Evaluate on 2023 onwards, untouched during training.
- Report AUC, log loss and a cost-aware backtest, not accuracy alone.
- Treat the naive "always up" rule as the baseline to beat.

## Consequences

- The label is noisy: daily equity returns are close to a random walk, so a
  small AUC lift is the realistic outcome.
- Class balance drifts with the market regime, which is exactly the kind of
  problem Vertex AI Model Monitoring exists for.
- The project's value is the pipeline and the honest evaluation, not a promise
  of alpha.

## Alternatives rejected

- **Regression on price level:** non-stationary, dominated by trend.
- **1-day horizon:** too noisy and too sensitive to microstructure.
- **Random train/test split:** leaks the future and produces fantasy metrics.

"""Local baseline before touching BigQuery ML.

Trains two simple models with a strictly temporal split and reports the
numbers that matter: AUC, log loss, accuracy and precision on the top decile.
The naive "always up" rule is printed next to them, because a model that does
not beat it has no story to tell.

Usage:
    python -m market_signals.models.baseline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "ret_1d",
    "log_ret_1d",
    "close_to_sma50",
    "rsi_14",
    "atr_14_pct",
    "vol_20",
    "volume_z_20",
    "range_pct",
    "gap_pct",
    "day_of_week",
    "month",
]
TRAIN_END = "2022-12-31"
TOP_FRACTION = 0.10


def temporal_split(
    frame: pd.DataFrame, train_end: str = TRAIN_END
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = frame[frame["date"] <= pd.Timestamp(train_end).date()]
    test = frame[frame["date"] > pd.Timestamp(train_end).date()]
    if train.empty or test.empty:
        raise ValueError("Temporal split produced an empty side; check the dates")
    return train, test


def top_decile_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    cutoff = max(1, int(len(scores) * TOP_FRACTION))
    order = np.argsort(scores)[::-1][:cutoff]
    return float(np.mean(y_true[order]))


def evaluate(name: str, scores: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    labels = (scores >= 0.5).astype(int)
    return {
        "model": name,
        "auc": float(roc_auc_score(y_true, scores)),
        "log_loss": float(log_loss(y_true, scores)),
        "accuracy": float(accuracy_score(y_true, labels)),
        "precision_top_decile": top_decile_precision(y_true, scores),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("data/processed/features.parquet")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/baseline_metrics.json")
    )
    args = parser.parse_args(argv)

    frame = pd.read_parquet(args.input)
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    train, test = temporal_split(frame)

    x_train = train[FEATURES].to_numpy(dtype=float)
    y_train = train["label"].to_numpy(dtype=int)
    x_test = test[FEATURES].to_numpy(dtype=float)
    y_test = test["label"].to_numpy(dtype=int)

    naive = float(y_test.mean())
    results = [
        {
            "model": "naive_always_up",
            "auc": 0.5,
            "accuracy": naive,
            "precision_top_decile": naive,
        },
        {
            "model": "majority_class",
            "accuracy": max(naive, 1 - naive),
        },
    ]

    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    logistic.fit(x_train, y_train)
    results.append(evaluate("logistic_regression", logistic.predict_proba(x_test)[:, 1], y_test))

    trees = HistGradientBoostingClassifier(
        max_depth=4,
        max_iter=300,
        learning_rate=0.05,
        l2_regularization=1.0,
        random_state=42,
    )
    trees.fit(x_train, y_train)
    results.append(evaluate("hist_gradient_boosting", trees.predict_proba(x_test)[:, 1], y_test))

    summary = {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_end": str(TRAIN_END),
        "test_start": str(test["date"].min()),
        "test_end": str(test["date"].max()),
        "positive_rate_test": naive,
        "results": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Train: {len(train):,} rows (up to {TRAIN_END})")
    print(f"Test:  {len(test):,} rows ({summary['test_start']} -> {summary['test_end']})")
    print(f"Positive rate in test (naive always-up accuracy): {naive:.3f}\n")
    for row in results:
        line = f"{row['model']:<24} acc={row.get('accuracy', float('nan')):.3f}"
        if "auc" in row:
            line += f"  auc={row['auc']:.3f}  logloss={row.get('log_loss', float('nan')):.3f}"
        if "precision_top_decile" in row:
            line += f"  top10%prec={row['precision_top_decile']:.3f}"
        print(line)
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

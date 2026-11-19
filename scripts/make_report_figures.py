"""Generate the figures used in the README from the stored metrics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def plot_auc(results: list[dict], output: Path) -> None:
    labels = [row["model"].replace("_", " ") for row in results]
    values = [row.get("auc", 0.5) for row in results]

    fig, ax = plt.subplots(figsize=(7, 3.2))
    colors = ["#9aa0a6", "#4285f4", "#34a853", "#ea4335"][: len(values)]
    bars = ax.barh(labels, values, color=colors, height=0.55)
    ax.axvline(0.5, color="#202124", linestyle="--", linewidth=1)
    ax.set_xlim(0.40, 0.60)
    ax.set_xlabel("ROC AUC on the 2023-2026 holdout")
    ax.set_title("Every model sits on the coin-flip line")
    for bar, value in zip(bars, values, strict=False):
        ax.text(value + 0.002, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_confusion(matrix: dict, output: Path) -> None:
    grid = [
        [matrix["expected_0_predicted_0"], matrix["expected_0_predicted_1"]],
        [matrix["expected_1_predicted_0"], matrix["expected_1_predicted_1"]],
    ]

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    image = ax.imshow(grid, cmap="Blues")
    ax.set_xticks([0, 1], labels=["0", "1"])
    ax.set_yticks([0, 1], labels=["0", "1"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("BigQuery ML — confusion matrix")
    for row in range(2):
        for column in range(2):
            ax.text(
                column,
                row,
                f"{grid[row][column]:,}",
                ha="center",
                va="center",
                color="white" if grid[row][column] > 1000 else "#202124",
                fontsize=12,
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)
    baseline = load(ROOT / "reports" / "baseline_metrics.json")
    bqml = load(ROOT / "reports" / "bqml_metrics.json")

    results = [row for row in baseline["results"] if "auc" in row]
    results.append({"model": "bigquery_ml", "auc": bqml["metrics"]["roc_auc"]})

    plot_auc(results, ASSETS / "model_auc.png")
    plot_confusion(bqml["confusion_matrix"], ASSETS / "confusion_matrix.png")
    print(f"Wrote {ASSETS / 'model_auc.png'} and {ASSETS / 'confusion_matrix.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import os
from pathlib import Path


def plot_metrics(metrics_by_system: dict[str, list[dict[str, float]]], output_dir: str | Path) -> None:
    out = Path(output_dir)
    os.environ.setdefault("MPLCONFIGDIR", str(out.parent / ".matplotlib"))
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    plots = {
        "average_cost": "Average partial monitoring cost",
        "cumulative_cost": "Cumulative partial monitoring cost",
        "human_review_rate": "Human review rate",
        "acted_accuracy": "Accuracy on automatic actions",
        "f1_remove": "F1 for remove decisions",
    }

    for metric, title in plots.items():
        plt.figure(figsize=(8, 5))
        for system_name, rows in metrics_by_system.items():
            steps = [row["step"] for row in rows]
            values = [row[metric] for row in rows]
            plt.plot(steps, values, label=system_name)
        plt.title(title)
        plt.xlabel("Test messages")
        plt.ylabel(metric)
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / f"{metric}.png", dpi=160)
        plt.close()

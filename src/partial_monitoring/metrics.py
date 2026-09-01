from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RunningMetrics:
    total: int = 0
    human_reviews: int = 0
    acted: int = 0
    correct_actions: int = 0
    cumulative_cost: float = 0.0
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def update(self, action: str, prediction: int | None, label: int, cost: float) -> dict[str, float]:
        self.total += 1
        self.cumulative_cost += cost
        if action == "ask_human":
            self.human_reviews += 1
        else:
            self.acted += 1
            self.correct_actions += int(prediction == label)
            if prediction == 1 and label == 1:
                self.tp += 1
            elif prediction == 1 and label == 0:
                self.fp += 1
            elif prediction == 0 and label == 0:
                self.tn += 1
            elif prediction == 0 and label == 1:
                self.fn += 1
        return self.snapshot()

    def snapshot(self) -> dict[str, float]:
        precision = self.tp / max(1, self.tp + self.fp)
        recall = self.tp / max(1, self.tp + self.fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        return {
            "n": self.total,
            "coverage": self.acted / max(1, self.total),
            "human_review_rate": self.human_reviews / max(1, self.total),
            "acted_accuracy": self.correct_actions / max(1, self.acted),
            "average_cost": self.cumulative_cost / max(1, self.total),
            "cumulative_cost": self.cumulative_cost,
            "precision_remove": precision,
            "recall_remove": recall,
            "f1_remove": f1,
        }

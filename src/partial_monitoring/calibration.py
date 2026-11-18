from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import RunningMetrics


class SigmoidCalibrator:
    def __init__(self) -> None:
        self.model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> None:
        if len(np.unique(labels)) != 2:
            raise ValueError("Calibration requires both classes in its held-out conversations")
        # Calibration uses the natural prevalence, without training class weights.
        self.model.fit(np.asarray(scores).reshape(-1, 1), labels)

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(np.asarray(scores).reshape(-1, 1))[:, 1]

    def predict_intervals(
        self, scores: np.ndarray, std: np.ndarray, alpha: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be finite and nonnegative")
        p = self.predict(scores)
        a = self.predict(scores - alpha * std)
        b = self.predict(scores + alpha * std)
        return p, np.minimum(a, b), np.maximum(a, b)

    def parameters(self) -> dict[str, float]:
        scaler, regression = self.model.steps[0][1], self.model.steps[1][1]
        return {
            "score_mean": float(scaler.mean_[0]),
            "score_scale": float(scaler.scale_[0]),
            "coefficient": float(regression.coef_[0, 0]),
            "intercept": float(regression.intercept_[0]),
        }


@dataclass(frozen=True)
class SelectiveThresholds:
    keep_max: float = -1.0
    remove_min: float = 2.0

    def actions(self, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        actions = np.full(len(lower), 2, dtype=np.int64)
        actions[upper <= self.keep_max] = 0
        actions[lower >= self.remove_min] = 1
        return actions

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def summarize_actions(
    labels: np.ndarray,
    actions: np.ndarray,
    human_cost: float,
    false_positive_cost: float,
    false_negative_cost: float,
) -> dict[str, float]:
    keep, remove, human = actions == 0, actions == 1, actions == 2
    positive = labels == 1
    tp = int((remove & positive).sum())
    fp = int((remove & ~positive).sum())
    fn = int((keep & positive).sum())
    tn = int((keep & ~positive).sum())
    running = RunningMetrics(
        total=len(labels), human_reviews=int(human.sum()), acted=int((keep | remove).sum()),
        correct_actions=tp + tn, tp=tp, fp=fp, fn=fn, tn=tn,
        positives=int(positive.sum()), human_positives=int((human & positive).sum()),
        cumulative_cost=float(human.sum() * human_cost + fp * false_positive_cost + fn * false_negative_cost),
    )
    return running.snapshot()


def tune_thresholds(
    lower: np.ndarray,
    upper: np.ndarray,
    labels: np.ndarray,
    human_cost: float = 0.2,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 25.0,
    min_pipeline_recall: float = 0.98,
    min_remove_precision: float = 0.95,
    min_remove_examples: int = 100,
    objective: str = "coverage",
) -> tuple[SelectiveThresholds, dict[str, float]]:
    if not (len(lower) == len(upper) == len(labels)) or len(labels) == 0:
        raise ValueError("Tuning requires aligned, nonempty validation arrays")
    if not (np.isfinite(lower).all() and np.isfinite(upper).all()):
        raise ValueError("Probability bounds must be finite")
    if np.any(lower < 0) or np.any(upper > 1) or np.any(lower > upper):
        raise ValueError("Probability bounds must satisfy 0 <= lower <= upper <= 1")
    if not 0 <= min_pipeline_recall <= 1 or not 0 <= min_remove_precision <= 1:
        raise ValueError("Recall and precision targets must be in [0, 1]; zero disables the constraint")
    if min_remove_examples < 1 or objective not in ("coverage", "cost"):
        raise ValueError("Invalid minimum removal support or selection objective")
    positives = int(np.sum(labels))
    if positives == 0:
        raise ValueError("Threshold tuning requires positive validation examples")

    def select_boundary(values, descending, valid_boundary, max_misses=None):
        order = np.argsort(-values if descending else values, kind="stable")
        sorted_values = values[order]
        count = np.arange(1, len(labels) + 1)
        pos = np.cumsum(labels[order])
        errors = count - pos if descending else pos
        error_cost = false_positive_cost if descending else false_negative_cost
        delta_cost = errors * error_cost - count * human_cost
        # Equal scores are indivisible: never tune by splitting a probability tie.
        ends = np.r_[sorted_values[:-1] != sorted_values[1:], True]
        valid = ends & valid_boundary(sorted_values) & (delta_cost <= 1e-12)
        if descending:
            valid &= (count >= min_remove_examples) & (pos / count >= min_remove_precision)
        else:
            valid &= pos <= max_misses
        indices = np.flatnonzero(valid)
        if not len(indices):
            return 2.0 if descending else -1.0
        selected = indices[-1] if objective == "coverage" else indices[np.argmin(delta_cost[indices])]
        return float(sorted_values[selected])

    max_misses = int(np.floor((1 - min_pipeline_recall) * positives + 1e-9))
    keep_max = select_boundary(upper, False, lambda p: p < 0.5, max_misses)
    remove_min = select_boundary(lower, True, lambda p: p >= 0.5)
    thresholds = SelectiveThresholds(keep_max, remove_min)
    metrics = summarize_actions(labels, thresholds.actions(lower, upper), human_cost, false_positive_cost, false_negative_cost)
    return thresholds, metrics

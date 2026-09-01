from __future__ import annotations

from dataclasses import dataclass

ACTION_NAMES = ("keep", "remove", "ask_human")
ACTION_TO_INDEX = {action: index for index, action in enumerate(ACTION_NAMES)}
INDEX_TO_ACTION = {index: action for action, index in ACTION_TO_INDEX.items()}


@dataclass(frozen=True)
class PolicyDecision:
    action: str
    prediction: int | None
    confidence: float


def decide(probability_removed: float, threshold: float = 0.8) -> PolicyDecision:
    confidence = max(probability_removed, 1.0 - probability_removed)
    if confidence < threshold:
        return PolicyDecision(action="ask_human", prediction=None, confidence=confidence)
    prediction = int(probability_removed >= 0.5)
    return PolicyDecision(action="remove" if prediction else "keep", prediction=prediction, confidence=confidence)


def partial_monitoring_cost(
    action: str,
    prediction: int | None,
    label: int,
    human_cost: float = 0.2,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 1.0,
) -> float:
    if action == "ask_human":
        return human_cost
    if prediction == label:
        return 0.0
    if prediction == 1 and label == 0:
        return false_positive_cost
    if prediction == 0 and label == 1:
        return false_negative_cost
    raise ValueError(f"Invalid prediction/label pair: prediction={prediction}, label={label}")


def action_prediction(action: str) -> int | None:
    if action == "keep":
        return 0
    if action == "remove":
        return 1
    if action == "ask_human":
        return None
    raise ValueError(f"Unknown action: {action}")


def action_costs(
    label: int,
    human_cost: float = 0.2,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 1.0,
) -> list[float]:
    return [
        partial_monitoring_cost("keep", 0, label, human_cost, false_positive_cost, false_negative_cost),
        partial_monitoring_cost("remove", 1, label, human_cost, false_positive_cost, false_negative_cost),
        partial_monitoring_cost("ask_human", None, label, human_cost, false_positive_cost, false_negative_cost),
    ]


def action_rewards(
    label: int,
    human_cost: float = 0.2,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 1.0,
) -> list[float]:
    return [-cost for cost in action_costs(label, human_cost, false_positive_cost, false_negative_cost)]

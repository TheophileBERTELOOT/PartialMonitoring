from partial_monitoring.data import parse_label
from partial_monitoring.policy import action_rewards, decide, partial_monitoring_cost


def test_parse_label_common_values():
    assert parse_label("removed") == 1
    assert parse_label("true") == 1
    assert parse_label("0") == 0
    assert parse_label("approved") == 0


def test_policy_asks_when_uncertain():
    decision = decide(0.55, threshold=0.8)
    assert decision.action == "ask_human"
    assert decision.prediction is None
    assert partial_monitoring_cost(decision.action, decision.prediction, label=1) == 0.2


def test_policy_acts_when_confident():
    decision = decide(0.95, threshold=0.8)
    assert decision.action == "remove"
    assert decision.prediction == 1
    assert partial_monitoring_cost(decision.action, decision.prediction, label=0) == 1.0


def test_bandit_rewards_match_partial_monitoring_costs():
    assert action_rewards(label=1, human_cost=0.2) == [-1.0, -0.0, -0.2]
    assert action_rewards(label=0, human_cost=0.2) == [-0.0, -1.0, -0.2]


def test_asymmetric_costs_penalize_missed_toxicity():
    assert partial_monitoring_cost("keep", 0, label=1, false_negative_cost=25.0) == 25.0
    assert partial_monitoring_cost("remove", 1, label=0, false_positive_cost=2.0) == 2.0
    assert action_rewards(label=1, human_cost=0.2, false_negative_cost=25.0) == [-25.0, -0.0, -0.2]

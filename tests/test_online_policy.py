import numpy as np

from partial_monitoring.bandits import BanditDecision, LinUCB
from partial_monitoring.training import evaluate_offline_bandit
from partial_monitoring.data import MessageExample
from partial_monitoring.dataset import ModerationDataset
from partial_monitoring.training import train_and_evaluate_bandit


def test_online_learns_only_after_human_review():
    class Bandit:
        def __init__(self):
            self.labels = []

        def decide(self, x):
            action = "keep" if x[0] == 0 else "ask_human"
            return BanditDecision(action, 0 if action == "keep" else 2, [0., -1., -.2], [0., 0., 0.])

        def update_full_feedback(self, x, label, *args):
            self.labels.append(label)

    bandit = Bandit()
    events, _ = evaluate_offline_bandit(
        bandit, np.array([[0.], [1.], [0.]]), np.array([1, 0, 1]),
        [{}, {}, {}], .2, 1., 25., online=True, exploration_rate=0,
    )
    assert bandit.labels == [0]
    assert [event["feedback_observed"] for event in events] == [False, True, False]


def test_exploration_requests_feedback_and_linear_update_is_exact():
    bandit = LinUCB(2, alpha=0, ridge=2)
    bandit.fit(np.array([[1., 0.], [0., 1.]]), np.array([0, 1]), class_weighting="none")
    events, _ = evaluate_offline_bandit(
        bandit, np.array([[1., 1.], [2., 1.]]), np.array([1, 0]),
        [{}, {}], .2, 1., 25., online=True, exploration_rate=1, seed=42,
    )
    assert all(event["action"] == "ask_human" and event["exploration_human"] for event in events)
    np.testing.assert_allclose(bandit._inv_a, np.linalg.inv(bandit.a[0]))
    np.testing.assert_allclose(bandit._theta, np.linalg.solve(bandit.a[0], bandit.b.T).T)


def test_online_training_runs_without_calibration(monkeypatch):
    from partial_monitoring import hf_local, training

    class Model:
        def to(self, device):
            return self

    monkeypatch.setattr(hf_local, "load_local_model", lambda _: Model())
    monkeypatch.setattr(training, "extract_bert_features_with_model", lambda dataset, *a, **kw: (
        np.array([[1., float(e.label)] for e in dataset.examples]),
        np.array([e.label for e in dataset.examples]),
        [{"conversation_id": e.conversation_id} for e in dataset.examples],
    ))
    train = ModerationDataset([MessageExample("train", 0, "hello", 0, ()), MessageExample("train", 1, "bad", 1, ())])
    test = ModerationDataset([MessageExample("test", 0, "hello", 0, ())])
    _, events, _, artifacts = train_and_evaluate_bandit(
        "linucb", train, test, None, "unused", policy="online", exploration_rate=1,
    )
    assert artifacts["policy"]["feedback"] == "human_only"
    assert events[0]["decision_rule"] == "online"
    assert events[0]["feedback_observed"]

from __future__ import annotations

from .bandits import LinUCB, NeuralUCB, extract_bert_features_with_model, get_device
from .dataset import ModerationDataset
from .metrics import RunningMetrics
from .policy import action_prediction, partial_monitoring_cost


def build_bandit(
    algorithm: str,
    feature_dim: int,
    alpha: float,
    neural_hidden_dim: int,
    neural_dropout: float,
    neural_epochs: int,
    batch_size: int,
    learning_rate: float,
    mc_samples: int,
    device_name: str,
):
    if algorithm == "linucb":
        return LinUCB(feature_dim, alpha=alpha)
    if algorithm == "neuralucb":
        return NeuralUCB(
            feature_dim,
            alpha=alpha,
            hidden_dim=neural_hidden_dim,
            dropout=neural_dropout,
            epochs=neural_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            mc_samples=mc_samples,
            device_name=device_name,
        )
    raise ValueError(f"Unknown bandit algorithm: {algorithm}")


def train_and_evaluate_bandit(
    algorithm: str,
    train_dataset: ModerationDataset,
    test_dataset: ModerationDataset,
    tokenizer,
    bert_model_name: str,
    human_cost: float = 0.2,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 1.0,
    class_weighting: str = "balanced",
    positive_weight: float | None = None,
    alpha: float = 1.0,
    batch_size: int = 32,
    max_length: int = 256,
    neural_hidden_dim: int = 128,
    neural_dropout: float = 0.1,
    neural_epochs: int = 10,
    learning_rate: float = 1e-3,
    mc_samples: int = 20,
    device_name: str = "auto",
) -> tuple[dict[str, float], list[dict[str, object]], list[dict[str, float]]]:
    from .hf_local import load_local_model

    device = get_device(device_name)
    bert_model = load_local_model(bert_model_name).to(device)
    train_x, train_y, _ = extract_bert_features_with_model(
        train_dataset,
        tokenizer,
        bert_model,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )
    test_x, test_y, test_metadata = extract_bert_features_with_model(
        test_dataset,
        tokenizer,
        bert_model,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )
    bandit = build_bandit(
        algorithm,
        train_x.shape[1],
        alpha,
        neural_hidden_dim,
        neural_dropout,
        neural_epochs,
        batch_size,
        learning_rate,
        mc_samples,
        device_name,
    )
    training_summary = bandit.fit(
        train_x,
        train_y,
        human_cost=human_cost,
        false_positive_cost=false_positive_cost,
        false_negative_cost=false_negative_cost,
        class_weighting=class_weighting,
        positive_weight=positive_weight,
    )
    events, metrics = evaluate_offline_bandit(
        bandit,
        test_x,
        test_y,
        test_metadata,
        human_cost,
        false_positive_cost,
        false_negative_cost,
    )
    return training_summary, events, metrics


def evaluate_offline_bandit(
    bandit,
    features,
    labels,
    metadata: list[dict[str, object]],
    human_cost: float,
    false_positive_cost: float,
    false_negative_cost: float,
) -> tuple[list[dict[str, object]], list[dict[str, float]]]:
    running = RunningMetrics()
    events: list[dict[str, object]] = []
    metric_rows: list[dict[str, float]] = []

    for x, label, meta in zip(features, labels, metadata):
        decision = bandit.decide(x)
        prediction = action_prediction(decision.action)
        label_int = int(label)
        cost = partial_monitoring_cost(
            decision.action,
            prediction,
            label_int,
            human_cost,
            false_positive_cost,
            false_negative_cost,
        )
        snapshot = running.update(decision.action, prediction, label_int, cost)
        event = {
            **meta,
            "score_keep": decision.scores[0],
            "score_remove": decision.scores[1],
            "score_ask_human": decision.scores[2],
            "uncertainty_keep": decision.uncertainty[0],
            "uncertainty_remove": decision.uncertainty[1],
            "uncertainty_ask_human": decision.uncertainty[2],
            "action": decision.action,
            "prediction": "" if prediction is None else prediction,
            "label": label_int,
            "cost": cost,
        }
        events.append(event)
        metric_rows.append(snapshot | {"step": snapshot["n"]})

    return events, metric_rows

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score

from .bandits import BanditDecision, LinUCB, NeuralUCB, extract_bert_features_with_model, get_device
from .calibration import SigmoidCalibrator, SelectiveThresholds, summarize_actions, tune_thresholds
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
    ridge: float = 1.0,
):
    if algorithm == "linucb":
        return LinUCB(feature_dim, alpha=alpha, ridge=ridge)
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
    calibration_dataset: ModerationDataset | None = None,
    tuning_dataset: ModerationDataset | None = None,
    policy: str = "calibrated",
    ridge: float = 1.0,
    pooling: str = "mean",
    min_pipeline_recall: float = 0.98,
    min_remove_precision: float = 0.95,
    min_remove_examples: int = 100,
    selection_objective: str = "coverage",
    recall_targets: tuple[float, ...] = (0.99, 0.98, 0.97, 0.95),
    exploration_rate: float = 0.05,
    seed: int = 13,
) -> tuple[dict[str, float], list[dict[str, object]], list[dict[str, float]], dict[str, object]]:
    from .hf_local import load_local_model

    if policy not in ("calibrated", "reward", "online"):
        raise ValueError(f"Unknown policy: {policy}")
    if policy == "online" and algorithm != "linucb":
        raise ValueError("Online policy currently requires LinUCB")
    if policy == "online" and (
        {e.conversation_id for e in train_dataset.examples}
        & {e.conversation_id for e in test_dataset.examples}
    ):
        raise ValueError("Online training and test conversations must be disjoint")
    if not 0 <= exploration_rate <= 1:
        raise ValueError("exploration_rate must be in [0, 1]")
    if policy == "calibrated":
        if not calibration_dataset or not tuning_dataset:
            raise ValueError("Calibrated policy requires nonempty calibration and tuning datasets")
        partitions = [train_dataset, calibration_dataset, tuning_dataset, test_dataset]
        ids = [{e.conversation_id for e in dataset.examples} for dataset in partitions]
        if any(ids[i] & ids[j] for i in range(4) for j in range(i + 1, 4)):
            raise ValueError("Training, calibration, tuning and test conversations must be disjoint")
    if not train_dataset or not test_dataset:
        raise ValueError("Training and evaluation datasets must be nonempty")
    device = get_device(device_name)
    bert_model = load_local_model(bert_model_name).to(device)
    train_x, train_y, _ = extract_bert_features_with_model(
        train_dataset,
        tokenizer,
        bert_model,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        pooling=pooling,
    )
    validation_features = []
    if policy == "calibrated":
        for dataset in (calibration_dataset, tuning_dataset):
            validation_features.append(extract_bert_features_with_model(
                dataset, tokenizer, bert_model, batch_size, max_length, device, pooling=pooling,
            ))
    test_x, test_y, test_metadata = extract_bert_features_with_model(
        test_dataset,
        tokenizer,
        bert_model,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        pooling=pooling,
    )
    del bert_model
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
        ridge=ridge,
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
    del train_x
    artifacts: dict[str, object] = {}
    probabilities = lower = upper = thresholds = None
    if policy == "calibrated":
        calibration_x, calibration_y, calibration_meta = validation_features[0]
        tuning_x, tuning_y, tuning_meta = validation_features[1]
        calibration_scores, calibration_std = bandit.predict_removal_scores(calibration_x)
        calibrator = SigmoidCalibrator()
        calibrator.fit(calibration_scores, calibration_y)
        tuning_scores, tuning_std = bandit.predict_removal_scores(tuning_x)
        del calibration_x, tuning_x, validation_features
        _, tuning_lower, tuning_upper = calibrator.predict_intervals(tuning_scores, tuning_std, alpha)
        thresholds, validation_metrics = tune_thresholds(
            tuning_lower, tuning_upper, tuning_y, human_cost, false_positive_cost, false_negative_cost,
            min_pipeline_recall, min_remove_precision, min_remove_examples, selection_objective,
        )
        artifacts["policy"] = {
            "name": "calibrated", "alpha": alpha, **thresholds.as_dict(),
            "calibrator": calibrator.parameters(),
            "calibration_examples": len(calibration_y), "tuning_examples": len(tuning_y),
            "min_pipeline_recall": min_pipeline_recall, "min_remove_precision": min_remove_precision,
            "recall_constraint_enabled": min_pipeline_recall > 0,
            "precision_constraint_enabled": min_remove_precision > 0,
            "action_cost_guard_enabled": True,
            "min_remove_examples": min_remove_examples, "selection_objective": selection_objective,
            "uncertainty_note": "Propagated regression/MC-dropout heuristic, not a guaranteed confidence interval",
            "validation": validation_metrics,
        }
        artifacts["validation_tradeoffs"] = []
        for target in sorted(set(recall_targets) | {min_pipeline_recall}, reverse=True):
            candidate, candidate_metrics = tune_thresholds(
                tuning_lower, tuning_upper, tuning_y, human_cost, false_positive_cost, false_negative_cost,
                target, min_remove_precision, min_remove_examples, selection_objective,
            )
            artifacts["validation_tradeoffs"].append({
                "min_pipeline_recall": target, **candidate.as_dict(), **candidate_metrics,
            })
        artifacts["validation_scores"] = []
        for stage, scores, std, labels, metadata in (
            ("calibration", calibration_scores, calibration_std, calibration_y, calibration_meta),
            ("tuning", tuning_scores, tuning_std, tuning_y, tuning_meta),
        ):
            p, lo, hi = calibrator.predict_intervals(scores, std, alpha)
            artifacts["validation_scores"].extend(
                {**meta, "stage": stage, "raw_score": float(s), "raw_std": float(u),
                 "probability_removed": float(v), "probability_lower": float(a),
                 "probability_upper": float(b), "label": int(y)}
                for meta, s, u, v, a, b, y in zip(metadata, scores, std, p, lo, hi, labels)
            )
        test_scores, test_std = bandit.predict_removal_scores(test_x)
        probabilities, lower, upper = calibrator.predict_intervals(test_scores, test_std, alpha)
        artifacts["probability_diagnostics"] = {
            "brier_score": float(np.mean((probabilities - test_y) ** 2)),
            "average_precision": float(average_precision_score(test_y, probabilities)),
            "positive_prevalence": float(test_y.mean()),
        }
    else:
        artifacts["policy"] = {"name": policy, "alpha": alpha}
        if policy == "online":
            artifacts["policy"].update(exploration_rate=exploration_rate, feedback="human_only", seed=seed)
    artifacts["baselines"] = {
        name: summarize_actions(test_y, np.full(len(test_y), index), human_cost, false_positive_cost, false_negative_cost)
        for name, index in (("all_keep", 0), ("all_remove", 1), ("all_human", 2))
    }
    events, metrics = evaluate_offline_bandit(
        bandit,
        test_x,
        test_y,
        test_metadata,
        human_cost,
        false_positive_cost,
        false_negative_cost,
        probabilities=probabilities, lower=lower, upper=upper, thresholds=thresholds,
        online=policy == "online", exploration_rate=exploration_rate, seed=seed,
    )
    return training_summary, events, metrics, artifacts


def evaluate_offline_bandit(
    bandit,
    features,
    labels,
    metadata: list[dict[str, object]],
    human_cost: float,
    false_positive_cost: float,
    false_negative_cost: float,
    probabilities=None,
    lower=None,
    upper=None,
    thresholds: SelectiveThresholds | None = None,
    online: bool = False,
    exploration_rate: float = 0.05,
    seed: int = 13,
) -> tuple[list[dict[str, object]], list[dict[str, float]]]:
    running = RunningMetrics()
    events: list[dict[str, object]] = []
    metric_rows: list[dict[str, float]] = []
    rng = np.random.default_rng(seed)

    actions = thresholds.actions(lower, upper) if thresholds is not None else None
    for i, (x, label, meta) in enumerate(zip(features, labels, metadata)):
        if actions is None:
            decision = bandit.decide(x)
            explored = online and rng.random() < exploration_rate
            if explored:
                decision = BanditDecision("ask_human", 2, decision.scores, decision.uncertainty)
        else:
            explored = False
            p = float(probabilities[i])
            index = int(actions[i])
            decision = BanditDecision(
                ("keep", "remove", "ask_human")[index], index,
                [-false_negative_cost * p, -false_positive_cost * (1 - p), -human_cost],
                [false_negative_cost * float(upper[i] - lower[i]) / 2,
                 false_positive_cost * float(upper[i] - lower[i]) / 2, 0.0],
            )
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
        snapshot["cost_saving_vs_all_human"] = 1 - snapshot["average_cost"] / human_cost
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
            "probability_removed": "" if probabilities is None else float(probabilities[i]),
            "probability_lower": "" if lower is None else float(lower[i]),
            "probability_upper": "" if upper is None else float(upper[i]),
            "decision_rule": "online" if online else ("reward" if thresholds is None else "calibrated_thresholds"),
            "exploration_human": bool(explored),
            "feedback_observed": bool(online and decision.action == "ask_human"),
        }
        events.append(event)
        metric_rows.append(snapshot | {"step": snapshot["n"]})
        if online and decision.action == "ask_human":
            bandit.update_full_feedback(x, label_int, human_cost, false_positive_cost, false_negative_cost, 1.0)

    return events, metric_rows

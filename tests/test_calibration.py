import itertools

import numpy as np
import pytest

from partial_monitoring.calibration import SigmoidCalibrator, summarize_actions, tune_thresholds
from partial_monitoring.metrics import RunningMetrics


def test_calibration_uses_natural_prevalence_and_expands_uncertainty():
    calibrator = SigmoidCalibrator()
    labels = np.r_[np.zeros(90), np.ones(10)].astype(int)
    calibrator.fit(np.zeros(100), labels)
    np.testing.assert_allclose(calibrator.predict(np.zeros(2)), 0.1, atol=1e-3)
    calibrator.fit(np.linspace(-2, 2, 100), labels)
    scores, std = np.array([-1., 1.]), np.ones(2) * 0.5
    p, lower0, upper0 = calibrator.predict_intervals(scores, std, 0)
    _, lower1, upper1 = calibrator.predict_intervals(scores, std, 1)
    np.testing.assert_allclose(p, lower0)
    np.testing.assert_allclose(p, upper0)
    assert np.all(lower1 < lower0)
    assert np.all(upper1 > upper0)


def test_calibration_rejects_single_class():
    with pytest.raises(ValueError, match="both classes"):
        SigmoidCalibrator().fit(np.arange(5), np.zeros(5))


def test_threshold_search_matches_brute_force_feasible_coverage():
    rng = np.random.default_rng(12)
    p = np.r_[np.linspace(0.001, .2, 40), np.linspace(.6, .99, 20)]
    labels = rng.binomial(1, p)
    lower, upper = np.maximum(0, p - .01), np.minimum(1, p + .01)
    thresholds, metrics = tune_thresholds(
        lower, upper, labels, false_negative_cost=2, min_pipeline_recall=.8,
        min_remove_precision=.7, min_remove_examples=3,
    )
    best_coverage = 0
    for keep, remove in itertools.product(np.r_[-1, upper[upper < .5]], np.r_[2, lower[lower >= .5]]):
        actions = np.full(len(labels), 2)
        actions[upper <= keep] = 0
        actions[lower >= remove] = 1
        candidate = summarize_actions(labels, actions, .2, 1, 2)
        keep_count, remove_count = candidate["auto_keep"], candidate["auto_remove"]
        if candidate["pipeline_recall_perfect_human"] < .8:
            continue
        if candidate["false_negatives"] * 2 > keep_count * .2 + 1e-9:
            continue
        if remove_count and (remove_count < 3 or candidate["precision_remove"] < .7):
            continue
        if candidate["false_positives"] > remove_count * .2 + 1e-9:
            continue
        best_coverage = max(best_coverage, candidate["coverage"])
    assert metrics["coverage"] == best_coverage
    assert thresholds.keep_max < thresholds.remove_min
    assert metrics["average_cost"] <= .2


def test_thresholds_do_not_split_ties_or_remove_without_support():
    p = np.r_[np.full(10, .001), np.full(2, .99)]
    labels = np.r_[np.zeros(9), 1, np.ones(2)].astype(int)
    thresholds, metrics = tune_thresholds(p, p, labels, min_pipeline_recall=.99, min_remove_examples=3)
    assert thresholds.keep_max == -1
    assert thresholds.remove_min == 2
    assert metrics["human_review_rate"] == 1


def test_zero_recall_target_allows_economically_eligible_misses():
    p = np.linspace(.001, .2, 1000)
    labels = np.zeros(1000, dtype=int)
    labels[[0, -1]] = 1
    _, constrained = tune_thresholds(p, p, labels)
    _, unconstrained = tune_thresholds(p, p, labels, min_pipeline_recall=0, min_remove_precision=0)
    assert constrained["auto_keep"] == 0
    assert unconstrained["auto_keep"] == 1000
    assert unconstrained["pipeline_recall_perfect_human"] == 0
    assert unconstrained["average_cost"] == .05


def test_zero_precision_target_keeps_cost_guard_and_minimum_support():
    p = np.linspace(.6, .99, 100)
    labels = np.ones(100, dtype=int)
    labels[:10] = 0
    _, constrained = tune_thresholds(p, p, labels)
    _, relaxed = tune_thresholds(p, p, labels, min_pipeline_recall=0, min_remove_precision=0)
    _, insufficient_support = tune_thresholds(
        p, p, labels, min_pipeline_recall=0, min_remove_precision=0, min_remove_examples=101,
    )
    assert constrained["auto_remove"] == 0
    assert relaxed["auto_remove"] == 100
    assert relaxed["precision_remove"] == .9
    assert insufficient_support["auto_remove"] == 0
    _, costly = tune_thresholds(p, p, 1 - labels, min_pipeline_recall=0, min_remove_precision=0)
    assert costly["auto_remove"] == 0


def test_larger_uncertainty_causes_abstention_at_fixed_thresholds():
    p = np.array([.01, .02, .8, .9])
    labels = np.array([0, 0, 1, 1])
    thresholds, _ = tune_thresholds(p, p, labels, min_remove_examples=2)
    confident = thresholds.actions(p, p)
    uncertain = thresholds.actions(np.zeros(4), np.ones(4))
    assert np.all(confident != 2)
    assert np.all(uncertain == 2)


def test_global_metrics_include_human_positives():
    metrics = RunningMetrics()
    metrics.update("remove", 1, 1, 0)
    metrics.update("ask_human", None, 1, .2)
    result = metrics.update("keep", 0, 1, 25)
    assert result["selective_recall_remove"] == .5
    assert result["global_auto_remove_recall"] == 1 / 3
    assert result["pipeline_recall_perfect_human"] == 2 / 3
    assert result["keep_error_rate"] == 1
    assert result["false_negatives"] == 1

#!/usr/bin/env python3
"""Reselect thresholds from saved validation scores, then replay saved test probabilities."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from partial_monitoring.calibration import tune_thresholds
from partial_monitoring.training import evaluate_offline_bandit


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing directories are rejected.")
    parser.add_argument("--systems", nargs="+", choices=("message", "context"))
    parser.add_argument("--no-quality-constraints", action="store_true", help="Disable recall and precision targets; costs remain active.")
    parser.add_argument("--min-remove-examples", type=int)
    parser.add_argument("--selection-objective", choices=("coverage", "cost"))
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("--output-dir must be a new directory, distinct from the source run")
    if args.min_remove_examples is not None and args.min_remove_examples < 1:
        parser.error("--min-remove-examples must be positive")
    source = json.loads((args.run_dir / "summary.json").read_text())
    config = dict(source["config"])
    if config.get("policy") != "calibrated" or config.get("eval_split", "test") != "test":
        parser.error("The source must be a calibrated run evaluated on held-out test data")
    systems = list(dict.fromkeys(args.systems or source["systems"]))
    if any(name not in source["systems"] for name in systems):
        parser.error("Requested system is absent from the source run")
    if args.no_quality_constraints:
        config.update(min_pipeline_recall=0., min_remove_precision=0.)
    if args.min_remove_examples is not None:
        config["min_remove_examples"] = args.min_remove_examples
    if args.selection_objective is not None:
        config["selection_objective"] = args.selection_objective
    config.update(
        output_dir=str(args.output_dir), systems=systems, recall_targets=[],
        no_quality_constraints=(config["min_pipeline_recall"] == config["min_remove_precision"] == 0),
    )
    summary = {
        **source, "config": config, "systems": {}, "retuned_from": str(args.run_dir.resolve()),
        "retuning_note": "Frozen model, calibrator and probabilities. Thresholds use tuning labels only. Repeated test comparisons are exploratory.",
    }
    costs = {key: config[key] for key in ("human_cost", "false_positive_cost", "false_negative_cost")}
    output_rows = []
    args.output_dir.mkdir(parents=True)
    for name in systems:
        validation = pd.read_csv(args.run_dir / f"validation_scores_{name}.csv", dtype={"conversation_id": str})
        tuning = validation[validation.stage.eq("tuning")]
        thresholds, validation_metrics = tune_thresholds(
            tuning.probability_lower.to_numpy(), tuning.probability_upper.to_numpy(), tuning.label.to_numpy(),
            **costs, min_pipeline_recall=config["min_pipeline_recall"],
            min_remove_precision=config["min_remove_precision"],
            min_remove_examples=config["min_remove_examples"], objective=config["selection_objective"],
        )
        # Test probabilities and labels are read only after selecting the thresholds.
        test = pd.read_csv(args.run_dir / f"events_{name}.csv", dtype={"conversation_id": str})
        if set(test.conversation_id) & set(validation.conversation_id):
            raise ValueError("Validation and test conversations must be disjoint")
        if test.duplicated(["conversation_id", "message_index"]).any():
            raise ValueError("Duplicate test message IDs")
        events, metrics = evaluate_offline_bandit(
            None, np.empty((len(test), 0)), test.label.to_numpy(),
            test[["conversation_id", "message_index"]].to_dict("records"),
            **costs, probabilities=test.probability_removed.to_numpy(),
            lower=test.probability_lower.to_numpy(), upper=test.probability_upper.to_numpy(), thresholds=thresholds,
        )
        old = source["systems"][name]
        policy = {
            **old["policy"], **thresholds.as_dict(),
            "min_pipeline_recall": config["min_pipeline_recall"],
            "min_remove_precision": config["min_remove_precision"],
            "recall_constraint_enabled": config["min_pipeline_recall"] > 0,
            "precision_constraint_enabled": config["min_remove_precision"] > 0,
            "action_cost_guard_enabled": True,
            "min_remove_examples": config["min_remove_examples"],
            "selection_objective": config["selection_objective"], "validation": validation_metrics,
        }
        summary["systems"][name] = {**old, "policy": policy, "test": metrics[-1]}
        output_rows.append({"system": name, **metrics[-1]})
        write_csv(args.output_dir / f"events_{name}.csv", events)
        write_csv(args.output_dir / f"metrics_{name}.csv", metrics)
        write_csv(args.output_dir / f"validation_tradeoffs_{name}.csv", [{
            "min_pipeline_recall": config["min_pipeline_recall"], **thresholds.as_dict(), **validation_metrics,
        }])
        shutil.copy2(args.run_dir / f"validation_scores_{name}.csv", args.output_dir / f"validation_scores_{name}.csv")
    write_csv(args.output_dir / "metrics_summary.csv", output_rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({row["system"]: row for row in output_rows}, indent=2))


if __name__ == "__main__":
    main()

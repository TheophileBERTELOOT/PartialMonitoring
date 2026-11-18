#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from partial_monitoring.data import (
    CAD_LABEL_MAPPING, load_conversations_with_stats, resolve_split_strategy,
    split_by_conversation, split_dataset,
)
from partial_monitoring.dataset import ModerationDataset
from partial_monitoring.hf_local import load_local_tokenizer
from partial_monitoring.plotting import plot_metrics
from partial_monitoring.training import train_and_evaluate_bandit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate offline contextual bandits for moderation.")
    parser.add_argument("--data-dir", "--data-path", dest="data_dir", required=True, help="Directory of conversation CSVs, a CSV file, or a CAD TSV file.")
    parser.add_argument("--data-format", choices=("auto", "csv", "cad"), default="auto", help="Auto detects CAD for a TSV file; otherwise reads conversation CSVs.")
    parser.add_argument("--split-strategy", choices=("auto", "provided", "random"), default="auto", help="Auto preserves CAD's train/dev/test splits; CSVs use random conversation splits.")
    parser.add_argument("--output-dir", default="outputs/run", help="Directory for models, logs, and plots.")
    parser.add_argument("--text-column", default=None, help="Message text column. Inferred if omitted.")
    parser.add_argument("--label-column", default=None, help="Moderation label column. Inferred if omitted.")
    parser.add_argument("--context-window", type=int, default=6, help="Previous messages used by the context model.")
    parser.add_argument("--no-recursive", action="store_true", help="Only read CSV files directly inside --data-dir.")
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument(
        "--eval-split",
        choices=("test", "all"),
        default="test",
        help="Use test for held-out evaluation; all is a reward-policy diagnostic only.",
    )
    parser.add_argument("--algorithm", choices=("linucb", "neuralucb"), default="linucb")
    parser.add_argument("--policy", choices=("calibrated", "reward", "online"), default="calibrated")
    parser.add_argument("--exploration-rate", type=float, default=0.05, help="Online policy probability of requesting human feedback for exploration.")
    parser.add_argument("--systems", nargs="+", choices=("message", "context"), default=["message", "context"])
    parser.add_argument("--pooling", choices=("mean", "pooler"), default="mean")
    parser.add_argument("--ridge", type=float, default=10.0, help="LinUCB ridge regularization.")
    parser.add_argument("--calibration-fraction", type=float, default=0.5, help="Fraction of validation conversations reserved for calibration; the rest tune thresholds.")
    parser.add_argument("--min-pipeline-recall", type=float, default=0.98, help="Minimum tuning recall, assuming perfect human decisions.")
    parser.add_argument("--min-remove-precision", type=float, default=0.95)
    parser.add_argument(
        "--no-quality-constraints", action="store_true",
        help="Disable minimum recall and removal precision, keeping action costs and minimum removal support. Overrides both targets and disables the recall sweep.",
    )
    parser.add_argument("--min-remove-examples", type=int, default=100)
    parser.add_argument("--selection-objective", choices=("coverage", "cost"), default="coverage")
    parser.add_argument("--recall-targets", nargs="+", type=float, default=[0.99, 0.98, 0.97, 0.95], help="Additional validation-only tradeoffs; test uses min-pipeline-recall.")
    parser.add_argument(
        "--bert-model",
        required=True,
        help="Local Hugging Face BERT model directory. No download is attempted.",
    )
    parser.add_argument("--max-length", type=int, default=256, help="Maximum BERT token length.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--human-cost", type=float, default=0.2)
    parser.add_argument("--false-positive-cost", type=float, default=1.0, help="Cost of removing a message labeled to be kept.")
    parser.add_argument("--false-negative-cost", type=float, default=10.0, help="Cost of keeping a message labeled for moderation.")
    parser.add_argument("--class-weighting", choices=("balanced", "none"), default="none")
    parser.add_argument("--positive-weight", type=float, default=None, help="Manual positive class weight; overrides --class-weighting.")
    parser.add_argument("--alpha", type=float, default=0.0, help="Conservative uncertainty multiplier; zero uses calibrated probabilities directly.")
    parser.add_argument("--neural-hidden-dim", type=int, default=128)
    parser.add_argument("--neural-dropout", type=float, default=0.1)
    parser.add_argument("--neural-epochs", type=int, default=10)
    parser.add_argument("--mc-samples", type=int, default=20, help="MC-dropout samples for NeuralUCB uncertainty.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', or 'cuda'.")
    args = parser.parse_args()
    for name in ("human_cost", "false_positive_cost", "false_negative_cost", "ridge"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if not math.isfinite(args.alpha) or args.alpha < 0:
        parser.error("--alpha must be finite and nonnegative")
    if not 0 <= args.exploration_rate <= 1:
        parser.error("--exploration-rate must be in [0, 1]")
    if args.policy == "online" and (args.algorithm != "linucb" or args.eval_split != "test"):
        parser.error("--policy online requires --algorithm linucb and --eval-split test")
    if args.positive_weight is not None and (not math.isfinite(args.positive_weight) or args.positive_weight <= 0):
        parser.error("--positive-weight must be finite and positive")
    if not 0 < args.calibration_fraction < 1:
        parser.error("--calibration-fraction must be in (0, 1)")
    if any(not 0 <= target <= 1 for target in [args.min_pipeline_recall, args.min_remove_precision, *args.recall_targets]):
        parser.error("Recall and precision targets must be in [0, 1]; zero disables the constraint")
    if args.min_remove_examples < 1 or args.context_window < 0 or args.max_length < 3 or args.batch_size < 1:
        parser.error("Invalid minimum removal support, context window, max length, or batch size")
    if args.policy == "calibrated" and args.eval_split != "test":
        parser.error("--policy calibrated requires --eval-split test to prevent data leakage")
    if args.mc_samples < 2 or args.neural_epochs < 1 or not 0 <= args.neural_dropout < 1:
        parser.error("Neural policy needs mc-samples >= 2, epochs >= 1 and dropout in [0, 1)")
    if args.no_quality_constraints:
        if args.policy != "calibrated":
            parser.error("--no-quality-constraints applies to --policy calibrated")
        args.min_pipeline_recall = 0.0
        args.min_remove_precision = 0.0
        args.recall_targets = []
    return args


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples, load_stats = load_conversations_with_stats(
        args.data_dir,
        text_column=args.text_column,
        label_column=args.label_column,
        context_window=args.context_window,
        recursive=not args.no_recursive,
        data_format=args.data_format,
    )
    split_strategy = resolve_split_strategy(examples, args.split_strategy)
    train_examples, val_examples, test_examples = split_dataset(
        examples,
        strategy=split_strategy,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    eval_examples = examples if args.eval_split == "all" else test_examples
    calibration_examples, tuning_examples = [], []
    if args.policy == "calibrated":
        if len({e.conversation_id for e in val_examples}) < 2:
            raise ValueError("Calibrated policy needs at least two validation conversations; increase the dataset or val-fraction")
        calibration_examples, _, tuning_examples = split_by_conversation(
            val_examples, train_fraction=args.calibration_fraction, val_fraction=0.0, seed=args.seed + 1,
        )
        for name, partition in (("calibration", calibration_examples), ("tuning", tuning_examples)):
            if {e.label for e in partition} != {0, 1}:
                raise ValueError(f"The {name} split needs both classes; increase validation data or change the split seed")
    if not train_examples or not eval_examples:
        raise ValueError("Training and evaluation splits must be nonempty")

    tokenizer = load_local_tokenizer(args.bert_model)

    datasets = {
        name: ModerationDataset(eval_examples, use_context=(name == "context"))
        for name in dict.fromkeys(args.systems)
    }
    positive_messages = sum(example.label for example in examples)
    train_positive_messages = sum(example.label for example in train_examples)
    eval_positive_messages = sum(example.label for example in eval_examples)
    summary: dict[str, object] = {
        "config": vars(args),
        "dataset": {
            "data_format": load_stats.data_format,
            "split_strategy": split_strategy,
            "label_mapping": CAD_LABEL_MAPPING if load_stats.data_format == "cad" else None,
            "messages": len(examples),
            "positive_messages": positive_messages,
            "positive_rate": positive_messages / max(1, len(examples)),
            "train_messages": len(train_examples),
            "train_positive_messages": train_positive_messages,
            "train_positive_rate": train_positive_messages / max(1, len(train_examples)),
            "val_messages": len(val_examples),
            "calibration_messages": len(calibration_examples),
            "tuning_messages": len(tuning_examples),
            "calibration_conversations": len({e.conversation_id for e in calibration_examples}),
            "tuning_conversations": len({e.conversation_id for e in tuning_examples}),
            "test_messages": len(test_examples),
            "eval_messages": len(eval_examples),
            "eval_positive_messages": eval_positive_messages,
            "eval_positive_rate": eval_positive_messages / max(1, len(eval_examples)),
            "conversations": len({example.conversation_id for example in examples}),
            "train_conversations": len({example.conversation_id for example in train_examples}),
            "val_conversations": len({example.conversation_id for example in val_examples}),
            "test_conversations": len({example.conversation_id for example in test_examples}),
            "eval_conversations": len({example.conversation_id for example in eval_examples}),
            "files_found": load_stats.files_found,
            "files_read": load_stats.files_read,
            "files_with_messages": load_stats.files_with_messages,
            "files_without_messages": load_stats.files_without_messages,
            "files_skipped_empty_header": load_stats.files_skipped_empty_header,
            "rows_seen": load_stats.rows_seen,
            "messages_skipped_empty_text": load_stats.messages_skipped_empty_text,
            "messages_skipped_unavailable_text": load_stats.messages_skipped_unavailable_text,
            "rows_skipped_excluded_split": load_stats.rows_skipped_excluded_split,
            "duplicate_annotations_merged": load_stats.duplicate_annotations_merged,
        },
        "systems": {},
    }
    metrics_by_system: dict[str, list[dict[str, float]]] = {}
    summary_rows: list[dict[str, object]] = []

    for system_name, test_dataset in datasets.items():
        torch.manual_seed(args.seed)
        bandit_train_dataset = ModerationDataset(
            train_examples,
            use_context=(system_name == "context"),
        )
        history, events, metrics, artifacts = train_and_evaluate_bandit(
            args.algorithm,
            bandit_train_dataset,
            test_dataset,
            tokenizer,
            args.bert_model,
            human_cost=args.human_cost,
            false_positive_cost=args.false_positive_cost,
            false_negative_cost=args.false_negative_cost,
            class_weighting=args.class_weighting,
            positive_weight=args.positive_weight,
            alpha=args.alpha,
            batch_size=args.batch_size,
            max_length=args.max_length,
            neural_hidden_dim=args.neural_hidden_dim,
            neural_dropout=args.neural_dropout,
            neural_epochs=args.neural_epochs,
            learning_rate=args.learning_rate,
            mc_samples=args.mc_samples,
            device_name=args.device,
            calibration_dataset=ModerationDataset(calibration_examples, use_context=(system_name == "context")),
            tuning_dataset=ModerationDataset(tuning_examples, use_context=(system_name == "context")),
            policy=args.policy,
            ridge=args.ridge,
            pooling=args.pooling,
            min_pipeline_recall=args.min_pipeline_recall,
            min_remove_precision=args.min_remove_precision,
            min_remove_examples=args.min_remove_examples,
            selection_objective=args.selection_objective,
            recall_targets=tuple(args.recall_targets),
            exploration_rate=args.exploration_rate,
            seed=args.seed,
        )
        for artifact_name in ("validation_scores", "validation_tradeoffs"):
            write_csv(output_dir / f"{artifact_name}_{system_name}.csv", artifacts.pop(artifact_name, []))
        write_csv(output_dir / f"events_{system_name}.csv", events)
        write_csv(output_dir / f"metrics_{system_name}.csv", metrics)
        metrics_by_system[system_name] = metrics
        final_metrics = metrics[-1] if metrics else {}
        summary["systems"][system_name] = {"training": history, "test": final_metrics, **artifacts}
        summary_rows.append({"system": system_name, **history, **final_metrics})

    write_csv(output_dir / "metrics_summary.csv", summary_rows)
    plot_metrics(metrics_by_system, output_dir / "plots", human_cost=args.human_cost)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

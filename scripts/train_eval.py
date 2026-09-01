#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from partial_monitoring.data import load_conversations_with_stats, split_by_conversation
from partial_monitoring.dataset import ModerationDataset
from partial_monitoring.hf_local import load_local_tokenizer
from partial_monitoring.plotting import plot_metrics
from partial_monitoring.training import train_and_evaluate_bandit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate offline contextual bandits for moderation.")
    parser.add_argument("--data-dir", required=True, help="Directory containing one CSV per conversation.")
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
        help="Which messages receive decision logs and metrics. Training still uses train+val.",
    )
    parser.add_argument("--algorithm", choices=("linucb", "neuralucb"), default="linucb")
    parser.add_argument(
        "--bert-model",
        required=True,
        help="Local Hugging Face BERT model directory. No download is attempted.",
    )
    parser.add_argument("--max-length", type=int, default=256, help="Maximum BERT token length.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--human-cost", type=float, default=0.2)
    parser.add_argument("--false-positive-cost", type=float, default=1.0, help="Cost of removing a non-toxic message.")
    parser.add_argument("--false-negative-cost", type=float, default=10.0, help="Cost of keeping a toxic message.")
    parser.add_argument("--class-weighting", choices=("balanced", "none"), default="balanced")
    parser.add_argument("--positive-weight", type=float, default=None, help="Manual positive class weight; overrides --class-weighting.")
    parser.add_argument("--alpha", type=float, default=1.0, help="UCB exploration strength.")
    parser.add_argument("--neural-hidden-dim", type=int, default=128)
    parser.add_argument("--neural-dropout", type=float, default=0.1)
    parser.add_argument("--neural-epochs", type=int, default=10)
    parser.add_argument("--mc-samples", type=int, default=20, help="MC-dropout samples for NeuralUCB uncertainty.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', or 'cuda'.")
    return parser.parse_args()


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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples, load_stats = load_conversations_with_stats(
        args.data_dir,
        text_column=args.text_column,
        label_column=args.label_column,
        context_window=args.context_window,
        recursive=not args.no_recursive,
    )
    train_examples, val_examples, test_examples = split_by_conversation(
        examples,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    eval_examples = examples if args.eval_split == "all" else test_examples

    tokenizer = load_local_tokenizer(args.bert_model)

    datasets = {
        "message": ModerationDataset(eval_examples, use_context=False),
        "context": ModerationDataset(eval_examples, use_context=True),
    }
    positive_messages = sum(example.label for example in examples)
    train_positive_messages = sum(example.label for example in train_examples)
    eval_positive_messages = sum(example.label for example in eval_examples)
    summary: dict[str, object] = {
        "config": vars(args),
        "dataset": {
            "messages": len(examples),
            "positive_messages": positive_messages,
            "positive_rate": positive_messages / max(1, len(examples)),
            "train_messages": len(train_examples),
            "train_positive_messages": train_positive_messages,
            "train_positive_rate": train_positive_messages / max(1, len(train_examples)),
            "val_messages": len(val_examples),
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
        },
        "systems": {},
    }
    metrics_by_system: dict[str, list[dict[str, float]]] = {}
    summary_rows: list[dict[str, object]] = []

    for system_name, test_dataset in datasets.items():
        bandit_train_dataset = ModerationDataset(
            train_examples + val_examples,
            use_context=(system_name == "context"),
        )
        history, events, metrics = train_and_evaluate_bandit(
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
        )
        write_csv(output_dir / f"events_{system_name}.csv", events)
        write_csv(output_dir / f"metrics_{system_name}.csv", metrics)
        metrics_by_system[system_name] = metrics
        final_metrics = metrics[-1] if metrics else {}
        summary["systems"][system_name] = {"training": history, "test": final_metrics}
        summary_rows.append({"system": system_name, **history, **final_metrics})

    write_csv(output_dir / "metrics_summary.csv", summary_rows)
    plot_metrics(metrics_by_system, output_dir / "plots")
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

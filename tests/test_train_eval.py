import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizer

from partial_monitoring.data import MessageExample
from partial_monitoring.dataset import ModerationDataset
from partial_monitoring.training import train_and_evaluate_bandit


def test_calibrated_training_rejects_overlapping_conversations_before_loading_model():
    dataset = ModerationDataset([MessageExample("same", 0, "hello", 1, ())])
    with pytest.raises(ValueError, match="disjoint"):
        train_and_evaluate_bandit(
            "linucb", dataset, dataset, None, "missing/model",
            calibration_dataset=dataset, tuning_dataset=dataset,
        )


@pytest.fixture
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("algorithm", ["linucb", "neuralucb"])
@pytest.mark.parametrize("data_format", ["csv", "cad"])
@pytest.mark.parametrize("quality_constraints", [True, False])
def test_cli_end_to_end_with_local_tiny_bert(tmp_path, monkeypatch, single_thread, algorithm, data_format, quality_constraints):
    model_dir = tmp_path / "bert"
    model_dir.mkdir()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "good", "bad", "message", "ctx", "[", "]"]
    (model_dir / "vocab.txt").write_text("\n".join(vocab))
    tokenizer = BertTokenizer(str(model_dir / "vocab.txt"))
    tokenizer.save_pretrained(model_dir)
    model = BertModel(BertConfig(
        vocab_size=len(vocab), hidden_size=8, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=16, max_position_embeddings=32,
    ))
    model.save_pretrained(model_dir)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    data_path = data_dir
    if data_format == "csv":
        for index in range(64):
            (data_dir / f"conversation_{index}.csv").write_text("text,label\ngood message,0\nbad message,1\n")
    else:
        data_path = data_dir / "cad.tsv"
        with data_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, delimiter="\t", fieldnames=[
                "info_id", "info_thread.id", "meta_created_utc", "meta_text", "annotation_Primary", "split",
            ])
            writer.writeheader()
            for index in range(64):
                split = "train" if index < 32 else "dev" if index < 48 else "test"
                for timestamp, text, label in [(2, "bad message", "PersonDirectedAbuse"), (1, "good message", "Neutral")]:
                    writer.writerow({
                        "info_id": f"{index}-{timestamp}", "info_thread.id": str(index),
                        "meta_created_utc": timestamp, "meta_text": text,
                        "annotation_Primary": label, "split": split,
                    })
    output_dir = tmp_path / "results"
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_eval.py"
    spec = importlib.util.spec_from_file_location("train_eval", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    argv = [
        str(script), "--data-path", str(data_path), "--bert-model", str(model_dir),
        "--output-dir", str(output_dir), "--train-fraction", ".5", "--val-fraction", ".25",
        "--max-length", "24", "--batch-size", "16", "--device", "cpu",
        "--min-remove-examples", "2", "--false-negative-cost", "25", "--alpha", "1",
        "--algorithm", algorithm, "--neural-epochs", "1", "--mc-samples", "3",
    ]
    if not quality_constraints:
        argv.append("--no-quality-constraints")
    monkeypatch.setattr(sys, "argv", argv)
    module.main()
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["dataset"]["data_format"] == data_format
    assert summary["dataset"]["split_strategy"] == ("provided" if data_format == "cad" else "random")
    assert summary["dataset"]["train_messages"] == 64
    assert summary["dataset"]["calibration_messages"] == 16
    assert summary["dataset"]["tuning_messages"] == 16
    for system in ("message", "context"):
        result = summary["systems"][system]
        assert result["training"]["train_examples"] == 64
        assert result["test"]["n"] == 32
        assert result["baselines"]["all_human"]["average_cost"] == .2
        assert result["policy"]["recall_constraint_enabled"] == quality_constraints
        assert result["policy"]["precision_constraint_enabled"] == quality_constraints
        assert result["policy"]["action_cost_guard_enabled"] is True
        if quality_constraints:
            assert result["policy"]["validation"]["pipeline_recall_perfect_human"] >= .98
        else:
            assert result["policy"]["min_pipeline_recall"] == 0
            assert result["policy"]["min_remove_precision"] == 0
            assert summary["config"]["recall_targets"] == []
        with (output_dir / f"validation_scores_{system}.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        calibration_ids = {r["conversation_id"] for r in rows if r["stage"] == "calibration"}
        tuning_ids = {r["conversation_id"] for r in rows if r["stage"] == "tuning"}
        with (output_dir / f"events_{system}.csv").open() as handle:
            events = list(csv.DictReader(handle))
        test_ids = {r["conversation_id"] for r in events}
        if data_format == "cad":
            assert test_ids == {str(index) for index in range(48, 64)}
            assert calibration_ids | tuning_ids == {str(index) for index in range(32, 48)}
        assert not calibration_ids & tuning_ids
        assert not (calibration_ids | tuning_ids) & test_ids
        assert all(r["decision_rule"] == "calibrated_thresholds" for r in events)
        assert all(0 <= float(r["probability_lower"]) <= float(r["probability_removed"]) <= float(r["probability_upper"]) <= 1 for r in events)
        with (output_dir / f"validation_tradeoffs_{system}.csv").open() as handle:
            tradeoffs = list(csv.DictReader(handle))
        assert len(tradeoffs) == (4 if quality_constraints else 1)
    assert (output_dir / "plots" / "pipeline_recall_perfect_human.png").exists()

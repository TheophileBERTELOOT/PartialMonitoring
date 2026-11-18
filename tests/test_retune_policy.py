import csv
import json
import subprocess
import sys
from pathlib import Path


def write_scores(path, probabilities, labels, conversation, stage=None):
    rows = []
    for index, (probability, label) in enumerate(zip(probabilities, labels)):
        row = {
            "conversation_id": conversation, "message_index": index, "label": label,
            "probability_removed": probability, "probability_lower": probability, "probability_upper": probability,
        }
        if stage is not None:
            row["stage"] = stage
        rows.append(row)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_retuning_uses_validation_only_and_keeps_original_files(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    config = {
        "policy": "calibrated", "eval_split": "test",
        "human_cost": .2, "false_positive_cost": 1., "false_negative_cost": 25.,
        "min_pipeline_recall": .98, "min_remove_precision": .95,
        "min_remove_examples": 100, "selection_objective": "coverage",
    }
    original_summary = json.dumps({"config": config, "systems": {"message": {"policy": {"name": "calibrated"}}}})
    (source / "summary.json").write_text(original_summary)
    write_scores(source / "validation_scores_message.csv", [.1, .2, .8, .9], [0, 0, 1, 1], "validation", "tuning")
    script = Path(__file__).resolve().parents[1] / "scripts" / "retune_policy.py"
    results = []
    for name, labels in (("first", [0, 1]), ("second", [1, 0])):
        write_scores(source / "events_message.csv", [.1, .9], labels, "test")
        destination = tmp_path / name
        subprocess.run([
            sys.executable, str(script), "--run-dir", str(source), "--output-dir", str(destination),
            "--no-quality-constraints", "--min-remove-examples", "1",
        ], check=True, capture_output=True, text=True)
        results.append(json.loads((destination / "summary.json").read_text()))
    policies = [result["systems"]["message"]["policy"] for result in results]
    assert policies[0] == policies[1]
    assert policies[0]["keep_max"] == .2
    assert policies[0]["remove_min"] == .8
    assert policies[0]["recall_constraint_enabled"] is False
    assert policies[0]["precision_constraint_enabled"] is False
    assert results[0]["systems"]["message"]["test"]["average_cost"] == 0
    assert results[1]["systems"]["message"]["test"]["average_cost"] == 13
    assert (source / "summary.json").read_text() == original_summary

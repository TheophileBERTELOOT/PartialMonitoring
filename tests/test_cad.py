import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from partial_monitoring.data import (
    MessageExample, load_conversations, load_conversations_with_stats, split_dataset,
)


def row(entry_id, text, label="Neutral", split="train", thread="one", timestamp=1):
    return {
        "info_id": entry_id, "info_thread.id": thread, "meta_text": text,
        "annotation_Primary": label, "split": split, "meta_created_utc": timestamp,
    }


def write_cad(tmp_path, rows):
    path = tmp_path / "cad.tsv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.mark.parametrize("label,expected", [
    ("Neutral", 0), ("CounterSpeech", 0), ("Slur", 0),
    ("IdentityDirectedAbuse", 1), ("AffiliationDirectedAbuse", 1), ("PersonDirectedAbuse", 1),
])
def test_cad_binary_labels(tmp_path, label, expected):
    path = write_cad(tmp_path, [row("entry", "message", label)])
    assert load_conversations(path)[0].label == expected


def test_cad_merges_annotations_before_building_chronological_context(tmp_path):
    path = write_cad(tmp_path, [
        row("reply", "latest", timestamp=3),
        row("root-post", "body", timestamp=1),
        row("middle", "first [linebreak] second", "CounterSpeech", timestamp=2),
        row("middle", "first [linebreak] second", "PersonDirectedAbuse", timestamp=2),
        row("root-title", "title", timestamp=1),
        row("excluded", "ignored bot", split="exclude_bot", timestamp=2),
        row("empty", "  ", timestamp=2),
        row("deleted", "[deleted]", timestamp=2),
        row("other", "different conversation", thread="two"),
    ])
    examples, stats = load_conversations_with_stats(path, context_window=2)
    thread = [example for example in examples if example.conversation_id == "one"]
    assert [e.text for e in thread] == ["title", "body", "first \n second", "latest"]
    assert [e.message_index for e in thread] == [0, 1, 2, 3]
    assert thread[2].label == 1
    assert thread[2].context == ("title", "body")
    assert thread[3].context == ("body", "first \n second")
    assert examples[-1].context == ()
    assert all(e.context == () for e in load_conversations(path, context_window=0))
    assert stats.data_format == "cad"
    assert stats.messages_loaded == 5
    assert stats.rows_seen == 9
    assert stats.duplicate_annotations_merged == 1
    assert stats.rows_skipped_excluded_split == 1
    assert stats.messages_skipped_empty_text == 1
    assert stats.messages_skipped_unavailable_text == 1


def test_cad_preserves_official_splits_and_keeps_threads_disjoint(tmp_path):
    path = write_cad(tmp_path, [
        row(f"{split}-{index}", "message", split=split, thread=f"{split}-{index}")
        for split in ("train", "dev", "test") for index in range(2)
    ])
    examples = load_conversations(path)
    train, val, test = split_dataset(examples, seed=99)
    assert {e.conversation_id for e in train} == {"train-0", "train-1"}
    assert {e.conversation_id for e in val} == {"dev-0", "dev-1"}
    assert {e.conversation_id for e in test} == {"test-0", "test-1"}
    assert (train, val, test) == split_dataset(examples, seed=1)
    random_parts = split_dataset(examples, strategy="random", train_fraction=.5, val_fraction=1/6)
    assert [len(part) for part in random_parts] == [3, 1, 2]
    ids = [{e.conversation_id for e in part} for part in random_parts]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])


@pytest.mark.parametrize("change,match", [
    ({"split": "test"}, "multiple splits"),
    ({"meta_text": "different"}, "Conflicting CAD annotations"),
    ({"meta_created_utc": 2}, "Conflicting CAD annotations"),
    ({"annotation_Primary": "unknown"}, "unsupported CAD label"),
    ({"split": "unknown"}, "unsupported CAD split"),
])
def test_cad_rejects_conflicting_or_unsupported_annotations(tmp_path, change, match):
    first = row("entry", "message")
    path = write_cad(tmp_path, [first, {**first, **change}])
    with pytest.raises(ValueError, match=match):
        load_conversations(path)


def test_provided_splits_require_complete_assignments_without_leakage():
    assigned = MessageExample("same", 0, "train", 0, (), split="train")
    missing = MessageExample("other", 0, "missing", 0, ())
    conflicting = MessageExample("same", 1, "test", 0, (), split="test")
    with pytest.raises(ValueError, match="assignment for every message"):
        split_dataset([assigned, missing])
    with pytest.raises(ValueError, match="multiple splits"):
        split_dataset([assigned, conflicting])
    with pytest.raises(ValueError, match="assignment for every message"):
        split_dataset([missing], strategy="provided")


def test_single_csv_file_matches_directory_loader(tmp_path):
    path = tmp_path / "one.csv"
    path.write_text("text,label\nhello,0\nreply,1\n")
    assert load_conversations(path) == load_conversations(tmp_path)


def test_cad_audit_reports_deduplicated_messages_and_splits(tmp_path):
    path = write_cad(tmp_path, [
        row("entry", "message", "CounterSpeech"),
        row("entry", "message", "PersonDirectedAbuse"),
        row("excluded", "bot", split="exclude_bot"),
    ])
    script = Path(__file__).resolve().parents[1] / "scripts" / "audit_data.py"
    result = subprocess.run(
        [sys.executable, str(script), "--data-path", str(path)],
        check=True, capture_output=True, text=True,
    )
    audit = json.loads(result.stdout)
    assert audit["loaded_messages"] == audit["positive_messages"] == 1
    assert audit["duplicate_annotations_merged"] == 1
    assert audit["rows_skipped_excluded_split"] == 1
    assert audit["splits"]["train"]["messages"] == 1
    assert audit["splits"]["val"]["messages"] == 0

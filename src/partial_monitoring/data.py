from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TEXT_COLUMN_CANDIDATES = ("text", "body", "message", "comment", "content",'body')
LABEL_COLUMN_CANDIDATES = ("label", "moderated", "removed", "is_removed", "mod_removed")
POSITIVE_LABELS = {"1", "true", "t", "yes", "y", "removed", "moderated", "delete", "deleted"}
NEGATIVE_LABELS = {"0", "false", "f", "no", "n", "kept", "keep", "approved", "ok"}
CAD_LABEL_MAPPING = {
    "IdentityDirectedAbuse": 1,
    "AffiliationDirectedAbuse": 1,
    "PersonDirectedAbuse": 1,
    "Neutral": 0,
    "CounterSpeech": 0,
    "Slur": 0,  # CAD's non-hateful uses of slurs.
}
CAD_SPLITS = {"train": "train", "dev": "val", "test": "test"}


@dataclass(frozen=True)
class MessageExample:
    conversation_id: str
    message_index: int
    text: str
    label: int
    context: tuple[str, ...]
    split: str | None = None


@dataclass(frozen=True)
class LoadStats:
    files_found: int
    files_read: int
    files_with_messages: int
    files_without_messages: int
    files_skipped_empty_header: int
    messages_loaded: int
    rows_seen: int
    messages_skipped_empty_text: int
    data_format: str = "csv"
    rows_skipped_excluded_split: int = 0
    duplicate_annotations_merged: int = 0
    messages_skipped_unavailable_text: int = 0


@dataclass
class _CADEntry:
    id: str
    text: str
    timestamp: int
    label: int
    split: str


def find_column(header: Iterable[str], explicit: str | None, candidates: tuple[str, ...]) -> str:
    columns = list(header)
    lower_to_original = {column.lower(): column for column in columns}
    if explicit:
        if explicit in columns:
            return explicit
        lowered = explicit.lower()
        if lowered in lower_to_original:
            return lower_to_original[lowered]
        raise ValueError(f"Column '{explicit}' not found. Available columns: {columns}")
    for candidate in candidates:
        if candidate in lower_to_original:
            return lower_to_original[candidate]
    raise ValueError(f"Could not infer column. Available columns: {columns}")


def parse_label(value: object) -> int:
    if value is None:
        raise ValueError("Missing label value")
    text = str(value).strip().lower()
    if text in POSITIVE_LABELS:
        return 1
    if text in NEGATIVE_LABELS:
        return 0
    try:
        return int(float(text) > 0)
    except ValueError as exc:
        raise ValueError(f"Unsupported label value: {value!r}") from exc


def load_conversations(
    data_dir: str | Path,
    text_column: str | None = None,
    label_column: str | None = None,
    context_window: int = 6,
    recursive: bool = True,
    data_format: str = "auto",
) -> list[MessageExample]:
    examples, _ = load_conversations_with_stats(
        data_dir,
        text_column=text_column,
        label_column=label_column,
        context_window=context_window,
        recursive=recursive,
        data_format=data_format,
    )
    return examples


def load_conversations_with_stats(
    data_dir: str | Path,
    text_column: str | None = None,
    label_column: str | None = None,
    context_window: int = 6,
    recursive: bool = True,
    data_format: str = "auto",
) -> tuple[list[MessageExample], LoadStats]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data path does not exist: {root}")
    if context_window < 0:
        raise ValueError("context_window must be nonnegative")
    if resolve_data_format(root, data_format) == "cad":
        return load_cad_with_stats(root, text_column, label_column, context_window)

    examples: list[MessageExample] = []
    relative_root = root.parent if root.is_file() else root
    search_root = [root] if root.is_file() else (root.rglob("*") if recursive else root.glob("*"))
    csv_paths = sorted(path for path in search_root if path.is_file() and path.suffix.lower() == ".csv")
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {root}")

    files_read = 0
    files_with_messages = 0
    files_skipped_empty_header = 0
    rows_seen = 0
    messages_skipped_empty_text = 0

    for csv_path in csv_paths:
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                files_skipped_empty_header += 1
                continue
            if reader.fieldnames == ['']:
                files_skipped_empty_header += 1
                continue
            text_col = find_column(reader.fieldnames, text_column, TEXT_COLUMN_CANDIDATES)
            label_col = find_column(reader.fieldnames, label_column, LABEL_COLUMN_CANDIDATES)
            previous: list[str] = []
            file_messages = 0
            for index, row in enumerate(reader):
                rows_seen += 1
                text = str(row.get(text_col, "") or "").strip()
                if not text:
                    messages_skipped_empty_text += 1
                    continue
                label = parse_label(row.get(label_col))
                context = tuple(previous[-context_window:]) if context_window > 0 else ()
                examples.append(
                    MessageExample(
                        conversation_id=csv_path.relative_to(relative_root).with_suffix("").as_posix(),
                        message_index=index,
                        text=text,
                        label=label,
                        context=context,
                    )
                )
                previous.append(text)
                file_messages += 1
            files_read += 1
            if file_messages:
                files_with_messages += 1

    stats = LoadStats(
        files_found=len(csv_paths),
        files_read=files_read,
        files_with_messages=files_with_messages,
        files_without_messages=files_read - files_with_messages,
        files_skipped_empty_header=files_skipped_empty_header,
        messages_loaded=len(examples),
        rows_seen=rows_seen,
        messages_skipped_empty_text=messages_skipped_empty_text,
    )
    return examples, stats


def resolve_data_format(data_path: str | Path, data_format: str = "auto") -> str:
    if data_format not in {"auto", "csv", "cad"}:
        raise ValueError(f"Unsupported data format: {data_format!r}")
    if data_format != "auto":
        return data_format
    path = Path(data_path)
    return "cad" if path.is_file() and path.suffix.lower() == ".tsv" else "csv"


def load_cad_with_stats(
    path: Path,
    text_column: str | None = None,
    label_column: str | None = None,
    context_window: int = 6,
) -> tuple[list[MessageExample], LoadStats]:
    """Read CAD annotations once per entry, keeping official splits and past-only context."""
    if not path.is_file():
        raise ValueError(f"CAD input must be a TSV file, not a directory: {path}")
    entries: dict[tuple[str, str], _CADEntry] = {}
    thread_splits: dict[str, str] = {}
    rows_seen = excluded = duplicates = empty = unavailable = 0
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        header = reader.fieldnames or []
        text_col = find_column(header, text_column, ("meta_text",))
        label_col = find_column(header, label_column, ("annotation_primary",))
        required = {"info_id", "info_thread.id", "meta_created_utc", "split"}
        if missing := required.difference(header):
            raise ValueError(f"Missing CAD columns: {sorted(missing)}")
        for row_index, row in enumerate(reader, start=2):
            rows_seen += 1
            raw_split = str(row.get("split") or "").strip()
            if raw_split.startswith("exclude_"):
                excluded += 1
                continue
            if raw_split not in CAD_SPLITS:
                raise ValueError(f"{path}:{row_index}: unsupported CAD split {raw_split!r}")
            split = CAD_SPLITS[raw_split]
            thread_id = str(row.get("info_thread.id") or "").strip()
            entry_id = str(row.get("info_id") or "").strip()
            if not thread_id or not entry_id or thread_id == "NA" or entry_id == "NA":
                raise ValueError(f"{path}:{row_index}: missing CAD thread or entry ID")
            previous_split = thread_splits.setdefault(thread_id, split)
            if previous_split != split:
                raise ValueError(f"CAD thread {thread_id!r} occurs in multiple splits: {previous_split}, {split}")
            text = str(row.get(text_col) or "").replace("[linebreak]", "\n").strip()
            if not text:
                empty += 1
                continue
            if text.lower() in {"[deleted]", "[removed]"}:
                unavailable += 1
                continue
            raw_label = str(row.get(label_col) or "").strip()
            if raw_label not in CAD_LABEL_MAPPING:
                raise ValueError(f"{path}:{row_index}: unsupported CAD label {raw_label!r}")
            label = CAD_LABEL_MAPPING[raw_label]
            try:
                timestamp = int(row["meta_created_utc"])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{row_index}: invalid CAD timestamp") from exc
            key = thread_id, entry_id
            if key in entries:
                entry = entries[key]
                if (entry.text, entry.timestamp) != (text, timestamp):
                    raise ValueError(f"Conflicting CAD annotations for entry {entry_id!r} in thread {thread_id!r}")
                entry.label = max(entry.label, label)
                duplicates += 1
            else:
                entries[key] = _CADEntry(entry_id, text, timestamp, label, split)

    threads: dict[str, list[_CADEntry]] = defaultdict(list)
    for (thread_id, _), entry in entries.items():
        threads[thread_id].append(entry)
    examples: list[MessageExample] = []
    for thread_id in sorted(threads):
        def order(entry: _CADEntry) -> tuple[int, int, str]:
            entry_id = entry.id
            kind = 0 if entry_id.endswith("-title") else 1 if entry_id.endswith("-post") else 2
            return entry.timestamp, kind, entry_id

        previous: list[str] = []
        for index, entry in enumerate(sorted(threads[thread_id], key=order)):
            examples.append(MessageExample(
                conversation_id=thread_id, message_index=index, text=entry.text,
                label=entry.label, context=tuple(previous[-context_window:]) if context_window else (),
                split=entry.split,
            ))
            previous.append(entry.text)
    stats = LoadStats(
        files_found=1, files_read=1, files_with_messages=int(bool(examples)),
        files_without_messages=int(not examples), files_skipped_empty_header=0,
        messages_loaded=len(examples), rows_seen=rows_seen, messages_skipped_empty_text=empty,
        data_format="cad", rows_skipped_excluded_split=excluded,
        duplicate_annotations_merged=duplicates, messages_skipped_unavailable_text=unavailable,
    )
    return examples, stats


def resolve_split_strategy(examples: list[MessageExample], strategy: str = "auto") -> str:
    if strategy not in {"auto", "random", "provided"}:
        raise ValueError(f"Unsupported split strategy: {strategy!r}")
    if strategy == "auto":
        return "provided" if any(example.split is not None for example in examples) else "random"
    return strategy


def split_dataset(
    examples: list[MessageExample],
    strategy: str = "auto",
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    seed: int = 13,
) -> tuple[list[MessageExample], list[MessageExample], list[MessageExample]]:
    """Use official partitions when present; otherwise split whole conversations randomly."""
    if resolve_split_strategy(examples, strategy) == "random":
        return split_by_conversation(examples, train_fraction, val_fraction, seed)
    partitions: dict[str, list[MessageExample]] = {"train": [], "val": [], "test": []}
    assigned: dict[str, str] = {}
    for example in examples:
        if example.split not in partitions:
            raise ValueError("Provided splits require a train/val/test assignment for every message")
        previous_split = assigned.setdefault(example.conversation_id, example.split)
        if previous_split != example.split:
            raise ValueError(f"Conversation {example.conversation_id!r} occurs in multiple splits")
        partitions[example.split].append(example)
    return partitions["train"], partitions["val"], partitions["test"]


def split_by_conversation(
    examples: list[MessageExample],
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    seed: int = 13,
) -> tuple[list[MessageExample], list[MessageExample], list[MessageExample]]:
    import random

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    if train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction must be lower than 1")

    conversation_ids = sorted({example.conversation_id for example in examples})
    rng = random.Random(seed)
    rng.shuffle(conversation_ids)

    n_total = len(conversation_ids)
    n_train = max(1, int(n_total * train_fraction))
    n_val = int(n_total * val_fraction)
    if n_total - n_train - n_val < 1 and n_total > 1:
        n_train = max(1, n_train - 1)

    train_ids = set(conversation_ids[:n_train])
    val_ids = set(conversation_ids[n_train : n_train + n_val])
    test_ids = set(conversation_ids[n_train + n_val :])

    train = [example for example in examples if example.conversation_id in train_ids]
    val = [example for example in examples if example.conversation_id in val_ids]
    test = [example for example in examples if example.conversation_id in test_ids]
    return train, val, test

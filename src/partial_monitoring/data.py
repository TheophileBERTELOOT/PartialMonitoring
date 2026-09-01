from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TEXT_COLUMN_CANDIDATES = ("text", "body", "message", "comment", "content",'body')
LABEL_COLUMN_CANDIDATES = ("label", "moderated", "removed", "is_removed", "mod_removed")
POSITIVE_LABELS = {"1", "true", "t", "yes", "y", "removed", "moderated", "delete", "deleted"}
NEGATIVE_LABELS = {"0", "false", "f", "no", "n", "kept", "keep", "approved", "ok"}


@dataclass(frozen=True)
class MessageExample:
    conversation_id: str
    message_index: int
    text: str
    label: int
    context: tuple[str, ...]


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
) -> list[MessageExample]:
    examples, _ = load_conversations_with_stats(
        data_dir,
        text_column=text_column,
        label_column=label_column,
        context_window=context_window,
        recursive=recursive,
    )
    return examples


def load_conversations_with_stats(
    data_dir: str | Path,
    text_column: str | None = None,
    label_column: str | None = None,
    context_window: int = 6,
    recursive: bool = True,
) -> tuple[list[MessageExample], LoadStats]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data directory does not exist: {root}")

    examples: list[MessageExample] = []
    search_root = root.rglob("*") if recursive else root.glob("*")
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
                context = tuple(previous[-context_window:])
                examples.append(
                    MessageExample(
                        conversation_id=csv_path.relative_to(root).with_suffix("").as_posix(),
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
    if not val:
        val = test[: max(1, len(test) // 2)]
    if not test:
        test = val
    return train, val, test

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from partial_monitoring.data import LABEL_COLUMN_CANDIDATES, TEXT_COLUMN_CANDIDATES, find_column, parse_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit conversation CSV files without training a model.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--max-error-examples", type=int, default=20)
    return parser.parse_args()


def add_example(items: list[str], value: str, limit: int) -> None:
    if len(items) < limit:
        items.append(value)


def main() -> None:
    args = parse_args()
    root = Path(args.data_dir)
    paths = root.rglob("*") if not args.no_recursive else root.glob("*")

    total_files = 0
    suffix_counts: Counter[str] = Counter()
    csv_files: list[Path] = []
    for path in paths:
        if not path.is_file():
            continue
        total_files += 1
        suffix = path.suffix.lower() or "<no_suffix>"
        suffix_counts[suffix] += 1
        if suffix == ".csv":
            csv_files.append(path)

    stats = {
        "data_dir": str(root),
        "recursive": not args.no_recursive,
        "total_files": total_files,
        "suffix_counts": dict(suffix_counts.most_common(25)),
        "csv_files": len(csv_files),
        "empty_or_missing_header_files": 0,
        "usable_header_files": 0,
        "files_with_rows": 0,
        "files_with_loaded_messages": 0,
        "files_without_loaded_messages": 0,
        "rows_seen": 0,
        "loaded_messages": 0,
        "empty_text_rows": 0,
        "bad_label_rows": 0,
        "missing_column_files": 0,
        "header_counts": {},
        "examples": {
            "empty_or_missing_header": [],
            "missing_columns": [],
            "bad_labels": [],
            "no_loaded_messages": [],
        },
    }
    header_counts: Counter[str] = Counter()

    for csv_path in sorted(csv_files):
        try:
            with csv_path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None or reader.fieldnames == [""]:
                    stats["empty_or_missing_header_files"] += 1
                    add_example(stats["examples"]["empty_or_missing_header"], str(csv_path), args.max_error_examples)
                    continue

                header_key = ",".join(reader.fieldnames)
                header_counts[header_key] += 1
                try:
                    text_col = find_column(reader.fieldnames, args.text_column, TEXT_COLUMN_CANDIDATES)
                    label_col = find_column(reader.fieldnames, args.label_column, LABEL_COLUMN_CANDIDATES)
                except ValueError:
                    stats["missing_column_files"] += 1
                    add_example(
                        stats["examples"]["missing_columns"],
                        f"{csv_path} header={reader.fieldnames}",
                        args.max_error_examples,
                    )
                    continue

                stats["usable_header_files"] += 1
                file_rows = 0
                file_messages = 0
                for row_index, row in enumerate(reader):
                    file_rows += 1
                    stats["rows_seen"] += 1
                    text = str(row.get(text_col, "") or "").strip()
                    if not text:
                        stats["empty_text_rows"] += 1
                        continue
                    try:
                        parse_label(row.get(label_col))
                    except ValueError:
                        stats["bad_label_rows"] += 1
                        add_example(
                            stats["examples"]["bad_labels"],
                            f"{csv_path}:{row_index} value={row.get(label_col)!r}",
                            args.max_error_examples,
                        )
                        continue
                    file_messages += 1
                    stats["loaded_messages"] += 1

                if file_rows:
                    stats["files_with_rows"] += 1
                if file_messages:
                    stats["files_with_loaded_messages"] += 1
                else:
                    stats["files_without_loaded_messages"] += 1
                    add_example(stats["examples"]["no_loaded_messages"], str(csv_path), args.max_error_examples)
        except UnicodeDecodeError as exc:
            stats["missing_column_files"] += 1
            add_example(stats["examples"]["missing_columns"], f"{csv_path} decode_error={exc}", args.max_error_examples)

    stats["header_counts"] = dict(header_counts.most_common(20))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build CSV and Markdown comparison tables from evaluation metrics."""

import argparse
import csv
import json
from pathlib import Path


COLUMNS = (
    "method",
    "pretrained",
    "frozen_backbone",
    "trainable_params",
    "epochs",
    "input_frames",
    "sampling_rate",
    "top1",
    "top5",
    "macro_acc",
    "best_checkpoint_path",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("outputs/results"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    paths = sorted(args.results_dir.glob("*/metrics.json"))
    if not paths:
        raise SystemExit(f"No metrics.json files found under {args.results_dir}")
    rows = []
    for path in paths:
        metrics = json.loads(path.read_text(encoding="utf-8"))
        rows.append({column: metrics.get(column, "") for column in COLUMNS})
    rows.sort(key=lambda row: str(row["method"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with (args.output_dir / "comparison_table.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    header = "| " + " | ".join(COLUMNS) + " |"
    separator = "| " + " | ".join("---" for _ in COLUMNS) + " |"
    body = [
        "| " + " | ".join(str(row[column]) for column in COLUMNS) + " |"
        for row in rows
    ]
    (args.output_dir / "comparison_table.md").write_text(
        "\n".join((header, separator, *body)) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(rows)} rows to {args.output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Convert Diving48 V2 JSON annotations to VideoMAEv2 list files."""

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    for key in ("data", "annotations", "videos"):
        if isinstance(payload.get(key), list):
            return payload[key]
    raise ValueError(f"{path} does not contain a recognized annotation list")


def get_value(record: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in record:
            return record[name]
    raise KeyError(f"Missing any of {names} in record: {record}")


def normalize_record(record: dict[str, Any]) -> tuple[str, int]:
    name = str(
        get_value(record, ("vid_name", "video", "video_name", "filename", "id"))
    )
    label = int(get_value(record, ("label", "class", "class_id", "action")))
    if not name.lower().endswith(".mp4"):
        name += ".mp4"
    if not 0 <= label < 48:
        raise ValueError(f"Label outside [0, 47]: {label} for {name}")
    return name, label


def stratified_split(
    records: list[tuple[str, int]], fraction: float, seed: int
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    by_class: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for record in records:
        by_class[record[1]].append(record)
    rng = random.Random(seed)
    train: list[tuple[str, int]] = []
    val: list[tuple[str, int]] = []
    for label in sorted(by_class):
        group = sorted(by_class[label])
        rng.shuffle(group)
        val_size = max(1, round(len(group) * fraction)) if fraction > 0 else 0
        val.extend(group[:val_size])
        train.extend(group[val_size:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def build_video_index(video_dir: Path) -> dict[str, Path | None]:
    index: dict[str, Path | None] = {}
    for video in video_dir.rglob("*.mp4"):
        index[video.name] = video if video.name not in index else None
    return index


def resolve_video(
    name: str, video_dir: Path, video_index: dict[str, Path | None]
) -> Path | None:
    direct = video_dir / name
    if direct.is_file():
        return direct
    return video_index.get(Path(name).name)


def write_list(
    path: Path,
    records: list[tuple[str, int]],
    video_dir: Path,
    video_index: dict[str, Path | None],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=" ", lineterminator="\n")
        for name, label in records:
            video = resolve_video(name, video_dir, video_index)
            relative = video.relative_to(video_dir).as_posix() if video else name
            writer.writerow((relative, label))


def summarize(records: list[tuple[str, int]]) -> dict[str, Any]:
    counts = Counter(label for _, label in records)
    return {
        "samples": len(records),
        "unique_classes": len(counts),
        "class_distribution": {str(i): counts.get(i, 0) for i in range(48)},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/diving48"))
    parser.add_argument("--output-report", type=Path, default=Path("outputs/data_report.json"))
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strict", action="store_true", help="Fail if any referenced video is missing."
    )
    args = parser.parse_args()
    if not 0 <= args.val_fraction < 1:
        parser.error("--val-fraction must be in [0, 1)")

    annotation_dir = args.data_dir / "annotations"
    video_dir = args.data_dir / "videos"
    train_json = annotation_dir / "Diving48_V2_train.json"
    test_json = annotation_dir / "Diving48_V2_test.json"
    for required in (train_json, test_json):
        if not required.is_file():
            print(f"Missing annotation: {required}", file=sys.stderr)
            print("Run scripts/download_diving48.py for placement instructions.", file=sys.stderr)
            return 2

    original_train = [normalize_record(item) for item in load_records(train_json)]
    test = [normalize_record(item) for item in load_records(test_json)]
    train, val = stratified_split(original_train, args.val_fraction, args.seed)
    if not val:
        print("Validation split is empty; use a positive --val-fraction.", file=sys.stderr)
        return 2

    all_records = train + val + test
    video_index = build_video_index(video_dir)
    missing = sorted(
        name
        for name, _ in all_records
        if resolve_video(name, video_dir, video_index) is None
    )
    found_count = len(video_index)
    write_list(args.data_dir / "train.csv", train, video_dir, video_index)
    write_list(args.data_dir / "val.csv", val, video_dir, video_index)
    write_list(args.data_dir / "test.csv", test, video_dir, video_index)

    report = {
        "seed": args.seed,
        "validation_fraction": args.val_fraction,
        "videos_found": found_count,
        "referenced_videos": len(all_records),
        "missing_video_count": len(missing),
        "missing_videos": missing,
        "splits": {
            "train": summarize(train),
            "val": summarize(val),
            "test": summarize(test),
        },
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))
    if any(report["splits"][split]["unique_classes"] != 48 for split in report["splits"]):
        print("Warning: at least one split does not contain all 48 classes.", file=sys.stderr)
    if missing:
        print(f"Warning: {len(missing)} referenced videos are missing.", file=sys.stderr)
        return 1 if args.strict else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())


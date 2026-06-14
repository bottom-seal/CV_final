#!/usr/bin/env python3
"""Download supplied Diving48 archives, or explain the required manual layout."""

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path


ANNOTATION_FILES = (
    "Diving48_V2_train.json",
    "Diving48_V2_test.json",
)


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output)
    partial.replace(destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/diving48"))
    parser.add_argument(
        "--annotation-base-url",
        help="Base URL containing the two Diving48 V2 JSON annotation files.",
    )
    parser.add_argument(
        "--video-archive-url",
        help="Optional authorized URL for a video archive. The script does not bypass access controls.",
    )
    parser.add_argument(
        "--video-archive-name", default="diving48_videos.tar.gz"
    )
    args = parser.parse_args()

    annotations = args.data_dir / "annotations"
    videos = args.data_dir / "videos"
    annotations.mkdir(parents=True, exist_ok=True)
    videos.mkdir(parents=True, exist_ok=True)

    if args.annotation_base_url:
        for name in ANNOTATION_FILES:
            target = annotations / name
            if target.exists():
                print(f"Already present: {target}")
                continue
            download(f"{args.annotation_base_url.rstrip('/')}/{name}", target)

    if args.video_archive_url:
        archive = args.data_dir / args.video_archive_name
        if not archive.exists():
            download(args.video_archive_url, archive)
        print(f"Downloaded video archive to {archive}")
        print("Extract it so MP4 files are directly under data/diving48/videos/.")

    missing_annotations = [
        str(annotations / name)
        for name in ANNOTATION_FILES
        if not (annotations / name).exists()
    ]
    video_count = sum(1 for _ in videos.rglob("*.mp4"))
    if missing_annotations or video_count == 0:
        print("\nDiving48 is not distributed from a stable anonymous URL by this project.")
        print("Obtain Diving48 V2 from its dataset provider, accepting its terms, then place:")
        print("  data/diving48/annotations/Diving48_V2_train.json")
        print("  data/diving48/annotations/Diving48_V2_test.json")
        print("  data/diving48/videos/*.mp4")
        print("You may also pass authorized URLs through --annotation-base-url and")
        print("--video-archive-url. No private token is stored.")
        if not args.annotation_base_url and not args.video_archive_url:
            return 0
        return 2

    print(f"Found both annotation files and {video_count} videos.")
    return 0


if __name__ == "__main__":
    sys.exit(main())


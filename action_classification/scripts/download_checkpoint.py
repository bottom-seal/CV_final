#!/usr/bin/env python3
"""Download official VideoMAE V1/V2 checkpoints."""

import argparse
import hashlib
import shutil
import urllib.request
from pathlib import Path


CHECKPOINTS = {
    "videomaev2-base": {
        "kind": "url",
        "source": (
            "https://huggingface.co/OpenGVLab/VideoMAE2/resolve/main/distill/"
            "vit_b_k710_dl_from_giant.pth"
        ),
        "filename": "videomaev2_base.pth",
    },
    "videomaev2-small": {
        "kind": "url",
        "source": (
            "https://huggingface.co/OpenGVLab/VideoMAE2/resolve/main/distill/"
            "vit_s_k710_dl_from_giant.pth"
        ),
        "filename": "videomaev2_small.pth",
    },
    "videomae-v1-base-k400-800e": {
        "kind": "gdrive",
        "source": "1JfrhN144Hdg7we213H1WxwR3lGYOlmIn",
        "filename": "videomae_v1_base_k400_800e.pth",
    },
    "videomae-v1-base-ssv2-800e": {
        "kind": "gdrive",
        "source": "181hLvyrrPW2IOGA46fkxdJk0tNLIgdB2",
        "filename": "videomae_v1_base_ssv2_800e.pth",
    },
    "videomae-v1-base-ssv2-2400e": {
        "kind": "gdrive",
        "source": "1I18dY_7rSalGL8fPWV82c0-foRUDzJJk",
        "filename": "videomae_v1_base_ssv2_2400e.pth",
    },
}
ALIASES = {
    "base": "videomaev2-base",
    "small": "videomaev2-small",
    "v1-base": "videomae-v1-base-k400-800e",
    "v1-ssv2": "videomae-v1-base-ssv2-2400e",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    choices = sorted(set(CHECKPOINTS) | set(ALIASES))
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=choices, default="videomaev2-base")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--url", help="Override with another authorized URL.")
    args = parser.parse_args()

    model_name = ALIASES.get(args.model, args.model)
    spec = CHECKPOINTS[model_name]
    destination = args.output_dir / spec["filename"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"Already present: {destination}")
        return

    partial = destination.with_suffix(destination.suffix + ".part")
    if args.url or spec["kind"] == "url":
        url = args.url or spec["source"]
        print(f"Downloading official checkpoint from {url}")
        with urllib.request.urlopen(url) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
    else:
        try:
            import gdown
        except ImportError as exc:
            raise SystemExit("Install dependencies first: pip install gdown") from exc
        print(f"Downloading official Google Drive checkpoint ID {spec['source']}")
        result = gdown.download(id=spec["source"], output=str(partial), quiet=False)
        if result is None:
            raise SystemExit("Google Drive download failed.")
    partial.replace(destination)

    digest = sha256(destination)
    destination.with_suffix(destination.suffix + ".sha256").write_text(
        f"{digest}  {destination.name}\n", encoding="utf-8"
    )
    print(f"Saved {destination}")
    print(f"SHA256 {digest}")


if __name__ == "__main__":
    main()

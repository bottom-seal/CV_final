#!/usr/bin/env python3
"""Run one experiment from a readable YAML configuration."""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def require(mapping: dict[str, Any], key: str, section: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required YAML field: {section}.{key}")
    return mapping[key]


def resolve_path(value: str | None) -> str:
    if not value:
        return ""
    path = Path(os.path.expandvars(os.path.expanduser(value)))
    if not path.is_absolute():
        path = ROOT / path
    return str(path.resolve())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args, extra_args = parser.parse_known_args()

    config_path = args.config.resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("The YAML root must be a mapping.")

    experiment = config.get("experiment", {})
    paths = config.get("paths", {})
    model = config.get("model", {})
    training = config.get("training", {})
    evaluation = config.get("evaluation", {})
    logging = config.get("logging", {})
    system = config.get("system", {})

    run_name = require(experiment, "name", "experiment")
    freeze_mode = require(model, "freeze_mode", "model")
    pretrained = bool(model.get("pretrained", True))
    values = {
        "RUN_NAME": run_name,
        "DATA_PATH": resolve_path(require(paths, "data", "paths")),
        "OUTPUT_DIR": resolve_path(require(paths, "output", "paths")),
        "MODEL_PATH": resolve_path(paths.get("checkpoint")) if pretrained else "",
        "MODEL": model.get("architecture", "vit_base_patch16_224"),
        "FREEZE_MODE": freeze_mode,
        "UNFREEZE_LAST_BLOCKS": model.get("unfreeze_last_blocks", 4),
        "GPUS": system.get("gpus", 0),
        "NUM_WORKERS": system.get("num_workers", 8),
        "BATCH_SIZE": training.get("batch_size", "auto"),
        "UPDATE_FREQ": training.get("gradient_accumulation_steps", 1),
        "EPOCHS": training.get("epochs", 35),
        "LR": training.get("learning_rate", 1e-4),
        "MIN_LR": training.get("min_learning_rate", 1e-6),
        "WARMUP_LR": training.get("warmup_learning_rate", 1e-8),
        "WARMUP_EPOCHS": training.get("warmup_epochs", 5),
        "WEIGHT_DECAY": training.get("weight_decay", 0.05),
        "LAYER_DECAY": training.get("layer_decay", 0.8),
        "DROP_PATH": training.get("drop_path", 0.2),
        "CLIP_GRAD": training.get("clip_grad", 5.0),
        "NUM_SAMPLE": training.get("repeated_samples", 1),
        "NUM_FRAMES": training.get("num_frames", 16),
        "SAMPLING_RATE": training.get("sampling_rate", 4),
        "SAVE_CKPT_FREQ": training.get("save_checkpoint_frequency", 1),
        "SEED": training.get("seed", 42),
        "TEST_NUM_SEGMENT": evaluation.get("num_segments", 5),
        "TEST_NUM_CROP": evaluation.get("num_crops", 3),
        "WANDB_PROJECT": logging.get("wandb_project", "videomaev2-diving48"),
        "WANDB_RUN_NAME": logging.get("wandb_run_name", run_name),
        "WANDB_MODE": logging.get("wandb_mode", "online"),
    }
    env = os.environ.copy()
    env.update({name: str(value) for name, value in values.items()})

    command = [str(ROOT / "scripts/run_training.sh")]
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    command.extend(extra_args)

    print(f"Config: {config_path}")
    for name in sorted(values):
        print(f"{name}={env[name]}")
    print("Command:", " ".join(command))
    if args.dry_run:
        return 0
    return subprocess.call(command, cwd=ROOT, env=env)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Evaluate a VideoMAEv2 checkpoint with per-video view aggregation."""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from timm.models import create_model
from torch.utils.data import DataLoader, SequentialSampler


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_REPO = ROOT / "VideoMAEv2"
sys.path.insert(0, str(OFFICIAL_REPO))
import models  # noqa: E402,F401
from dataset.datasets import VideoClsDataset  # noqa: E402


def load_state(model: torch.nn.Module, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint.get("module", checkpoint))
    cleaned = {}
    for original_key, value in state.items():
        key = original_key
        for prefix in ("_orig_mod.", "module.", "backbone.", "encoder."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        cleaned[key] = value
    expected = model.state_dict()
    for key in ("head.weight", "head.bias"):
        if key in cleaned and cleaned[key].shape != expected[key].shape:
            del cleaned[key]
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    allowed_missing = {"head.weight", "head.bias"}
    bad_missing = [key for key in missing if key not in allowed_missing]
    if bad_missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch. Missing={bad_missing}, unexpected={unexpected}"
        )


def load_class_names(data_path: Path) -> list[str]:
    vocab_path = data_path / "annotations" / "Diving48_vocab.json"
    if not vocab_path.exists():
        return [str(index) for index in range(48)]
    payload = json.loads(vocab_path.read_text(encoding="utf-8"))
    if isinstance(payload, list) and len(payload) == 48:
        return [str(item) for item in payload]
    return [str(index) for index in range(48)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=ROOT / "data/diving48")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", default="vit_base_patch16_224")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--sampling-rate", type=int, default=4)
    parser.add_argument("--test-num-segment", type=int, default=5)
    parser.add_argument("--test-num-crop", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--pretrained", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--freeze-mode", choices=("full", "linear", "partial"), default="full"
    )
    parser.add_argument("--unfreeze-last-blocks", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--wandb-project", default="")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    args = parser.parse_args()

    list_path = args.data_path / f"{args.split}.csv"
    if not list_path.exists():
        raise FileNotFoundError(f"Missing split list: {list_path}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")
    output_dir = args.output_dir or ROOT / "outputs/results" / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_args = SimpleNamespace(reprob=0.0, data_set="Diving48")
    dataset = VideoClsDataset(
        anno_path=str(list_path),
        data_root=str(args.data_path / "videos"),
        mode="test",
        clip_len=args.num_frames,
        frame_sample_rate=args.sampling_rate,
        num_segment=1,
        test_num_segment=args.test_num_segment,
        test_num_crop=args.test_num_crop,
        num_crop=args.test_num_crop,
        keep_aspect_ratio=True,
        crop_size=224,
        short_side_size=224,
        new_height=256,
        new_width=320,
        sparse_sample=False,
        args=dataset_args,
    )
    loader = DataLoader(
        dataset,
        sampler=SequentialSampler(dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    model = create_model(
        args.model,
        img_size=224,
        pretrained=False,
        num_classes=48,
        all_frames=args.num_frames,
        tubelet_size=2,
        drop_rate=0.0,
        drop_path_rate=0.0,
        attn_drop_rate=0.0,
        head_drop_rate=0.0,
        drop_block_rate=None,
        use_mean_pooling=True,
        init_scale=0.001,
        with_cp=False,
    )
    load_state(model, args.checkpoint)
    if args.freeze_mode == "linear":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.head.parameters():
            parameter.requires_grad = True
    elif args.freeze_mode == "partial":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for block in model.blocks[-args.unfreeze_last_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        for module_name in ("fc_norm", "norm", "head_dropout", "head"):
            module = getattr(model, module_name, None)
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad = True
    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    device = torch.device(args.device)
    model.to(device).eval()

    probabilities: dict[str, list[np.ndarray]] = defaultdict(list)
    labels: dict[str, int] = {}
    start = time.perf_counter()
    with torch.inference_mode():
        for images, target, video_ids, _, _ in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type, enabled=device.type == "cuda"
            ):
                output = model(images)
            probs = output.softmax(dim=-1).float().cpu().numpy()
            for video_id, label, prob in zip(video_ids, target.tolist(), probs):
                probabilities[video_id].append(prob)
                labels[video_id] = int(label)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    video_ids = sorted(probabilities)
    y_true = np.array([labels[video_id] for video_id in video_ids])
    scores = np.stack(
        [np.mean(probabilities[video_id], axis=0) for video_id in video_ids]
    )
    y_pred = scores.argmax(axis=1)
    top1 = float((y_pred == y_true).mean() * 100)
    top5 = float(
        np.mean(
            [
                label in np.argsort(-score)[:5]
                for label, score in zip(y_true, scores)
            ]
        )
        * 100
    )
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(48))
    denominators = matrix.sum(axis=1)
    per_class = np.divide(
        matrix.diagonal(),
        denominators,
        out=np.zeros(48, dtype=float),
        where=denominators != 0,
    )
    present_classes = denominators > 0
    macro_acc = float(per_class[present_classes].mean() * 100)
    names = load_class_names(args.data_path)

    with (output_dir / "per_class_accuracy.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("class_id", "class_name", "samples", "accuracy"))
        for index in range(48):
            writer.writerow(
                (
                    index,
                    names[index],
                    int(denominators[index]),
                    per_class[index] * 100,
                )
            )

    figure, axis = plt.subplots(figsize=(14, 12))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    axis.set_title(f"Diving48 confusion matrix: {args.run_name}")
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(figure)

    metrics = {
        "method": args.run_name,
        "pretrained": args.pretrained,
        "frozen_backbone": {
            "full": "No",
            "linear": "Yes",
            "partial": "Partially",
        }[args.freeze_mode],
        "trainable_params": trainable_params,
        "epochs": args.epochs,
        "input_frames": args.num_frames,
        "sampling_rate": args.sampling_rate,
        "top1": top1,
        "top5": top5,
        "macro_acc": macro_acc,
        "num_videos": len(video_ids),
        "inference_seconds": elapsed,
        "inference_seconds_per_video": elapsed / max(len(video_ids), 1),
        "best_checkpoint_path": str(args.checkpoint.resolve()),
        "split": args.split,
        "test_num_segment": args.test_num_segment,
        "test_num_crop": args.test_num_crop,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))

    if args.wandb_project:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            name=f"{args.run_name}_{args.split}_eval",
            mode=args.wandb_mode,
            config=vars(args),
        )
        run.log(
            {
                **metrics,
                "confusion_matrix": wandb.plot.confusion_matrix(
                    y_true=y_true,
                    preds=y_pred,
                    class_names=names,
                ),
            }
        )
        run.summary.update(metrics)
        run.finish()


if __name__ == "__main__":
    main()

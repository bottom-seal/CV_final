# VideoMAE V2 Transfer Learning on Diving48

This project compares random initialization, linear probing, partial
fine-tuning, VideoMAE V1 full fine-tuning, and VideoMAE V2 full fine-tuning on
Diving48. It keeps the official
OpenGVLab/VideoMAEv2 training path and adds dataset preparation, experiment
wrappers, W&B logging, and per-video evaluation.

When cloning this project, populate the official repositories and apply the
small local compatibility patch:

```bash
git clone --recurse-submodules https://github.com/bottom-seal/CV_final.git
cd CV_final/action_classification
git -C VideoMAEv2 apply patches/videomaev2-local.patch
```

For an existing clone, run `git submodule update --init --recursive` instead.

The included official repositories are pinned at:

- VideoMAE V2: `29eab1e8a588d1b3ec0cdec7b03a86cca491b74b`
- VideoMAE V1: `14ef8d856287c94ef1f985fe30f958eb4ec2c55d`

## 1. Environment

Python 3.10 or 3.11 is recommended because the official code uses an older
`timm` API. Adjust `TORCH_INDEX_URL` if another PyTorch CUDA wheel is needed.

```bash
cd /mnt/disk2/lmz/CV_final/action_classification
PYTHON=python3.11 TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 \
  bash scripts/setup_env.sh
source .venv/bin/activate
```

DeepSpeed is not required for these single-GPU scripts.

## 2. Diving48

This project does not bypass dataset access terms or store private tokens.
Show the expected manual placement, or pass authorized URLs if you have them:

```bash
python scripts/download_diving48.py
```

Place the provider's Diving48 V2 files here:

```text
data/diving48/
├── annotations/
│   ├── Diving48_V2_train.json
│   ├── Diving48_V2_test.json
│   └── Diving48_vocab.json          # optional
└── videos/
    └── *.mp4
```

Create a deterministic, class-stratified 90/10 train/validation split. The
official V2 test annotations remain untouched as `test.csv`.

```bash
python scripts/prepare_diving48.py --drop-missing
```

This also writes `outputs/data_report.json`. In the supplied V2 annotations, label ID `30` has no samples in either official JSON while IDs still extend to `47`; the project preserves those source IDs and keeps a 48-output head.

VideoMAEv2 parses space-delimited
list files (`relative/video.mp4 label`), so the preparation script follows the
implementation rather than the comma shown in an old documentation comment.

## 3. Weights

Download the official VideoMAE V2 Base checkpoint:

```bash
python scripts/download_checkpoint.py --model videomaev2-base
```

The official VideoMAE V1 ViT-Base Kinetics-400 800-epoch pretraining
checkpoint is also supported:

```bash
python scripts/download_checkpoint.py --model videomae-v1-base-k400-800e
```

VideoMAE V1 Something-Something V2 pretraining is available at both 800 and
2400 epochs. The `v1-ssv2` alias selects the stronger 2400-epoch checkpoint:

```bash
python scripts/download_checkpoint.py --model videomae-v1-base-ssv2-800e
python scripts/download_checkpoint.py --model videomae-v1-base-ssv2-2400e
# Short alias for 2400 epochs:
python scripts/download_checkpoint.py --model v1-ssv2
```

This is the official K710-distilled VideoMAE V2 Base checkpoint in OpenGVLab's
Hugging Face repository. It is directly compatible with
`run_class_finetuning.py`, so no Transformers conversion is needed. A SHA256
file is saved beside it.

## 4. W&B

Authenticate without storing a key in this repository:

```bash
wandb login
# or
export WANDB_API_KEY=...
```

Use `WANDB_MODE=offline` for an offline run or `WANDB_MODE=disabled` to disable
uploading.

## 5. Train Manually

Each launcher exposes `DATA_PATH`, `OUTPUT_DIR`, `MODEL_PATH`, `GPUS`,
`BATCH_SIZE`, `EPOCHS`, `LR`, `WANDB_PROJECT`, and `WANDB_RUN_NAME`.
`BATCH_SIZE=auto` chooses 4 below 22 GB VRAM and 8 otherwise. The local 16 GB
RTX 4080 SUPER therefore defaults to 4.

```bash
bash scripts/train_linear_probe.sh
bash scripts/train_full_finetune.sh
bash scripts/train_partial_finetune.sh
bash scripts/train_random_init.sh
```

Example overrides:

```bash
BATCH_SIZE=2 EPOCHS=3 WANDB_MODE=offline \
  bash scripts/train_full_finetune.sh

UNFREEZE_LAST_BLOCKS=2 LR=1e-4 \
  bash scripts/train_partial_finetune.sh
```

The launchers use literal learning rates through `--no_lr_scale`. Checkpoints
include `checkpoint-latest.pth`, `checkpoint-best.pth`, and one numbered
checkpoint per epoch. Auto-resume uses the latest numbered checkpoint.

Random initialization is optional because training a video ViT-Base from
scratch is substantially more expensive and may need more epochs.

## 6. YAML Configs

Experiment parameters can be edited in `configs/`:

```text
configs/linear_probe.yaml
configs/full_finetune.yaml
configs/partial_finetune.yaml
configs/random_init.yaml
configs/videomae_v1_full_finetune.yaml
configs/videomae_v1_ssv2_full_finetune.yaml
```

Preview all resolved paths and parameters without starting training:

```bash
python scripts/run_experiment.py configs/full_finetune.yaml --dry-run
```

Start a run manually from YAML:

```bash
python scripts/run_experiment.py configs/linear_probe.yaml
python scripts/run_experiment.py configs/full_finetune.yaml
python scripts/run_experiment.py configs/partial_finetune.yaml
python scripts/run_experiment.py configs/random_init.yaml
python scripts/run_experiment.py configs/videomae_v1_full_finetune.yaml
python scripts/run_experiment.py configs/videomae_v1_ssv2_full_finetune.yaml
```

Extra official CLI arguments can be appended after `--`, for example:

```bash
python scripts/run_experiment.py configs/full_finetune.yaml -- --no_auto_resume
```

The YAML files control paths, architecture, freeze mode, epochs, batch size,
learning rates, warmup, weight and layer decay, drop path, gradient clipping,
frame sampling, checkpoint frequency, evaluation views, W&B, GPU, and workers.


## Baselines

The project now has three baseline types:

- `random_init.yaml`: same ViT-Base architecture without pretrained weights.
- `linear_probe.yaml`: frozen VideoMAE V2 encoder with only the head trained.
- `videomae_v1_full_finetune.yaml`: official VideoMAE V1 Base K400-800e
  pretraining, fully fine-tuned with the same downstream settings as V2.
- `videomae_v1_ssv2_full_finetune.yaml`: official VideoMAE V1 Base SSV2-2400e
  pretraining with the same downstream settings, isolating pretraining domain.

The V1 comparison intentionally uses the shared VideoMAE V2 downstream
training harness. The V1 and V2 Base encoder parameter layouts are compatible,
and the loader strips the V1 `encoder.` prefix while ignoring decoder-only
pretraining weights. This keeps the data, augmentation, optimizer, schedule,
and evaluation identical, so the comparison isolates pretrained initialization
more cleanly than using two different training frameworks.

Start the V1 baseline with:

```bash
python scripts/run_experiment.py configs/videomae_v1_full_finetune.yaml
```

## 7. Evaluation

The evaluator averages temporal and spatial views per video and saves top-1,
top-5, macro accuracy, per-class accuracy, a confusion matrix, video count,
and inference timing.

Full fine-tune:

```bash
bash scripts/eval.sh
```

Linear probe:

```bash
MODEL_PATH=outputs/runs/linear_probe_videomaev2_base/checkpoint-best.pth \
OUTPUT_DIR=outputs/results/linear_probe_videomaev2_base \
WANDB_RUN_NAME=linear_probe_videomaev2_base EPOCHS=20 \
bash scripts/eval.sh --freeze-mode linear
```

Quick one-view evaluation:

```bash
bash scripts/eval.sh --test-num-segment 1 --test-num-crop 1
```

Use `--split val` for model-selection checks. The default is the untouched
official `test.csv`. Results are written under
`outputs/results/<run_name>/` as `metrics.json`,
`per_class_accuracy.csv`, and `confusion_matrix.png`.

For random initialization add `--no-pretrained`. For partial fine-tuning add
`--freeze-mode partial --unfreeze-last-blocks 2` or `4`.

## 8. Comparison Table

After evaluating every run:

```bash
python scripts/compare_results.py
```

This creates `outputs/comparison_table.csv` and
`outputs/comparison_table.md` with the requested method, pretraining, freezing,
parameter count, schedule, accuracy, and checkpoint columns.

## Reproducibility Notes

- Preparation and training use seed 42.
- The validation split is stratified and recorded in `data_report.json`.
- The wrappers preserve the official model, augmentation, optimizer,
  scheduler, checkpoint loader, and validation loop.
- Use 1x1 evaluation while debugging and 5x3 for final reported numbers.
- Official training evaluates `val.csv` at the end. Run `scripts/eval.sh`
  afterward to report the untouched `test.csv`.

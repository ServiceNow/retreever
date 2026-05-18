# ReTreever

Hierarchical tree-based retrieval. The model learns a binary tree of latent
embeddings — at each internal node, a split function decides which subtree a
context belongs to. Training is contrastive: in-batch positives are the
question/context pair, negatives are the rest of the batch.

This repository is the minimal training stack for ReTreever on six datasets:

| Dataset                 | Modality | Source                                      |
|-------------------------|----------|---------------------------------------------|
| `nq`                    | Text     | `ServiceNow/dssk_training_data` (NQ filter) |
| `hotpotqa`              | Text     | HF `hotpot_qa` (distractor)                 |
| `topiocqa_all_history`  | Text     | HF `McGill-NLP/TopiOCQA`                    |
| `repliqa_0to3_4`        | Text     | HF `ServiceNow/repliqa`                     |
| `imagenet1k`            | Image    | HF `ILSVRC/imagenet-1k` (gated)             |
| `voxceleb2`             | Audio    | https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html (registration required) |

---

## Install

```bash
conda create -n retreever python=3.10 -y
conda activate retreever
pip install -r requirements.txt
pip install -e .
```

Then configure your local paths (this file is gitignored):

```bash
cp local_paths.py.example local_paths.py
$EDITOR local_paths.py    # set HF_CACHE_DIR and DATA_PATHS
```

`local_paths.DATA_PATHS` maps each dataset key (`nq`, `hotpotqa`, …) to the
directory where its prepared on-disk data lives. The data-prep scripts below
write into those directories.

---

## Prepare datasets

Each prep script writes a `DatasetDict` with `train` / `val` / `test` splits
plus a `cuid2text` lookup, in the schema that ReTreever's collators consume.
Run only the ones you need.

### HotpotQA

```bash
python -m scripts.data_prep.hotpotqa --out-dir $DATA_PATHS_hotpotqa
```

Pulls `hotpot_qa` (config `distractor`) from Hugging Face and emits 10
distractor paragraphs per row with a 0/1 `useful_contexts` indicator for the
two supporting paragraphs.

### RepliQA (0to3_4 variant)

```bash
python -m scripts.data_prep.repliqa_0to3_4 --out-dir $DATA_PATHS_repliqa_0to3_4
```

Loads `ServiceNow/repliqa`, holds out 400 random `document_id`s from
`repliqa_3` (seed=42) as val, joins `repliqa_0..3 \ val` as train, and uses
`repliqa_4` as test. Uses `long_answer` as the gold context.

### ImageNet-1K

```bash
# Accept the license at https://huggingface.co/datasets/ILSVRC/imagenet-1k first,
# then `huggingface-cli login`.
python -m scripts.data_prep.imagenet1k --out-dir $DATA_PATHS_imagenet1k
```

Writes a `torchvision.ImageFolder`-style tree
`{out}/{train,val}/class_NNNN/*.JPEG`. Full extraction is ~150 GB; pass
`--max-per-class 100` for a quick smoke test.

### VoxCeleb2

VoxCeleb2 must be obtained from Oxford VGG (registration required):

```bash
# 1. Download vox2_dev_aac.zip and vox2_test_aac.zip from the official site.
unzip vox2_dev_aac.zip  -d /raw/voxceleb2/dev/
unzip vox2_test_aac.zip -d /raw/voxceleb2/test/

# 2. Flatten {speaker}/{video}/*.m4a -> {speaker}/*.m4a in the layout
#    the ReTreever loader expects.
python -m scripts.data_prep.voxceleb2 \
    --train-src /raw/voxceleb2/dev/aac \
    --val-src   /raw/voxceleb2/test/aac \
    --out-dir   $DATA_PATHS_voxceleb2
```

By default the converter creates symlinks (fast, no extra disk). Pass `--copy`
to materialize real files.

---

## Train

The training entrypoint is `scripts/train.py`, configured by Hydra. The
top-level config is `scripts/config/train.yaml` which composes
`config/model/retreever.yaml`, `config/train/retreever.yaml`, and
`config/logging/wandb.yaml`. Override any field on the command line.

### A first run (HotpotQA, single GPU)

```bash
WANDB_MODE=disabled \
python -m scripts.train \
    dataset=hotpotqa \
    savedir=runs/hotpotqa_baseline
```

That's it — `dataset=...` picks the on-disk directory from
`local_paths.DATA_PATHS`, and everything else defaults reasonably.

### With and without stochastic depth

Stochastic-depth training samples a random tree depth at each step, which
is what the ReTreever paper does by default. Two flavors:

```bash
# WITH stochastic depth (default; samples depth ~ heavy-tailed)
python -m scripts.train dataset=hotpotqa \
    train.hierarchical=true \
    train.depth_scheduler_type=random \
    savedir=runs/hotpotqa_stochastic

# WITHOUT stochastic depth — train at full depth on every step
python -m scripts.train dataset=hotpotqa \
    train.hierarchical=false \
    savedir=runs/hotpotqa_constant_depth
```

Other depth schedules available: `linear`, `linear_weighted`, `exponential`,
`random_uniform`, `random_linear`. See
`retreever/training/depth_schedulers.py`.

### Picking an encoder for each modality

`scripts/config/model/retreever.yaml` defaults to `encoder_type: bge`, which is
text-only. Image/audio datasets need a matching encoder; override on the CLI:

```bash
# ImageNet — DINOv2 vision encoder
python -m scripts.train dataset=imagenet1k \
    model.encoder_type=dinov2-base \
    model.encoder_token_level=False \
    savedir=runs/imagenet1k_dinov2

# VoxCeleb2 — Wav2Vec2 audio encoder
python -m scripts.train dataset=voxceleb2 \
    model.encoder_type=wav2vec2-base \
    model.encoder_token_level=False \
    savedir=runs/voxceleb2_w2v2
```

Available text encoders include `bge`, `dpr`, `bert`, `distilbert`,
`contriever`, `simcse`; vision: `dinov2-*`, `resnet50`, `clip-vit-*`; audio:
`wav2vec2-*`, `hubert-*`, `wavlm-*`, `ast`, `clap`. See
`retreever/models/encoders.py` for the full list and the model-name strings
each one resolves to.

### Smoke test (5 steps + 1 eval cycle, any dataset)

```bash
WANDB_MODE=disabled python -m scripts.train \
    dataset=nq \
    train.steps=5 train.train_batch_size=4 train.test_batch_size=4 \
    model.tree_depth=4 \
    logging.log_every=2 logging.factor_val_irrelevant_ctxs=1 \
    savedir=/tmp/retreever_smoke
```

Runs in ~4 minutes on a single H100 and triggers eval (hit@k / NDCG@k /
mAP@k) four times.

### Resuming and checkpoints

Checkpoints are written under `savedir/checkpoint-{step}`. Resume with
`ckpt=<path>`, or simply re-launch with the same `savedir` and the trainer
will pick up the latest checkpoint automatically.

---

## Repository layout

```
retreever/
  models/         ReTreever, encoders, trees, split functions, indexing
  data/           Dataset loaders + collators (text / image / audio)
  training/       HF Trainer subclass + depth schedulers
  evaluation/     Retrieval metrics (hit@k, NDCG@k, mAP@k) and eval loop
  utils/          Losses, distributed gather, path resolver
scripts/
  train.py        Hydra entrypoint
  config/         Hydra config tree
  data_prep/      One script per dataset (HF -> on-disk DatasetDict / ImageFolder)
local_paths.py    Machine-specific paths (gitignored; copy from .example)
```

---

## Tips

- **Single GPU**: launch via `python -m scripts.train ...`. No DeepSpeed
  wrapper needed; HF Trainer handles single-GPU correctly.
- **Multi-GPU / multi-node**: launch via `deepspeed hydra_entrypoint.py
  --deepspeed=scripts/config/deepspeed.json ...`. The contrastive loss
  switches to a cross-process gather automatically when `world_size > 1`.
- **Disabling wandb**: set `WANDB_MODE=disabled` in the environment, or
  `debug=true` on the CLI to log to stdout instead.
- **Memory**: `train_batch_size=64` with a `bge-large` encoder and
  `tree_depth=10` fits comfortably on a single 80 GB H100.

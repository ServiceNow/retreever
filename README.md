# ReTreever

Hierarchical tree-based retrieval. The model learns a binary tree of latent
embeddings — at each internal node, a split function decides which subtree a
context belongs to. Training is contrastive: in-batch positives are the
question/context pair, negatives are the rest of the batch.

This repository is the minimal training stack for ReTreever. It ships
prep scripts and a tested training loop for five public datasets out of the
box, and a documented path for plugging in your own retrieval dataset
(see [Bring your own dataset](#bring-your-own-dataset)).

| Dataset                 | Modality | Source                                      |
|-------------------------|----------|---------------------------------------------|
| `hotpotqa`              | Text     | HF `hotpot_qa` (distractor)                 |
| `topiocqa`  | Text     | HF `McGill-NLP/TopiOCQA`                    |
| `repliqa`        | Text     | HF `ServiceNow/repliqa`                     |
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

`local_paths.DATA_PATHS` maps each dataset key (`hotpotqa`,
`topiocqa`, …) to the directory where its prepared on-disk data
lives. The data-prep scripts below write into those directories.

---

## Prepare datasets

Each prep script writes a `DatasetDict` with `train` / `val` (and sometimes
`test`) splits, in the schema that ReTreever's collators consume (see
[Bring your own dataset](#bring-your-own-dataset)). Each `--out-dir` should
match the path you registered in `local_paths.DATA_PATHS` for that dataset.
Run only the ones you need.

### HotpotQA

```bash
python -m scripts.data_prep.hotpotqa --out-dir /abs/path/to/hotpotqa
```

Pulls `hotpot_qa` (config `distractor`) from Hugging Face and emits 10
distractor paragraphs per row with a 0/1 `useful_contexts` indicator for the
two supporting paragraphs.

### TopiOCQA

```bash
python -m scripts.data_prep.topiocqa --out-dir /abs/path/to/topiocqa
```

Pulls `McGill-NLP/TopiOCQA` JSONLs directly from Hugging Face, holds out 200
random conversations from train (seed=42) as the new val split, promotes the
original valid split to test, and flattens each turn into a question that
concatenates the conversation history with `' [SEP] '`. Skips UNANSWERABLE
turns.

### RepliQA

```bash
python -m scripts.data_prep.repliqa --out-dir /abs/path/to/repliqa
```

Loads `ServiceNow/repliqa`, holds out 400 random `document_id`s from
`repliqa_3` (seed=42) as val, joins `repliqa_0..3 \ val` as train, and uses
`repliqa_4` as test. Uses `long_answer` as the gold context.

### ImageNet-1K

```bash
# Accept the license at https://huggingface.co/datasets/ILSVRC/imagenet-1k first,
# then `huggingface-cli login`.
python -m scripts.data_prep.imagenet1k --out-dir /abs/path/to/imagenet1k
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
    --out-dir   /abs/path/to/voxceleb2
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
    dataset=hotpotqa \
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

## Bring your own dataset

Training on a custom retrieval corpus takes three small steps:

### 1. Convert your raw data to the ReTreever schema

ReTreever consumes Hugging Face `DatasetDict`s saved with `save_to_disk`.
Each row must contain at least these fields:

| Field          | Type      | Required? | Notes                                              |
|----------------|-----------|-----------|----------------------------------------------------|
| `question`     | `str`     | yes       | The query string.                                  |
| `context`      | `str`     | yes\*     | The gold context for the query.                    |
| `contexts_list`| `list[str]` | yes\* | Used by multi-context datasets (e.g., HotpotQA's 10 distractor paragraphs). |
| `useful_contexts`| `list[int]` | with `contexts_list` | 0/1 indicator per entry in `contexts_list`. |
| `context_uid`  | `int`     | yes       | Unique integer ID for the gold context. Used as the retrieval target during eval. |

\* The collator reads either `context` (single string) or
`contexts_list` + `useful_contexts` (multi-context). Pick whichever fits your
data. To start, the simplest format is just `question` + `context` + `context_uid`.

`context_uid` is just a stable integer assigned once per distinct context
string in your dataset; the eval loop uses it as the ground-truth label
for retrieval.

A minimum prep script looks like this:

```python
from datasets import Dataset, DatasetDict

def assign_uids(rows):
    text_to_uid = {}
    for row in rows:
        if row["context"] not in text_to_uid:
            text_to_uid[row["context"]] = len(text_to_uid)
        row["context_uid"] = text_to_uid[row["context"]]
    return rows

train_rows = [{"question": ..., "context": ...} for ...]  # your data
val_rows   = [{"question": ..., "context": ...} for ...]

# Share the uid map across splits so that the same context gets the same uid.
all_rows = train_rows + val_rows
assign_uids(all_rows)

DatasetDict({
    "train": Dataset.from_list(train_rows),
    "val":   Dataset.from_list(val_rows),
}).save_to_disk("/abs/path/to/my_dataset")
```

See `scripts/data_prep/hotpotqa.py`, `scripts/data_prep/topiocqa.py`,
and `scripts/data_prep/repliqa.py` for fuller examples that follow
this pattern.

### 2. Register the dataset path

Open `local_paths.py` (created from `local_paths.py.example`) and add an
entry to `DATA_PATHS`:

```python
DATA_PATHS = {
    # ...existing entries...
    "my_dataset": "/abs/path/to/my_dataset",
}
```

### 3. Train

```bash
WANDB_MODE=disabled python -m scripts.train \
    dataset=my_dataset \
    savedir=runs/my_dataset_baseline
```

The defaults in `scripts/config/model/retreever.yaml` and
`scripts/config/train/retreever.yaml` are reasonable starting points for
text retrieval. For non-text datasets, override `model.encoder_type` (see
[Picking an encoder for each modality](#picking-an-encoder-for-each-modality))
and set `model.encoder_token_level=False`.

A few defaults you may want to tune for your corpus size:

- `model.tree_depth=10` → `log2(num_unique_contexts) + 2` is a sensible target.
- `train.train_batch_size=64` → reduce on smaller GPUs; HF Trainer also
  exposes `train.skip_steps` for gradient accumulation.
- `train.steps=200_000` → set this proportional to dataset size and
  `train_batch_size`.

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
- **Disabling wandb**: set `WANDB_MODE=disabled` in the environment, or pass
  `debug=true` on the CLI (the latter sets the W&B mode to `disabled` so
  metrics don't try to upload).
- **Memory**: `train_batch_size=64` with a `bge-large` encoder and
  `tree_depth=10` fits comfortably on a single 80 GB H100.

# ReTreever Training and Evaluation Scripts

This directory contains scripts for training and evaluating ReTreever models.

## Training

Train a ReTreever model using Hydra configuration:

```bash
# Train on Natural Questions with default settings
python scripts/train.py dataset=nq data_dir=/path/to/nq

# Train on ImageNet1k  
python scripts/train.py dataset=imagenet1k data_dir=/path/to/imagenet1k \
    model.encoder_type=dinov2-large

# Train on VoxCeleb2 audio retrieval
python scripts/train.py dataset=voxceleb2 data_dir=/path/to/voxceleb2 \
    model.encoder_type=ast

# Train with custom config
python scripts/train.py --config-name examples/imagenet1k.yaml data_dir=/path/to/data

# Train with MRL
python scripts/train.py model_type=mrl dataset=nq data_dir=/path/to/nq
```

### Configuration

Configuration files are located in `scripts/config/`:
- `train.yaml` - Main configuration file
- `model/retreever.yaml` - ReTreever model settings
- `model/mrl.yaml` - MRL model settings
- `train/retreever.yaml` - Training hyperparameters
- `logging/wandb.yaml` - W&B logging settings
- `examples/` - Example configs for different datasets

### Key Training Arguments

Override any config parameter from command line:

```bash
# Change learning rate
python scripts/train.py train.learning_rate=0.001

# Adjust batch size
python scripts/train.py train.train_batch_size=128

# Change tree depth
python scripts/train.py model.tree_depth=8

# Use different encoder
python scripts/train.py model.encoder_type=bge

# Disable hierarchical training
python scripts/train.py train.hierarchical=false
```

### Supported Datasets

**Text Retrieval:**
- `nq` - Natural Questions
- `hotpotqa` - HotpotQA multi-hop reasoning
- `repliqa` - RepliQA conversational QA
- `topiocqa` - TopiocQA conversational info-seeking

**Image Retrieval:**
- `imagenet1k` - ImageNet-1K classification

**Audio Retrieval:**
- `voxceleb2` - VoxCeleb2 speaker identification

**Text-Image Retrieval:**
- `coco` - COCO captions
- `flickr30k` - Flickr30k captions

### Supported Encoders

**Text:** `distilbert`, `bge`  
**Image:** `dinov2-large`, `dinov2-base`, `resnet50`, `resnet101`, `clip-vit-large-patch14`  
**Audio:** `ast`, `wav2vec2`, `hubert`  
**Multimodal:** `flava`

### Loss Functions

- `contrastive` - Standard contrastive loss
- `multi_label_contrastive` - For classification tasks (ImageNet, VoxCeleb)
- `mrl` - Matryoshka Representation Learning

## Evaluation

Evaluate a trained model:

```bash
# Evaluate ReTreever on test set
python scripts/evaluate.py \
    --model_ckpt checkpoints/nq/pytorch_model.bin \
    --model_cfg checkpoints/nq/config.yaml \
    --model_type retreever \
    --dataset nq \
    --data_dir /path/to/nq \
    --split test \
    --batch_size 64

# Evaluate with custom k values
python scripts/evaluate.py \
    --model_ckpt checkpoints/imagenet1k/pytorch_model.bin \
    --model_cfg checkpoints/imagenet1k/config.yaml \
    --dataset imagenet1k \
    --data_dir /path/to/imagenet1k \
    --k_values 1 5 10 20 50 \
    --save_path eval_results/

# Evaluate MRL model
python scripts/evaluate.py \
    --model_ckpt checkpoints/mrl_nq/pytorch_model.bin \
    --model_cfg checkpoints/mrl_nq/config.yaml \
    --model_type mrl \
    --dataset nq \
    --data_dir /path/to/nq
```

### Evaluation Metrics

The evaluation script computes:
- **Hit@K** - Fraction of queries with ≥1 relevant item in top-K
- **NDCG@K** - Normalized Discounted Cumulative Gain at K
- **Recall@K** - Proportion of relevant items retrieved in top-K
- **mAP@K** - Mean Average Precision at K

Default K values: [1, 3, 10, 50, 100]

### Evaluation Arguments

- `--model_ckpt` - Path to model checkpoint (.bin)
- `--model_cfg` - Path to model config (.yaml)
- `--model_type` - Model type: `retreever` or `mrl`
- `--dataset` - Dataset name
- `--data_dir` - Path to dataset
- `--split` - Dataset split (default: `test`)
- `--batch_size` - Evaluation batch size (default: 64)
- `--k_values` - K values for metrics (default: 1 3 10 50 100)
- `--num_distractors` - Number of distractor contexts (default: 0)
- `--subset_size` - Evaluate on subset (default: full dataset)
- `--device` - Device: `cuda` or `cpu`
- `--save_path` - Path to save results

## Example Workflows

### Train ReTreever on Natural Questions

```bash
# 1. Prepare data directory structure
export DATA_DIR=/path/to/nq

# 2. Train model
python scripts/train.py \
    dataset=nq \
    data_dir=$DATA_DIR \
    model.encoder_type=distilbert \
    train.steps=50000 \
    savedir=checkpoints/nq_retreever

# 3. Evaluate
python scripts/evaluate.py \
    --model_ckpt checkpoints/nq_retreever/checkpoint-50000/pytorch_model.bin \
    --model_cfg checkpoints/nq_retreever/config.yaml \
    --dataset nq \
    --data_dir $DATA_DIR
```

### Train on ImageNet1k

```bash
python scripts/train.py \
    --config-name examples/imagenet1k.yaml \
    data_dir=/path/to/imagenet1k \
    savedir=checkpoints/imagenet1k
```

### Train MRL Model

```bash
python scripts/train.py \
    model_type=mrl \
    dataset=nq \
    data_dir=/path/to/nq \
    model=mrl \
    train=mrl
```

## Distributed Training

Use DeepSpeed for multi-GPU training:

```bash
# Create deepspeed config (deepspeed_config.json)
deepspeed scripts/train.py \
    dataset=nq \
    data_dir=/path/to/nq \
    deepspeed=deepspeed_config.json
```

## Monitoring

Training metrics are logged to Weights & Biases by default. Configure in `config/logging/wandb.yaml`:

```yaml
wandb_project: "retreever"
wandb_run: "exp_${dataset}_${model_type}"
```

Disable W&B:
```bash
python scripts/train.py logging.wandb_project=null
```

## Checkpointing

Models are checkpointed every `logging.log_every` steps (default: 1000). Resume training:

```bash
python scripts/train.py dataset=nq data_dir=/path/to/nq ckpt=/path/to/checkpoint
```

Or auto-resume from last checkpoint:
```bash
# Automatically resumes if checkpoints exist in savedir
python scripts/train.py dataset=nq data_dir=/path/to/nq savedir=checkpoints/nq
```

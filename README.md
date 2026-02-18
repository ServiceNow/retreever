# ReTreever: Hierarchical Retrieval with Coarse-To-Fine Representations

**ReTreever** is a framework for training and evaluating hierarchical retrieval models with multi-resolution representations. It supports multiple modalities (text, images, audio) and encoder fine-tuning strategies.

## Features

- 🌳 **Hierarchical Retrieval**: Tree-based retrieval with stochastic or constant depth training
- 🎓 **Depth Curriculum Learning**: Multiple schedulers (random heavy-tailed, random linear, random uniform, linear warmup, exponential warmup)
- 🎯 **Multi-Resolution Learning**: Matryoshka Representation Learning (MRL) for efficient embedding compression
- 🔧 **Encoder Fine-tuning**: Shared MLP/Linear adapters with zero-init normalization
- 🖼️ **Multi-Modal**: Text (DistilBERT, BGE), Images (DinoV2, ResNet, CLIP), Audio (AST), Text-Image (FLAVA)
- ⚡ **Efficient Training**: DeepSpeed integration, mixed precision, and distributed training
- 📊 **Comprehensive Evaluation**: NDCG@k, Hit@k, MAP@k metrics with FAISS-based indexing

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/retreever.git
cd retreever

# Install in development mode
pip install -e .

# Or install from PyPI (when available)
pip install retreever

# For GPU support with FAISS
pip install retreever[gpu]
```

## Quick Start

### Training a Model

```python
from retreever.models import ReTreever
from retreever.training import Trainer
from retreever.utils import load_config

# Load configuration
cfg = load_config("configs/imagenet/retreever_stochastic_dinov2.yaml")

# Create model
model = ReTreever(
    encoder_type="dinov2-large",
    tree_type="no_propagation_tree",
    tree_depth=10,
    **cfg.model
)

# Train
trainer = Trainer(model=model, config=config)
trainer.train()
```

## Hierarchical Training with Depth Scheduling

ReTreever supports **hierarchical training**, where the model is trained to make predictions at different tree depths. This is crucial for learning multi-resolution representations.

### Training Strategies

**Constant Depth Training** (`hierarchical: false`)
- Trains at full tree depth throughout training
- Simpler but less flexible

**Stochastic Depth Training** (`hierarchical: true`)
- Randomly samples depth at each training step
- Better multi-resolution representations

### Depth Schedulers

Configure via `depth_scheduler_type` in config:

```yaml
train:
  hierarchical: true  # Enable depth scheduling
  depth_scheduler_type: "random"  # Scheduler type
  depth_warmup_ratio: 0.1  # For non-random schedulers
```

**Available schedulers:**

1. **`random`** (default for stochastic) - RandomHeavyTailedDepthScheduler
   - Samples with quadratic weights: P(depth=d) ∝ d²
   - Strongly biases towards deeper levels
   - Best for final model performance

2. **`random_linear`** - RandomDepthScheduler
   - Samples with linear weights: P(depth=d) ∝ d
   - Moderate bias towards deeper levels

3. **`random_uniform`** - RandomUniformDepthScheduler
   - Uniform sampling across all depths
   - Equal probability for each depth

4. **`linear`** - LinearDepthScheduler
   - Gradually increases depth during warmup
   - Starts at shallow, ends at full depth
   - Good for curriculum learning

5. **`exponential`** - ExponentialDepthScheduler
   - Exponentially increases time spent at each depth
   - Spends more time at deeper levels

### Example Configurations

**Stochastic with heavy-tailed sampling (typical):**
```yaml
train:
  hierarchical: true
  depth_scheduler_type: "random"
  steps: 200000
```

**Constant depth (no scheduling):**
```yaml
train:
  hierarchical: false
  steps: 200000
```

**Linear warmup over first 10% of training:**
```yaml
train:
  hierarchical: true
  depth_scheduler_type: "linear"
  depth_warmup_ratio: 0.1
  steps: 200000
```

### Using Command Line

```bash
# Train ImageNet model with DinoV2
python scripts/train.py --config configs/imagenet/retreever_stochastic_dinov2.yaml

# Evaluate a trained model on image retrieval
python scripts/evaluate_image.py --model_ckpt path/to/checkpoint.pt \
                                   --model_cfg path/to/config.yaml \
                                   --dataset imagenet \
                                   --data_dir /path/to/imagenet1k

# Evaluate a trained model on text retrieval
python scripts/evaluate_text.py --model_ckpt path/to/checkpoint.pt \
                                  --model_cfg path/to/config.yaml \
                                  --dataset nq \
                                  --data_dir /path/to/nq

# Evaluate a trained model on audio retrieval
python scripts/evaluate_audio.py --model_ckpt path/to/checkpoint.pt \
                                   --model_cfg path/to/config.yaml \
                                   --dataset voxceleb2 \
                                   --data_dir /path/to/voxceleb2
```

## Supported Models

### Encoder Fine-tuning Strategies

Three strategies are supported:

| Strategy | Description |
|----------|-------------|
| `shared_mlp_zero_init_norm` | Shared MLP adapter with zero-init and L2 normalization |
| `shared_linear_zero_init_norm` | Shared linear adapter with zero-init and L2 normalization |
| `mrl` | Matryoshka Representation Learning (projection layer) |

All strategies share a single adapter between the query and context encoders.

### Indexing Strategies

| Strategy | Description |
|----------|-------------|
| `faiss_tree_rep` | FAISS-based tree representation indexing |
| `tree_rep_multi_index_faiss` | FAISS with multi-index (one per tree depth) |

### Split Functions

| Function | Description |
|----------|-------------|
| `linear` | Linear projection split |
| `mlp` | MLP projection split |
| `cross_attn` | Cross-attention split (supports token-level encoding) |

## Model Zoo

### Image Models (ImageNet-1K)

| Model | Encoder | Strategy | Config | NDCG@10 |
|-------|---------|----------|--------|---------|
| **ReTreever (Constant)** | DinoV2-Large | Tree-based | `imagenet/retreever_constant_dinov2.yaml` | TBD |
| **ReTreever (Stochastic)** | DinoV2-Large | Tree-based | `imagenet/retreever_stochastic_dinov2.yaml` | TBD |
| **MRL** | DinoV2-Large | Multi-resolution | `imagenet/mrl_dinov2.yaml` | TBD |

### Audio Models (VoxCeleb2)

| Model | Encoder | Strategy | Config | NDCG@10 |
|-------|---------|----------|--------|---------|
| **ReTreever (Stochastic)** | AST | Tree-based | `voxceleb/retreever_stochastic_ast.yaml` | TBD |
| **ReTreever (Constant)** | AST | Tree-based | `voxceleb/retreever_constant_ast.yaml` | TBD |
| **MRL** | AST | Multi-resolution | `voxceleb/mrl_ast.yaml` | TBD |

### Text Models (NQ, HotpotQA, RepliQA, TopiocQA)

| Dataset | Model | Encoder | Config | NDCG@10 |
|---------|-------|---------|--------|---------|
| **NQ** | ReTreever (MLP Adapter) | DistilBERT | `text/nq_retreever_mlp_distilbert.yaml` | TBD |
| **NQ** | ReTreever (Linear Adapter) | DistilBERT | `text/nq_retreever_linear_distilbert.yaml` | TBD |
| **NQ** | MRL | DistilBERT | `text/nq_mrl_distilbert.yaml` | TBD |
| **HotpotQA** | ReTreever (MLP Adapter) | DistilBERT | `text/hotpotqa_retreever_mlp_distilbert.yaml` | TBD |
| **RepliQA** | ReTreever (MLP Adapter) | DistilBERT | `text/repliqa_retreever_mlp_distilbert.yaml` | TBD |
| **TopiocQA** | ReTreever (MLP Adapter) | DistilBERT | `text/topiocqa_retreever_mlp_distilbert.yaml` | TBD |

### Multimodal Models (COCO, Flickr30k)

| Dataset | Model | Encoder | Config | NDCG@10 |
|---------|-------|---------|--------|---------|
| **COCO** | ReTreever | FLAVA | `multimodal/coco_flava.yaml` | TBD |
| **Flickr30k** | ReTreever | FLAVA | `multimodal/flickr_flava.yaml` | TBD |

## Directory Structure

```
retreever/
├── configs/              # Model configuration files
│   ├── imagenet/        # ImageNet configs
│   ├── voxceleb/        # VoxCeleb configs
│   ├── text/            # Text retrieval configs
│   └── multimodal/      # Multimodal configs
├── retreever/           # Main package
│   ├── models/          # Model architectures
│   │   ├── retreever.py # Main ReTreever model
│   │   ├── mrl.py       # MRL model
│   │   ├── adapters.py  # Adapter classes (3 strategies)
│   │   ├── encoders.py  # Encoder implementations
│   │   ├── split_functions.py  # Tree split functions
│   │   └── indexing_strategies.py  # FAISS indexing
│   ├── training/        # Training pipeline
│   ├── data/            # Data loading
│   ├── evaluation/      # Evaluation metrics
│   └── utils/           # Utilities
├── scripts/             # Training/evaluation scripts
│   ├── train.py         # Training script
│   ├── evaluate.py      # General evaluation
│   ├── evaluate_text.py # Text retrieval evaluation
│   ├── evaluate_image.py # Image retrieval evaluation
│   └── evaluate_audio.py # Audio retrieval evaluation
└── tests/               # Unit & integration tests
```

## Documentation

- [Training Guide](docs/training_guide.md) - Detailed training instructions
- [Configuration Guide](CONFIG_GUIDE.md) - Understanding config files and cache settings

## Citation

If you use ReTreever in your research, please cite:

```bibtex
@article{retreever2026,
  title={ReTreever: Hierarchical Retrieval with Matryoshka Representations},
  author={Your Team},
  journal={arXiv preprint},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

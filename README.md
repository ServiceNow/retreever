# ReTreever: Hierarchical Retrieval with Coarse-To-Fine Representations

**ReTreever** is a flexible framework for training and evaluating hierarchical retrieval models with multi-resolution representations. It supports multiple modalities (text, images, audio) and various training strategies including product propagation, Matryoshka Representation Learning (MRL), and encoder fine-tuning with adapters.

## Features

- 🌳 **Hierarchical Retrieval**: Tree-based retrieval with stochastic/constant depth sampling strategies
- 🎓 **Depth Curriculum Learning**: 5 schedulers (random heavy-tailed, random linear, random uniform, linear warmup, exponential warmup)
- 🎯 **Multi-Resolution Learning**: Matryoshka Representation Learning for efficient embedding compression
- 🔧 **Flexible Fine-tuning**: Support for MLP/Linear adapters with zero-init normalization
- 🖼️ **Multi-Modal**: Text (DistilBERT, BGE), Images (DinoV2, ResNet, CLIP), Audio (AST), Text-Image (FLAVA)
- ⚡ **Efficient Training**: DeepSpeed integration, mixed precision, and distributed training
- 📊 **Comprehensive Evaluation**: NDCG@k, Recall@k, MRR metrics with FAISS-based indexing

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
config = load_config("configs/imagenet/prod_prop_stochastic_dinov2.yaml")

# Create model
model = ReTreever(
    encoder_type="dinov2-large",
    tree_type="no_propagation_tree",
    tree_depth=10,
    **config.model
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
- Used in: `prod_prop_revisited_constant_*` models

**Stochastic Depth Training** (`hierarchical: true`)
- Randomly samples depth at each training step
- Better multi-resolution representations
- Used in: `prod_prop_revisited_stochastic_*` models

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
python scripts/train.py --config configs/imagenet/prod_prop_stochastic_dinov2.yaml

# Evaluate a trained model
python scripts/evaluate.py --checkpoint path/to/checkpoint.pt --dataset imagenet1k

# Train text retrieval on NQ dataset
python scripts/train.py --config configs/text/nq_adapter_mlp_distilbert.yaml
```

## Model Zoo

### Image Models (ImageNet1k)

| Model | Encoder | Strategy | Config | NDCG@10 |
|-------|---------|----------|--------|---------|
| **Product Propagation (Constant)** | DinoV2-Large | Tree-based | `imagenet/prod_prop_constant_dinov2.yaml` | TBD |
| **Product Propagation (Stochastic)** | DinoV2-Large | Tree-based | `imagenet/prod_prop_stochastic_dinov2.yaml` | TBD |
| **MRL** | DinoV2-Large | Multi-resolution | `imagenet/mrl_dinov2.yaml` | TBD |
| **Adapter (MLP)** | DinoV2-Large | Fine-tuned | `imagenet/adapter_mlp_dinov2.yaml` | TBD |
| **Adapter (Linear)** | DinoV2-Large | Fine-tuned | `imagenet/adapter_linear_dinov2.yaml` | TBD |

### Audio Models (VoxCeleb2)

| Model | Encoder | Strategy | Config | NDCG@10 |
|-------|---------|----------|--------|---------|
| **Product Propagation (Stochastic)** | AST | Tree-based | `voxceleb/prod_prop_stochastic_ast.yaml` | TBD |
| **Product Propagation (Constant)** | AST | Tree-based | `voxceleb/prod_prop_constant_ast.yaml` | TBD |
| **MRL** | AST | Multi-resolution | `voxceleb/mrl_ast.yaml` | TBD |
| **Adapter (MLP)** | AST | Fine-tuned | `voxceleb/adapter_mlp_ast.yaml` | TBD |

### Text Models (NQ, HotpotQA, RepliQA, TopiocQA)

| Dataset | Model | Encoder | Config | NDCG@10 |
|---------|-------|---------|--------|---------|
| **NQ** | MLP Adapter | DistilBERT | `text/nq_adapter_mlp_distilbert.yaml` | TBD |
| **NQ** | Linear Adapter | DistilBERT | `text/nq_adapter_linear_distilbert.yaml` | TBD |
| **NQ** | MRL | DistilBERT | `text/nq_mrl_distilbert.yaml` | TBD |
| **HotpotQA** | MLP Adapter | DistilBERT | `text/hotpotqa_adapter_mlp_distilbert.yaml` | TBD |
| **RepliQA** | MLP Adapter | DistilBERT | `text/repliqa_adapter_mlp_distilbert.yaml` | TBD |
| **TopiocQA** | MLP Adapter | DistilBERT | `text/topiocqa_adapter_mlp_distilbert.yaml` | TBD |

### Multimodal Models (COCO, Flickr30k)

| Dataset | Model | Encoder | Config | NDCG@10 |
|---------|-------|---------|--------|---------|
| **COCO** | FLAVA | FLAVA | `multimodal/coco_flava.yaml` | TBD |
| **Flickr30k** | FLAVA | FLAVA | `multimodal/flickr_flava.yaml` | TBD |

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
│   ├── training/        # Training pipeline
│   ├── data/            # Data loading
│   ├── evaluation/      # Evaluation metrics
│   ├── utils/           # Utilities
│   └── hub/             # Pre-trained models
├── scripts/             # Training/evaluation scripts
├── tests/               # Unit & integration tests
└── docs/                # Documentation

```

## Documentation

- [Training Guide](docs/training_guide.md) - Detailed training instructions
- [Model Zoo](docs/model_zoo.md) - All supported models and their configurations
- [API Reference](docs/api_reference.md) - Complete API documentation
- [Configuration Guide](docs/configuration.md) - Understanding config files

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

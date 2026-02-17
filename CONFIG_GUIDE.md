# Configuration Guide

This directory contains the main configuration file for the retreever package. The configuration centralizes environment-specific settings like cache directories, data paths, and compute options.

## Quick Start

The default configuration is already set up for the current system at `/mnt/dssk/data_rw/hf_cache`. If you're using the system as-is, you don't need to change anything.

## For Release / Your Own Setup

1. **Copy the template:**
   ```bash
   cp retreever/config_template.py retreever/config.py
   ```

2. **Edit `retreever/config.py`** to match your system:
   - Set `HF_CACHE_DIR` to where you want to store downloaded models
   - Update `DATA_ROOT` and `DATASET_PATHS` for your datasets
   - Adjust `CHECKPOINT_DIR`, `OUTPUT_DIR`, `EVAL_DIR` for your storage
   - Configure W&B settings if using Weights & Biases

3. **Do not commit** your customized `config.py` with sensitive paths

## Configuration Options

### HuggingFace Cache (`HF_CACHE_DIR`)

**What it does:** All encoder models (DistilBERT, BGE, DinoV2, CLIP, etc.) are downloaded from HuggingFace and cached locally. This setting controls where they're stored.

**Default:** `/mnt/dssk/data_rw/hf_cache` (current system)

**Why it matters:**
- First run downloads ~10-50GB of models (depending on which encoders you use)
- Subsequent runs reuse cached models (much faster)
- Should point to fast storage with enough space
- All models are already downloaded at the default location

**Example paths:**
- Development: `/home/username/.cache/huggingface`
- Shared cluster: `/mnt/shared/hf_cache`
- Fast SSD: `/ssd/cache/huggingface`

### Data Paths (`DATA_ROOT`, `DATASET_PATHS`)

**What it does:** Specifies where training and evaluation datasets are stored.

**Default:** `/mnt/dssk/data_rw` with subdirectories for each dataset

**Supported datasets:**
- Text retrieval: MS MARCO, Natural Questions, HotpotQA, TopiocQA
- Image retrieval: ImageNet-1K
- Audio retrieval: VoxCeleb2
- Multi-modal: COCO, Flickr30k

### Output Directories

- **`CHECKPOINT_DIR`**: Model checkpoints saved during training
- **`OUTPUT_DIR`**: Experiment logs and results
- **`EVAL_DIR`**: Evaluation metrics and analysis
- **`WANDB_DIR`**: Weights & Biases local cache

### Compute Settings

- **`DEFAULT_DEVICE`**: "cuda", "cpu", or "mps" (auto-detected)
- **`NUM_WORKERS`**: Number of data loader worker processes (default: 4)
- **`USE_AMP`**: Mixed precision training for faster computation (default: True)

## How It's Used

The config is imported throughout the codebase:

```python
from retreever import config

# Encoders automatically use config.HF_CACHE_DIR
encoder, _ = get_encoders("bge")  # Uses config cache

# You can override if needed
encoder, _ = get_encoders("bge", cache_dir="/custom/path")

# Access config values
print(config.HF_CACHE_DIR)
print(config.CHECKPOINT_DIR)
```

## Environment Variables Set

The config automatically sets these environment variables:
- `HF_HOME`: Main HuggingFace cache directory
- `TRANSFORMERS_CACHE`: Transformers library cache
- `HF_DATASETS_CACHE`: HuggingFace datasets cache
- `HUGGINGFACE_HUB_CACHE`: Hub client cache

These ensure **all** HuggingFace downloads go to the specified location.

## Verifying Configuration

Check your configuration:

```bash
cd /home/toolkit/retreever
python -c "from retreever import config; print(config.HF_CACHE_DIR)"
```

Or run the config as a script:

```bash
python retreever/config.py
```

## Testing with Configuration

The unit tests automatically use the configured cache directory:

```bash
# Tests use config.HF_CACHE_DIR by default
cd /home/toolkit/retreever
PYTHONPATH=/home/toolkit/retreever:$PYTHONPATH python -m pytest tests/

# All encoder downloads will use the configured cache
# Since models are already at /mnt/dssk/data_rw/hf_cache, tests run fast
```

## Troubleshooting

### Models downloading to wrong location

Check that environment variables are set:
```python
import os
print(os.environ.get("HF_HOME"))
print(os.environ.get("TRANSFORMERS_CACHE"))
```

Make sure you import config before instantiating any models:
```python
from retreever import config  # Sets environment variables
from retreever.models.encoders import get_encoders  # Now uses correct cache
```

### Permission denied errors

Ensure you have write permissions to all configured directories:
```bash
ls -la /mnt/dssk/data_rw/hf_cache
chmod -R u+w /mnt/dssk/data_rw/hf_cache
```

### Out of space errors

Check cache size and available space:
```bash
du -sh /mnt/dssk/data_rw/hf_cache
df -h /mnt/dssk/data_rw
```

Models by size (approximate):
- DistilBERT: ~250MB
- BGE-Large: ~1.3GB
- DinoV2-Giant: ~4GB
- CLIP-Large: ~1.7GB
- AST: ~350MB
- FLAVA: ~900MB

## For Production/Release

When preparing for release:

1. **Update config_template.py** with sensible defaults
2. **Document required storage** (total ~10-50GB depending on encoders used)
3. **Add config.py to .gitignore** (done)
4. **In README, tell users:**
   ```bash
   # First time setup
   cp retreever/config_template.py retreever/config.py
   # Edit retreever/config.py with your paths
   vim retreever/config.py
   ```

5. **Environment variable override:** Users can also set environment variables directly:
   ```bash
   export HF_HOME=/my/custom/cache
   python scripts/train.py
   ```

## Current System Configuration

The current system is configured with:
- **HF Cache:** `/mnt/dssk/data_rw/hf_cache` (models already downloaded)
- **Data Root:** `/mnt/dssk/data_rw`
- **Checkpoints:** `/mnt/dssk/data_rw/checkpoints`
- **Outputs:** `/mnt/dssk/data_rw/outputs`

This is optimized for the current development environment.

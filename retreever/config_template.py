"""
Configuration template for retreever package.

COPY THIS FILE to `retreever/config.py` and customize the paths for your setup.
DO NOT commit your customized config.py with sensitive information.

This template shows all configurable parameters with example values.
"""

import os
from pathlib import Path

# =============================================================================
# HUGGING FACE CACHE CONFIGURATION
# =============================================================================

# Primary cache directory for all HuggingFace model downloads
# This is where transformers, tokenizers, processors, and model weights are stored
# 
# CUSTOMIZE THIS PATH for your system:
# - Local machine: "/home/username/.cache/huggingface"
# - Shared cluster: "/mnt/shared/hf_cache"
# - Fast SSD: "/ssd/cache/huggingface"
HF_CACHE_DIR = "/path/to/your/hf_cache"

# Set environment variables for HuggingFace libraries
# These ensure all downloads go to the specified cache directory
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_CACHE_DIR, "datasets")
os.environ["HUGGINGFACE_HUB_CACHE"] = HF_CACHE_DIR

# =============================================================================
# DATA PATHS
# =============================================================================

# Root directory for datasets
# CUSTOMIZE THIS PATH for your system
DATA_ROOT = Path("/path/to/your/data")

# Specific dataset paths (customize as needed)
# Add or remove datasets based on what you're using
DATASET_PATHS = {
    "msmarco": DATA_ROOT / "msmarco",
    "nq": DATA_ROOT / "nq",
    "hotpotqa": DATA_ROOT / "hotpotqa",
    "imagenet": DATA_ROOT / "imagenet",
    "coco": DATA_ROOT / "coco",
    "flickr30k": DATA_ROOT / "flickr30k",
    "voxceleb2": DATA_ROOT / "voxceleb2",
}

# =============================================================================
# CHECKPOINT AND OUTPUT PATHS
# =============================================================================

# Directory for saving model checkpoints during training
# Should be on fast storage with enough space
CHECKPOINT_DIR = Path("/path/to/checkpoints")

# Directory for experiment outputs (logs, results, etc.)
OUTPUT_DIR = Path("/path/to/outputs")

# Directory for evaluation results and metrics
EVAL_DIR = Path("/path/to/eval_results")

# =============================================================================
# LOGGING AND WANDB CONFIGURATION
# =============================================================================

# Weights & Biases configuration
WANDB_PROJECT = "retreever"  # Your W&B project name
WANDB_ENTITY = None  # Set to your W&B username/team if needed
WANDB_DIR = Path("/path/to/wandb")  # Local W&B cache directory

# =============================================================================
# COMPUTE CONFIGURATION
# =============================================================================

# Default device for training/inference
# Options: "cuda", "cpu", "mps" (for Apple Silicon)
DEFAULT_DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

# Number of data loader workers
# Rule of thumb: 4 * num_gpus, but tune based on your system
NUM_WORKERS = 4

# Mixed precision training
# Set to True for faster training with minimal accuracy impact
USE_AMP = True

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_cache_dir() -> str:
    """Get the HuggingFace cache directory.
    
    Returns:
        str: Path to HF cache directory
    """
    return HF_CACHE_DIR


def get_dataset_path(dataset_name: str) -> Path:
    """Get path to a specific dataset.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Path: Path to dataset directory
        
    Raises:
        KeyError: If dataset name not found in DATASET_PATHS
    """
    if dataset_name not in DATASET_PATHS:
        raise KeyError(f"Dataset '{dataset_name}' not found. Available: {list(DATASET_PATHS.keys())}")
    return DATASET_PATHS[dataset_name]


def ensure_dirs() -> None:
    """Create all necessary directories if they don't exist."""
    for directory in [CHECKPOINT_DIR, OUTPUT_DIR, EVAL_DIR, WANDB_DIR, Path(HF_CACHE_DIR)]:
        directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# INITIALIZATION
# =============================================================================

# Ensure critical directories exist
ensure_dirs()

# Print configuration on import (useful for debugging)
if __name__ == "__main__":
    print("=" * 80)
    print("RETREEVER CONFIGURATION")
    print("=" * 80)
    print(f"HF Cache Directory: {HF_CACHE_DIR}")
    print(f"Data Root: {DATA_ROOT}")
    print(f"Checkpoint Directory: {CHECKPOINT_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Eval Directory: {EVAL_DIR}")
    print(f"W&B Project: {WANDB_PROJECT}")
    print(f"Default Device: {DEFAULT_DEVICE}")
    print("=" * 80)

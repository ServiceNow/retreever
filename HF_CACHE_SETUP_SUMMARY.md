# HuggingFace Cache Configuration - Summary

## What Was Done

Created a centralized configuration system for retreever that manages all environment-specific paths, especially the HuggingFace model cache.

### Files Created/Modified

1. **`retreever/config.py`** - Main configuration file
   - Sets `HF_CACHE_DIR = "/mnt/dssk/data_rw/hf_cache"`
   - Configures all environment variables (`HF_HOME`, `TRANSFORMERS_CACHE`, etc.)
   - Defines data paths, checkpoint directories, output paths
   - Already configured for current system - models are pre-downloaded

2. **`retreever/config_template.py`** - Template for users
   - Copy this to `config.py` for custom setups
   - Includes documentation for all configuration options
   - For release: users customize paths for their system

3. **`CONFIG_GUIDE.md`** - Comprehensive documentation
   - Explains what each configuration option does
   - How to set up for different environments
   - Troubleshooting common issues
   - ~300 lines of detailed documentation

4. **`.gitignore`** - Prevents committing sensitive paths
   - Ignores `retreever/config.py` (user-specific)
   - Keeps `retreever/config_template.py` (template for all users)

### Code Changes

Updated all encoder and model files to use the configuration:

- **`retreever/models/encoders.py`**: All encoders now use `config.HF_CACHE_DIR` by default
  - DistilBERTEncoder, BGEEncoder, DinoV2Encoder, ResNetEncoder
  - CLIPEncoder, ASTEncoder, FlavaEncoder
  - Tokenizers and processors also use the cache

- **`retreever/models/retreever.py`**: ReTreever model uses config for cache
- **`retreever/models/mrl.py`**: MRL model uses config for cache  
- **`scripts/train.py`**: Training script uses config (removed explicit `cache_dir=None`)

### How It Works

```python
# Before: Models downloaded to default HF cache (~/.cache/huggingface)
encoder, _ = get_encoders("bge")

# Now: Models use configured cache (/mnt/dssk/data_rw/hf_cache)
from retreever import config  # Sets environment variables
encoder, _ = get_encoders("bge")  # Uses config.HF_CACHE_DIR automatically

# Can still override if needed
encoder, _ = get_encoders("bge", cache_dir="/custom/path")
```

### Environment Variables Set

When `retreever.config` is imported, it automatically sets:
- `HF_HOME=/mnt/dssk/data_rw/hf_cache`
- `TRANSFORMERS_CACHE=/mnt/dssk/data_rw/hf_cache`
- `HF_DATASETS_CACHE=/mnt/dssk/data_rw/hf_cache/datasets`
- `HUGGINGFACE_HUB_CACHE=/mnt/dssk/data_rw/hf_cache`

This ensures **all** HuggingFace downloads go to the configured location.

## Current System Status

✅ **Working Configuration:**
- HF Cache: `/mnt/dssk/data_rw/hf_cache`
- All models already downloaded (no downloads needed for tests)
- Tests run fast using cached models
- Config tested and working

✅ **Verified:**
```bash
$ python retreever/config.py
================================================================================
RETREEVER CONFIGURATION
================================================================================
HF Cache Directory: /mnt/dssk/data_rw/hf_cache
Data Root: /mnt/dssk/data_rw
Checkpoint Directory: /mnt/dssk/data_rw/checkpoints
Output Directory: /mnt/dssk/data_rw/outputs
Eval Directory: /mnt/dssk/data_rw/eval_results
W&B Project: retreever
Default Device: cpu
================================================================================

$ python -c "from retreever.models.encoders import get_encoders; q, c = get_encoders('distilbert'); print('✅ Encoder loaded from cache')"
✅ Encoder loaded from cache
```

## For Release

When preparing retreever for public release:

1. **Users will need to:**
   ```bash
   # Copy template to active config
   cp retreever/config_template.py retreever/config.py
   
   # Edit with their paths
   vim retreever/config.py
   # Set HF_CACHE_DIR to their desired location
   # Set DATA_ROOT to their data directory
   # Etc.
   ```

2. **First run will download models** (~10-50GB depending on encoders used)
   - Subsequent runs reuse cached models
   - Can pre-download models: `python -c "from retreever.models.encoders import get_encoders; get_encoders('bge')"`

3. **Alternative: Environment variables**
   ```bash
   export HF_HOME=/my/custom/cache
   python scripts/train.py  # Will use custom cache
   ```

## Benefits

✅ **Centralized configuration** - All paths in one place  
✅ **Easy customization** - Single file to edit for different setups  
✅ **No hardcoded paths** - Works across different systems  
✅ **Git-safe** - User configs not committed (in .gitignore)  
✅ **Fast testing** - Models already cached, tests run quickly  
✅ **Documented** - CONFIG_GUIDE.md has full documentation  

## Testing with Config

All unit tests automatically use the configured cache:

```bash
cd /home/toolkit/retreever
PYTHONPATH=/home/toolkit/retreever:$PYTHONPATH python -m pytest tests/

# Encoders load from /mnt/dssk/data_rw/hf_cache
# No downloads needed - models already there
# Tests run fast
```

## Summary

The retreever package now has a proper configuration system that:
- Centralizes all environment-specific settings
- Points to `/mnt/dssk/data_rw/hf_cache` where models are already downloaded
- Works seamlessly with all encoders and model loading
- Provides a template for users to customize for their own systems
- Is fully documented with troubleshooting guides

**Everything is ready for development and release!** 🎉

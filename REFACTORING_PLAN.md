# Retreever Code Cleanup and Refactoring Plan

## Changes Required

### 1. Finetune Strategies - Keep Only These 3:
- `shared_mlp_zero_init_norm` 
- `shared_linear_zero_init_norm`
- `mrl`

**Remove:**
- last_layer
- linear
- linear_zero_init  
- mlp
- mlp_no_residual
- mlp_zero_init
- shared_mlp_zero_init
- pre_norm_mlp
- bottleneck
- adapter
- bitfit
- layernorm

### 2. Indexing Strategies - Remove:
- GreedyIndexing
- TreeRepAnnoyIndexing

**Keep:**
- TreeRepFaissIndexing (the main one used in production)

### 3. Split Functions - Keep Only These 3:
- linear
- mlp
- cross_attn

**Remove:**
- llm_split
- transformer_encoder  
- Any others not in the above list

### 4. Encoder Cache Directory:
- All encoder instantiations must accept and use `cache_dir` parameter
- Default to `config.HF_CACHE_DIR` when None
- Pass through from all call sites (train.py, eval scripts, tests)

### 5. Test Files Updates:
- Mock all encoder downloads in tests OR ensure they use cache_dir
- Remove tests for unsupported strategies
- Update test files to match simplified API

### 6. Evaluation Scripts:
Create comprehensive evaluation scripts based on research-dssk/comprehensive_evaluation.py:
- `scripts/evaluate_text.py` - For NQ, HotpotQA, etc.
- `scripts/evaluate_image.py` - For ImageNet  
- `scripts/evaluate_audio.py` - For VoxCeleb2

### 7. README Simplification:
- Remove "prod_prop_revisited" terminology
- Use clean names: "ReTreever", "ReTreever-Stochastic", "MRL"
- Product propagation is implicit in all ReTreever models
- Highlight: encoder + split_function + optional_adapter

##Implementation Strategy

### Phase 1: Core Code Cleanup (PRIORITY)
1. Remove unused adapter classes from retreever.py
2. Update ReTreever.__init__ to only support 3 strategies
3. Remove unused indexing strategies from indexing_strategies.py
4. Update trees.py (if needed) and split_functions.py

### Phase 2: Test File Updates
1. Mock encoders in tests using pytest fixtures
2. Remove test files/classes for unsupported features
3. Update remaining tests to use cache_dir properly

### Phase 3: Evaluation Scripts
1. Create evaluate_text.py based on comprehensive_evaluation.py
2. Create evaluate_image.py for ImageNet
3. Create evaluate_audio.py for VoxCeleb2
4. Document evaluation procedures

### Phase 4: Documentation
1. Update README.md with simplified model names
2. Update TESTING.md to reflect new test structure
3. Create EVALUATION.md guide

## Files To Modify

### retreever/models/:
- `retreever.py` - Remove unused adapters and strategies
- `indexing_strategies.py` - Remove unused strategies
- `split_functions.py` - Remove unused split functions (if any)
- `mrl.py` - Ensure cache_dir passed through

### tests/:
- `test_finetune_strategies.py` - Remove most tests, keep 3
- `test_indexing.py` - Remove tests for removed strategies
- `test_models.py` - Update to test only supported features
- Add `conftest.py` - Mock encoder fixtures

### scripts/:
- `train.py` - Ensure cache_dir passed to all models
- Create `evaluate_text.py`
- Create `evaluate_image.py`
- Create `evaluate_audio.py`

### Documentation:
- `README.md` - Simplify model naming
- `TESTING.md` - Update test instructions
- Create `EVALUATION.md`

## Verification Checklist

- [ ] Only 3 finetune strategies remain in code
- [ ] Only FAISS indexing supported
- [ ] Only 3 split functions supported
- [ ] All encoder calls use cache_dir
- [ ] Tests use mocked encoders or cache_dir
- [ ] Evaluation scripts created and tested
- [ ] README simplified and clear
- [ ] All tests pass
- [ ] Documentation complete

#!/usr/bin/env python3
"""
Comprehensive Test Suite Summary for ReTreever

This script provides a summary of all unit tests created for the retreever package.
Run with: python scripts/run_all_tests.py
"""

import subprocess
import sys

# Test files and their coverage
TEST_SUITES = {
    "test_trees.py": {
        "description": "All 5 tree types (qr_tree, probabilistic_tree, no_propagation_tree, no_tree, identity_tree)",
        "components": ["QuadraticallyRelaxedTree", "ProbabilisticallyRelaxedTree", "NoPropagationTree", "NoTree", "IdentityTree"],
        "test_classes": 6,
    },
    "test_split_functions.py": {
        "description": "All 3 split function types (linear, mlp, cross_attn)",
        "components": ["LinearSplit", "MLPSplit", "CrossAttentionSplit"],
        "test_classes": 8,
    },
    "test_depth_schedulers.py": {
        "description": "All 5 depth schedulers with stochastic depth scheduling",
        "components": ["LinearDepthScheduler", "ExponentialDepthScheduler", "RandomDepthScheduler", 
                       "RandomHeavyTailedDepthScheduler", "RandomUniformDepthScheduler"],
        "test_classes": 7,
    },
    "test_finetune_strategies.py": {
        "description": "All 15 encoder finetuning strategies",
        "components": ["last_layer", "linear", "linear_zero_init", "mlp", "mlp_no_residual", 
                       "mlp_zero_init", "shared_mlp_zero_init", "shared_mlp_zero_init_norm",
                       "shared_linear_zero_init_norm", "pre_norm_mlp", "mrl", "bottleneck", 
                       "adapter", "bitfit", "layernorm"],
        "test_classes": 8,
    },
    "test_encoders_comprehensive.py": {
        "description": "All encoder modalities: text (2), image (12), audio (1), multi-modal (1)",
        "components": ["distilbert", "bge", "dinov2-small/base/large/giant", "resnet18/34/50/101/152",
                       "clip-vit-base-patch32/16", "clip-vit-large-patch14", "ast", "flava"],
        "test_classes": 12,
    },
    "test_product_propagation.py": {
        "description": "Core product propagation algorithm and tree routing",
        "components": ["Product propagation", "Tree routing", "Split decisions", "Hierarchical representations"],
        "test_classes": 10,
    },
    "test_models.py": {
        "description": "ReTreever and MRL model instantiation and forward passes",
        "components": ["ReTreever", "MRL", "Model configurations", "Gradient flow", "Evaluation modes"],
        "test_classes": 10,
    },
    "test_indexing.py": {
        "description": "All 3 indexing strategies (greedy, Annoy, FAISS)",
        "components": ["GreedyIndexing", "TreeRepAnnoyIndexing", "TreeRepFaissIndexing"],
        "test_classes": 11,
    },
}

def print_summary():
    """Print test suite summary."""
    print("=" * 80)
    print("RETREEVER COMPREHENSIVE UNIT TEST SUITE")
    print("=" * 80)
    print()
    
    total_classes = 0
    total_components = 0
    
    for test_file, info in TEST_SUITES.items():
        print(f"📝 {test_file}")
        print(f"   Description: {info['description']}")
        print(f"   Test Classes: {info['test_classes']}")
        print(f"   Components Tested: {len(info['components'])}")
        print()
        
        total_classes += info['test_classes']
        total_components += len(info['components'])
    
    print("=" * 80)
    print(f"TOTAL: {len(TEST_SUITES)} test files, {total_classes} test classes, {total_components}+ components")
    print("=" * 80)
    print()

def run_tests(test_file=None):
    """Run pytest on specified test file or all tests."""
    print("\n🧪 Running Tests...\n")
    
    cmd = [
        "python", "-m", "pytest",
        "tests/" + (test_file if test_file else ""),
        "--override-ini=addopts=",
        "--disable-warnings",
        "-v",
        "--tb=short",
    ]
    
    result = subprocess.run(cmd, cwd="/home/toolkit/retreever")
    return result.returncode

if __name__ == "__main__":
    print_summary()
    
    print("\nTest Coverage Summary:")
    print("✅ Tree Types: 5 types covering product propagation and stochastic routing")
    print("✅ Split Functions: 3 types (linear, MLP, cross-attention)")
    print("✅ Depth Schedulers: 5 schedulers for hierarchical training")
    print("✅ Finetuning Strategies: 15 adapter and parameter-efficient methods")
    print("✅ Encoders: 16 encoders across text, image, audio, and multi-modal")
    print("✅ Product Propagation: Core algorithm with probability distributions")
    print("✅ Models: ReTreever and MRL with all configurations")
    print("✅ Indexing: 3 strategies for efficient retrieval")
    print()
    
    print("Run individual test files with:")
    print("  cd /home/toolkit/retreever")
    print("  PYTHONPATH=/home/toolkit/retreever:$PYTHONPATH python -m pytest tests/test_<name>.py --override-ini='addopts=' --disable-warnings")
    print()
    
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        sys.exit(run_tests(test_file))
    else:
        print("To run all tests: python scripts/run_all_tests.py all")
        print("To run specific test: python scripts/run_all_tests.py test_trees.py")

"""Metrics for evaluating tree-based retrieval during training."""

import numpy as np
from scipy.special import rel_entr
from scipy.stats import entropy
import evaluate

from transformers import EvalPrediction
from retreever.utils.algo import BinarySearchTree

# Load metrics
accuracy = evaluate.load("accuracy")


def retreeval_metrics(eval_pred: EvalPrediction):
    """
    Compute tree-based retrieval metrics during training.
    
    Args:
        eval_pred: EvalPrediction object with predictions and labels
            predictions[0]: query logits (batch_size, num_nodes)
            predictions[1]: context logits (batch_size, num_nodes)
            predictions[2]: depth tensor
            
    Returns:
        Dictionary of metrics
    """
    metrics = {}

    q_logits = eval_pred.predictions[0]
    c_logits = eval_pred.predictions[1]
    depth = eval_pred.predictions[2]
    
    depth = depth[0].item()
    start_index = 2 ** depth - 1
    end_index = 2 ** (depth + 1) - 1
    
    if end_index > q_logits.shape[1]:
        print(
            f"Skipping metrics: requested depth {depth}, but logits shape is {q_logits.shape}"
        )
        return metrics
    
    q_logits = q_logits[:, start_index:end_index]
    c_logits = c_logits[:, start_index:end_index]

    # Accuracy: query and context are hard-assigned to same leaf
    q_labels = np.argmax(q_logits, 1)
    c_labels = np.argmax(c_logits, 1)

    metrics |= accuracy.compute(references=q_labels, predictions=c_labels)

    # Tree-specific metrics
    metrics |= sparsity_measure(c_logits, "ctx")
    metrics |= sparsity_measure(q_logits, "query")

    metrics |= assignments_spread(c_labels, c_logits.shape, "ctx")
    metrics |= assignments_spread(q_labels, q_logits.shape, "query")
    metrics |= lowest_common_ancestor_depth(c_labels, q_labels, c_logits.shape[1], "query")

    return metrics


def sparsity_measure(logits, tag="logits"):
    """
    Compute sparsity measures for tree assignments.
    
    Measures:
    1. Percentage of total mass in the highest-weight position
    2. Number of top positions needed to account for 90% of total weight
    
    Args:
        logits: Logits array (2D) where each row is a sample
        tag: Tag for labeling metrics
        
    Returns:
        Dictionary of sparsity metrics
    """
    # Normalize logits into probabilities (sum to 1)
    probabilities = logits / logits.sum(axis=1, keepdims=True)

    # Metric 1: % of mass in the highest-weight position
    max_probs = np.max(probabilities, axis=1)
    avg_max_prob = max_probs.mean()

    # Metric 2: Number of top positions to account for 90% of total weight
    sorted_probs = np.sort(probabilities, axis=1)[:, ::-1]
    cumulative_sums = np.cumsum(sorted_probs, axis=1)
    top_k_coverage = np.argmax(cumulative_sums >= 0.9, axis=1) + 1
    avg_top_k_coverage = np.mean(top_k_coverage)

    return {
        f"{tag}-avg-max-prob": avg_max_prob,
        f"{tag}-avg-top-k-90": avg_top_k_coverage,
    }


def assignments_spread(labels, num_leaves, tag="ctx"):
    """
    Compute metrics to evaluate tree utilization and balance.
    
    Args:
        labels: Hard leaf assignments (1D array)
        num_leaves: Shape tuple, second element is number of leaves
        tag: Tag for labeling metrics
        
    Returns:
        Dictionary of tree spread metrics
    """
    # Create one-hot representation
    one_hot_labels = np.zeros(num_leaves)
    one_hot_labels[np.arange(len(labels)), labels] = 1

    # Count points assigned to each leaf and normalize
    leaf_rates = one_hot_labels.sum(0)
    leaf_rates /= leaf_rates.sum()

    # Entropy: higher = more balanced
    labels_entropy = entropy(leaf_rates).sum()
    
    # KL divergence from uniform: lower = more uniform
    labels_kl = rel_entr(leaf_rates, 1 / len(leaf_rates)).sum()

    # Maximal margin between leaf rates (measures imbalance)
    min_prob, max_prob = min(leaf_rates), max(leaf_rates)

    # Number of used leaves
    num_used_leaves = len(np.unique(labels))

    return {
        f"{tag}-label-entropy": labels_entropy,
        f"{tag}-label-kl": labels_kl,
        f"{tag}-leaf-rate-margin": max_prob - min_prob,
        f"{tag}-used-leaves": num_used_leaves,
        f"{tag}-used-leaves-pct": num_used_leaves / num_leaves[1],
    }


def lowest_common_ancestor_depth(c_labels, q_labels, num_leaves, tag="query"):
    """
    Compute the Lowest Common Ancestor (LCA) metrics between predicted
    query labels and context labels.
    
    Higher normalized LCA depth = query and context routed closer together
    
    Args:
        c_labels: Context leaf assignments
        q_labels: Query leaf assignments
        num_leaves: Number of leaves in the tree
        tag: Tag for labeling metrics
        
    Returns:
        Dictionary of LCA metrics
    """
    # Compute tree depth from number of leaves
    tree_depth = int(np.log2(num_leaves))

    # Define BST for LCA computation
    bst = BinarySearchTree(tree_depth)
    vectorized_LCA = np.vectorize(bst.find_LCA)
    vectorized_node_depth = np.vectorize(bst.get_node_depth)

    # Convert to full tree node indices (add offset for leaf layer)
    q_labels += 2**tree_depth - 1
    c_labels += 2**tree_depth - 1

    # Compute LCA nodes
    LCA_nodes = vectorized_LCA(q_labels, c_labels)
    
    # Get depth of each LCA node
    LCA_depth = vectorized_node_depth(LCA_nodes)
    
    # Normalize: perfect score is 1 (same leaf)
    LCA_normalized_depth = LCA_depth / tree_depth

    return {
        f"{tag}-LCA-norm-avg-depth": LCA_normalized_depth.mean(),
        f"{tag}-LCA-norm-min": LCA_normalized_depth.min(),
        f"{tag}-LCA-norm-max": LCA_normalized_depth.max(),
    }

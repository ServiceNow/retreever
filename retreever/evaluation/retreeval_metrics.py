import numpy as np

from scipy.special import rel_entr
from scipy.stats import entropy

import evaluate  # TODO: Add the right version of evaluate to requirements.txt

from transformers import EvalPrediction
from retreever.utils.algo import BinarySearchTree

# load metrics
accuracy = evaluate.load("accuracy")


def retreeval_metrics(eval_pred: EvalPrediction):
    metrics = {}

    q_logits = eval_pred.predictions[0]
    c_logits = eval_pred.predictions[1]
    depth = eval_pred.predictions[2]
    
    depth = depth[0].item()
    start_index = 2 ** depth - 1
    end_index = 2 ** (depth + 1) - 1
    
    if end_index > q_logits.shape[1]:
        print(f"Something off has happened with the retrreval metrics computation, requested depth {depth}, but the dimensionso f the logits are {q_logits.shape}, skipping metrics computation.")
        return metrics
    
    q_logits = q_logits[:, start_index:end_index]
    c_logits = c_logits[:, start_index:end_index]

    # accuracy: query and context are hard-assigned to same leaf
    q_labels = np.argmax(q_logits, 1)
    c_labels = np.argmax(c_logits, 1)

    metrics |= accuracy.compute(references=q_labels, predictions=c_labels)

    metrics |= sparsity_measure(c_logits, "ctx")
    metrics |= sparsity_measure(q_logits, "query")

    metrics |= assignments_spread(c_labels, c_logits.shape, "ctx")
    metrics |= assignments_spread(q_labels, q_logits.shape, "query")
    metrics |= lowest_common_ancestor_depth(c_labels, q_labels, c_logits.shape[1], "query")

    return metrics


def sparsity_measure(logits, tag="logits"):
    """
    Computes sparsity measures for the given logits:
    1. Percentage of total mass in the logits (after normalization) contained in the position with the highest weight.
    2. Number of top positions needed to account for 90% of the total weight.

    Args:
        logits (np.ndarray): Logits array (2D) where each row corresponds to a sample.
        tag (str): A tag to label the metrics.

    Returns:
        dict: A dictionary containing the sparsity metrics.
    """
    # Normalize logits into probabilities (sum to 1)
    probabilities = logits / logits.sum(axis=1, keepdims=True)

    # Metric 1: % of mass in the highest-weight position
    max_probs = np.max(probabilities, axis=1)
    avg_max_prob = (
        max_probs.mean()
    )  # Average percentage of total mass in the highest-weight position

    # Metric 2: Number of top positions to account for 90% of total weight
    # Sort probabilities along the last axis in descending order
    sorted_probs = np.sort(probabilities, axis=1)[:, ::-1]
    # Compute cumulative sums along the sorted probabilities
    cumulative_sums = np.cumsum(sorted_probs, axis=1)
    # Find the first index where cumulative sum >= 90% for each row
    top_k_coverage = np.argmax(cumulative_sums >= 0.9, axis=1) + 1
    avg_top_k_coverage = np.mean(top_k_coverage)  # Average number of top positions for 90% weight

    return {
        f"{tag}-avg-max-prob": avg_max_prob,
        f"{tag}-avg-top-k-90": avg_top_k_coverage,
    }


def assignments_spread(labels, num_leaves, tag="ctx"):
    """Computes several metrics to evaluate how much of the tree is used and what are the unbalances."""

    # spread of labels (hard scores) over leaves. The higher the better
    one_hot_labels = np.zeros(num_leaves)
    one_hot_labels[
        np.arange(len(labels)), labels
    ] = 1  # one-hot representation of hard leaf assignments

    # count number of points assigned to each leaf and normalize
    leaf_rates = one_hot_labels.sum(0)
    leaf_rates /= leaf_rates.sum()

    labels_entropy = entropy(leaf_rates).sum()  # the higher, the more balanced the tree
    labels_kl = rel_entr(
        leaf_rates, 1 / len(leaf_rates)
    ).sum()  # the lower, the closer to a uniform utilizing of the tree

    # Maximal margin between leaf rates (to measure usage unbalance)
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
    Computes the least common ancestor metrics between the  q_labels (predicted)
    and c_labels (hard gt assigments) based on the tree depth.
    """

    # Compute the tree depth from the number of leaves
    tree_depth = np.log2(num_leaves)
    tree_depth = int(tree_depth)

    # Define a bst to use its LCA function
    bst = BinarySearchTree(tree_depth)
    # Vectorize the LCA function
    vectorized_LCA = np.vectorize(bst.find_LCA)
    vectorized_node_depth = np.vectorize(bst.get_node_depth)

    # Get the original leaf index
    q_labels += 2**tree_depth - 1
    c_labels += 2**tree_depth - 1

    # Get the list of all LCA nodes between query predictions and their gt contexts.
    LCA_nodes = vectorized_LCA(q_labels, c_labels)
    # We are interested in the depth of the lowest common ancestor
    # So compute the depth of each LCA node
    LCA_depth = vectorized_node_depth(LCA_nodes)
    # Normalize the depth of the LCA
    # A perfect normalized score is 1 (gt and pred in same leaf)
    LCA_normalized_depth = LCA_depth / tree_depth

    return {
        f"{tag}-LCA-norm-avg-depth": LCA_normalized_depth.mean(),
        f"{tag}-LCA-norm-min": LCA_normalized_depth.min(),
        f"{tag}-LCA-norm-max": LCA_normalized_depth.max(),
    }

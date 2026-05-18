# Adapted from https://github.com/vzantedeschi/LatentTrees/tree/main
# TODO: license

import numpy as np
import torch

from typing import Tuple

from retreever.models.split_functions import split_dict
from retreever.utils.algo import BinarySearchTree

# ----------------------------------------------------------------------- TREE MODULES


class Tree(torch.nn.Module):
    def __init__(
        self,
        input_size: Tuple[int],
        depth: int,
        split_fn: str,
        needs_propagation: bool,
        **split_fn_params,
    ):
        """Module to enforce tree structure.

        Args:
            input_size (Tuple[int]): size of inputs to route, e.g., (embedding_dim,) or (num_tokens, embedding_dim). Inputs will be flattened.
            depth (int): depth of the tree
            split_fn (str): type of split function. Supported options are provided in retreever.models.split_functions.split_dict
        """
        super(Tree, self).__init__()

        # class to go from vector to tree semantics (e.g. to know which indexes correspond to leaves and which to split nodes)
        self.bst = BinarySearchTree(depth)
        self.num_leaves = self.bst.nb_leaves

        self.split_type = split_fn
        split_fn_out = self.bst.nb_split if needs_propagation else self.bst.nb_nodes
        split_fn_params["tree_depth"] = depth
        self.split = split_dict[self.split_type](input_size, split_fn_out, **split_fn_params)

        # explicit bias initialized so that tree is balanced for first batch. Only needed for linear split fn
        if self.split_type == "linear":
            self.offset = torch.nn.Parameter(torch.zeros(self.bst.nb_split), requires_grad=True)
        else:
            self.offset = torch.zeros(self.bst.nb_split)

    def predict(self, x: torch.Tensor):
        """Returns a point's route and leaf assignment

        Args:
            x (torch.Tensor): batch of inputs to route

        Returns:
            (torch.Tensor, torch.Tensor): split scores for the whole tree, leaf id which each point is assigned to
        """

        routes = self.forward(x, False)

        labels = torch.argmax(routes[:, self.bst.leaves], 1)

        return routes, labels

    def _init_split_fn(self, x: torch.Tensor, depth: int = 0):
        """Initialize split offset so that points are routed everywhere with equal probability, starting from chosen <depth>."""
        # This initialization to be done only if split is linear.
        if self.split_type != "linear":
            return

        start_index = 2**depth - 1
        self.split._init_params(start_index)

        scores = self.split(x)
        offset = self.offset.detach().clone()
        offset[start_index:] = (
            -scores[:, start_index:].mean(0)
            + 0.1 * torch.rand(scores.size(1) - start_index, device=scores.device)
            - 0.05
        )

        # set of points assigned to each node
        # all points are assigned to root
        node_masks = [np.array([True] * len(x))]

        # loop over left nodes (skip root)
        for left in range(1, self.bst.nb_split, 2):
            right = left + 1  # right node
            parent = self.bst.get_parent(left)  # parent node

            score_parent = (
                (scores[:, parent] + offset[parent]).cpu().detach().numpy()
            )  # parent's score

            node_masks.append(
                node_masks[parent] & (score_parent > 0)
            )  # points going to the left node
            node_masks.append(
                node_masks[parent] & (score_parent < 0)
            )  # points going to the right node

            if left >= start_index:
                # adjust offset so that tree is balanced
                if sum(node_masks[left]) > 0:
                    offset[left] = -scores[node_masks[left], left].mean()

                if sum(node_masks[right]) > 0:
                    offset[right] = -scores[node_masks[right], right].mean()

        self.offset = torch.nn.parameter.Parameter(offset, requires_grad=True)

    def _ancestor_bounded_score_propagation(self, split_scores):
        B, _ = split_scores.shape

        # init to 1, as anyway tree_scores will later be clamped between [0, 1]
        bounded_scores = torch.ones((B, self.bst.nb_nodes), device=split_scores.device)

        # upper bound children's score to parent's score
        # trick to avoid inplace operations involving bounded_scores
        bounded_scores[:, self.bst.desc_left] = torch.min(
            bounded_scores[:, self.bst.split_nodes], split_scores
        )  # left children get positive scores
        bounded_scores[:, self.bst.desc_right] = torch.min(
            bounded_scores[:, self.bst.split_nodes], -split_scores
        )  # right children get negative scores

        for _ in range(self.bst.depth):
            bounded_scores[:, self.bst.desc_left] = torch.min(
                bounded_scores[:, self.bst.desc_left], bounded_scores[:, self.bst.split_nodes]
            )  # upper bound left children scores with their parents'
            bounded_scores[:, self.bst.desc_right] = torch.min(
                bounded_scores[:, self.bst.desc_right], bounded_scores[:, self.bst.split_nodes]
            )  # upper bound child children scores with their parents'

        bounded_scores = self.bound_scores(bounded_scores)
        return bounded_scores

    def _probability_score_propagation(self, split_scores):
        B, _ = split_scores.shape

        # Step 1: Convert split scores to probabilities using the sigmoid function
        probabilities = torch.sigmoid(split_scores)

        # Step 2: Initialize scores for each node to 1 (since we're using multiplicative probabilities)
        propagated_scores = torch.ones((B, self.bst.nb_nodes), device=split_scores.device)

        # Step 3: Propagate probabilities down the tree
        for _ in range(self.bst.depth):
            # For left children, multiply with the probability of going left
            propagated_scores[:, self.bst.desc_left] = (
                propagated_scores[:, self.bst.split_nodes] * probabilities
            )

            # For right children, multiply with the probability of going right (1 - probability)
            propagated_scores[:, self.bst.desc_right] = propagated_scores[
                :, self.bst.split_nodes
            ] * (1 - probabilities)

        return propagated_scores

    def forward(self, x: torch.Tensor, mask: torch.Tensor, depth: int = -1):
        """Routes x through the tree, i.e., call all split functions and applies hierarchical constraints.

        Args:
            x (torch.Tensor): points to be routed, of shape (num_points, *input_shape)
            mask (torch.Tensor): mask to indicate which tokens in x are relevant. Meaningful when token level encoding used.
            depth (int, optional): if non-negative, returns the tree scores only for the nodes at the given depth. -1 (all depths).

        Returns:
            torch,Tensor: tree scores (one scalar per node) of shape (num_points, num_nodes)
        """

        # soft_routes[i] is the score assigned to node i via score propagation through the tree.
        soft_routes = self._compute_tree_scores(x, mask)

        if depth > -1:
            return soft_routes[:, 2**depth - 1 : 2 ** (depth + 1) - 1]
        else:
            return soft_routes

    def get_bias(self):
        return self.offset.detach()

    def bound_scores(self, scores):
        if self.bounding_fn == "clamp":
            return torch.clamp(scores + 0.5, 0, 1)
        elif self.bounding_fn == "sigmoid":
            return torch.sigmoid(scores)
        raise ValueError(f"Bounding fnc {self.bounding_fn} unrecognized")


class QuadraticallyRelaxedTree(Tree):
    def __init__(
        self,
        input_size: Tuple[int],
        depth: int,
        split_fn: str = "linear",
        **split_fn_params,
    ):
        self.bounding_fn = "clamp"
        self.propagation_fn = "ancestor_bounded"
        super(QuadraticallyRelaxedTree, self).__init__(
            input_size, depth, split_fn, needs_propagation=True, **split_fn_params)

        self.can_train = False

    def _compute_tree_scores(self, x: torch.Tensor, mask: torch.Tensor):
        """Returns the tree evaluations of the points. Each dimension i is the output of the i-th split function, upper bounded by the ancestors' split function outputs."""
        # compute split outputs
        split_scores = self.split(x, mask)
        if self.split_type == "linear":
            split_scores += self.offset
        bounded_scores = self._ancestor_bounded_score_propagation(split_scores)
        return bounded_scores


class ProbabilisticallyRelaxedTree(Tree):
    def __init__(
        self,
        input_size: Tuple[int],
        depth: int,
        split_fn: str = "linear",
        **split_fn_params,
    ):
        self.bounding_fn = "sigmoid"
        self.propagation_fn = "probability"

        super(ProbabilisticallyRelaxedTree, self).__init__(
            input_size, depth, split_fn, needs_propagation=True, **split_fn_params
        )

        self.can_train = False

    def _compute_tree_scores(self, x: torch.Tensor, mask: torch.Tensor):
        """Returns the tree evaluations of the points. Each dimension i is the output of the i-th split function, upper bounded by the ancestors' split function outputs."""
        # compute split outputs
        split_scores = self.split(x, mask)

        if self.split_type == "linear":
            split_scores += self.offset

        # Propagate scores through the tree to get scores for leaf node
        bounded_scores = self._probability_score_propagation(split_scores)
        return bounded_scores


class NoTree(torch.nn.Module):
    def __init__(
        self,
        input_size: Tuple[int],
        split_fn: str,
        **split_fn_params,
    ):
        """Module that keeps representations flat and non-probabilistic, hence does not apply tree constraints.

        Args:
            input_size (Tuple[int]): size of inputs to route, e.g., (embedding_dim,) or (num_tokens, embedding_dim). Inputs will be flattened.
            split_fn (str): type of split function. Supported options are provided in retreever.models.split_functions.split_dict
        """
        super(NoTree, self).__init__()

        self.num_leaves = 2 ** split_fn_params.get("depth")
        # ignore depth param, as not needed for this flat tree
        split_fn_params.pop("depth", None)

        self.split_type = split_fn
        self.split = split_dict[self.split_type](input_size, self.num_leaves, **split_fn_params)

        # explicit bias initialized so that tree is balanced for first batch. Only needed for linear split fn
        if self.split_type == "linear":
            self.offset = torch.nn.Parameter(torch.zeros(input_size), requires_grad=True)
        else:
            self.offset = torch.zeros(input_size)

    def _init_split_fn(self, *args, **kwargs):
        pass

    def forward(self, x: torch.Tensor, mask: torch.Tensor, depth: int = -1, *args, **kwargs):
        """Calls all split functions and returns their outputs without applying any tree propagation.

        Args:
            x (torch.Tensor): points to be routed, of shape (num_points, *input_shape)
            mask (torch.Tensor): mask to indicate which tokens in x are relevant. Meaningful when token level encoding used.
            depth (int, optional): if non-negative, returns the first 2**depth dimensions of the representation. -1 (all dimensions).

        Returns:
            torch,Tensor: tree scores (one scalar per node) of shape (num_points, num_nodes)
        """

        split_scores = self.split(x, mask)

        if self.split_type == "linear":
            split_scores += self.offset

        if depth > -1:
            return split_scores[:, : 2**depth]
        else:
            return split_scores

    def get_bias(self):
        return self.offset.detach()


class NoPropagationTree(Tree):
    def __init__(
        self,
        input_size: Tuple[int],
        depth: int,
        split_fn: str = "linear",
        **split_fn_params,
    ):
        """Module that still works with tree constraints, but the split function takes care of score propagation.
        The tree itself does not impose any propagation as is done in ProbabilityRelaxedTrees.
        The tree is simply responsible to produce a valid probability distribution at each level of the tree, and hence
        it performs a softmax at each level with a learnable temp coeff.
        
        TODO: Explore merging this in Propabilistically Relaxed Tree.

        Args:
            input_size (Tuple[int]): size of inputs to route, e.g., (embedding_dim,) or (num_tokens, embedding_dim). Inputs will be flattened.
            depth (int): depth of the tree
            split_fn (str): type of split function. Supported options are provided in retreever.models.split_functions.split_dict
        """
        self.bounding_fn = "sigmoid"

        self.prod_bound_in_split = split_fn_params.get("scoring_fn_name") == "product_propagation"
        super(NoPropagationTree, self).__init__(
            input_size, depth, split_fn, needs_propagation=self.prod_bound_in_split, **split_fn_params
        )
        
        self.dropout_location = split_fn_params.get("dropout_location", "inside_split")

        self.depth = depth
        self.temp_coeff = torch.nn.Parameter(torch.tensor(0.0))  # Learnable temperature

        self.dropout = torch.nn.Dropout(0.1)


    def _compute_tree_scores(self, x: torch.Tensor, mask: torch.Tensor):
        """Returns the tree evaluations of the points. Each dimension i is the output of the i-th split function, upper bounded by the ancestors' split function outputs."""
        # compute split outputs
        # split_scores = self.dropout(self.split(x, mask))
        split_scores = self.split(x, mask)[0]
        
        if self.dropout_location in ["after_split", "both"]:
            split_scores = self.dropout(split_scores)
        
        if self.prod_bound_in_split:
            return split_scores
        
        if self.split_type == "linear":
            split_scores += self.offset
            

        # Normalize scores per level
        bounded_scores = torch.ones(split_scores.shape, device=split_scores.device)
        start = 1
        for level in range(1, self.depth + 1):
            num_nodes = 2 ** level  # Number of nodes at this level
            end = start + num_nodes

            # Collect normalized scores
            bounded_scores[:, start:end] = torch.softmax(split_scores[:, start:end] * torch.exp(self.temp_coeff).clamp(1e-4, 30.0), dim=-1)
            start = end

        return bounded_scores
        # Initialize bounded_scores with the same shape as split_scores.
        
        # (Assumes total nodes = 2^(depth+1) - 1)
        # bounded_scores = torch.zeros(split_scores.shape, device=split_scores.device)

        # # Compute indices for the last level:
        # # Last level: indices [2**self.depth - 1, 2**(self.depth+1) - 1)
        # start_last = 2 ** self.depth - 1
        # end_last = 2 ** (self.depth + 1) - 1

        # # Normalize only the last level with softmax.
        # bounded_scores[:, start_last:end_last] = torch.softmax(
        #     split_scores[:, start_last:end_last] * torch.exp(self.temp_coeff).clamp(1e-4, 30.0), dim=-1
        # )

        # # Now, propagate scores upward level-by-level.
        # # For each non-leaf level (from level self.depth-1 down to level 0):
        # for level in reversed(range(self.depth)):
        #     # Determine the parent's indices for this level.
        #     # For level 0: indices [0, 1), for level i (> 0): indices [2**level - 1, 2**(level+1) - 1)
        #     start_parent = 2 ** level - 1
        #     end_parent = 2 ** (level + 1) - 1

        #     # Child level (immediately below) indices:
        #     start_child = 2 ** (level + 1) - 1
        #     end_child = 2 ** (level + 2) - 1
        #     num_parents = 2 ** level
        #     # Number of children should be 2 * num_parents.
        #     # Extract child scores and reshape: shape -> (batch_size, num_parents, 2)
        #     child_scores = bounded_scores[:, start_child:end_child].view(bounded_scores.size(0), num_parents, 2)
        #     # Sum the two children scores for each parent.
        #     parent_scores = child_scores.sum(dim=-1)  # shape: (batch_size, num_parents)
        #     # Assign these sums to the parent's positions.
        #     bounded_scores[:, start_parent:end_parent] = parent_scores

        # return bounded_scores
    

class IdentityTree(torch.nn.Module):
    def __init__(
        self,
        input_size: Tuple[int],
        depth: int,
        split_fn: str = "linear",
        **split_fn_params,
    ):
        """Identity tree that simply returns input unchanged."""
        super(IdentityTree, self).__init__()
        
        self.num_leaves = 2 ** depth
        self.depth = depth
        
        # Create a dummy offset for compatibility with get_bias()
        self.offset = torch.zeros(1)

    def _init_split_fn(self, *args, **kwargs):
        pass

    def forward(self, x: torch.Tensor, mask: torch.Tensor, depth: int = -1, *args, **kwargs):
        """Returns input unchanged."""
        assert len(x.shape) == 2
        return x

    def get_bias(self):
        return self.offset.detach()
    

tree_dict = {  # supported tree modules
    "qr_tree": QuadraticallyRelaxedTree,
    "probabilistic_tree": ProbabilisticallyRelaxedTree,
    "no_propagation_tree": NoPropagationTree,
    "no_tree": NoTree,
    "identity_tree": IdentityTree, 
}

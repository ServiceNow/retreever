import numpy as np

from typing import Iterable, Optional

"""Utilities that do not fit in another file, and that when you look at them you say "yep, that's an algorithm".

We shouldn't need "fancy" imports for this file.
"""


class BinarySearchTree:
    def __init__(self, depth: int = 2):
        r"""Utils to get binary tree semantics, e.g., for converting from flat (vector) indices to tree level.
        Example: search tree of depth=2
        max num nodes T = 2^(depth+1) - 1 = 7 nodes.

               0
             /   \
            1     2
           / \   / \
          3   4 5   6

        Taken from https://github.com/vzantedeschi/LatentTrees/tree/main
        TODO: license
        """
        self.depth = depth

        self.nb_nodes = 2 ** (depth + 1) - 1
        self.nodes = range(self.nb_nodes)

        # split nodes indices
        self.nb_split = 2**depth - 1
        self.split_nodes = range(self.nb_split)

        # leaf nodes indices
        self.leaves = range(self.nb_split, self.nb_nodes)
        self.nb_leaves = len(self.leaves)

        # left nodes
        self.desc_left = range(1, self.nb_nodes, 2)

        # right nodes
        self.desc_right = range(2, self.nb_nodes, 2)

    def get_parent(self, n: int):
        """Returns parent's index"""
        if n == 0 or n >= self.nb_nodes:
            return None

        return (n - 1) // 2

    def is_ancestor(self, n1: int, n2: int):
        """Whether n1 is an ancestor of n2"""
        return self.find_LCA(n1, n2) == n1

    def predict(self, z: np.array):
        """Returns leaf assignments, as if each leaf corresponded to a class.

        Args:
            z (np.array): node value vectors of size [N, self.nb_nodes]
        """
        labels = np.argmax(z[:, self.leaves], 1)

        return labels

    def to_adj_matrix(self, depth: Optional[int] = None):
        """Returns node adjecency matrix.

        Args:
            depth (int): optional maximal depth to rectrict adjacencies to top levels
        """
        adj_matrix = np.zeros((self.nb_nodes, self.nb_nodes))
        adj_matrix[self.split_nodes, self.desc_left] = 1
        adj_matrix[self.split_nodes, self.desc_right] = 1

        if depth is None:
            return adj_matrix
        else:
            return adj_matrix[: 2 ** (depth + 1) - 1, : 2 ** (depth + 1) - 1]

    def normalize(self, z: np.array, depth: Optional[int] = None):
        """Normalize node values at same depth

        Args:
            z (np.array): node value vector of size [N, self.nb_nodes]
            depth (int): optional maximal depth up to which normalization is performed
        """
        if depth is None:
            depth = self.depth

        for d in range(depth + 1):
            i = 2**d - 1
            z[i : i + 2**d] /= sum(z[i : i + 2**d])

        return np.nan_to_num(z)

    def find_LCA(self, n1: int, n2: int):
        """Returns Lowest Common Ancestor between two nodes of the tree"""

        if n1 not in self.nodes or n2 not in self.nodes:
            raise ValueError(f"Cannot find LCA. One of {n1} or {n2} not in the tree.")

        while n1 != n2:
            n1, n2 = min(n1, n2), max(n1, n2)

            n2 = self.get_parent(n2)

        return n1

    def get_nodes_level(self, z: np.array, depth: int = 0):
        """Returns assignments at level <depth>

        Args:
            z (np.array): node value vector of size [N, self.nb_nodes]
            depth (int): level at which points are assigned
        """
        leaves = np.argmax(z[:, self.leaves], 1) + self.nb_split

        res = leaves
        for i in range(self.depth - depth):
            res = (res - 1) // 2

        return res

    def get_node_ancestors(self, n: int):
        """Returns list of ancestor indices of a node (including itself)"""

        if n not in self.nodes:
            raise ValueError(f"Cannot find LCA. Node {n} is not in the tree.")

        ancs = [n]

        while n > 0:
            n = self.get_parent(n)
            ancs.append(n)

        return ancs

    def get_node_depth(self, n: int) -> int:
        """Returns the depth of a given node n."""

        if n not in self.nodes:
            raise ValueError(f"Cannot find LCA. Node {n} is not in the tree.")

        depth = 0
        while n != 0:
            n = self.get_parent(n)
            depth += 1
        return depth


def super_interval(
    target: tuple[int, int], candidates: Iterable[tuple[int, int]], strict: bool = True
) -> Optional[int]:
    """Find among candidates the largest interval that contains a target interval."""
    target_left, target_right = target
    assert target_left <= target_right
    best_index = None
    best_length = -1
    second_best_length = -1
    for index, (left, right) in enumerate(candidates):
        assert left <= right
        if left <= target_left and right >= target_right:
            length = right - left
            if length >= best_length:
                best_index = index
                second_best_length = best_length
                best_length = length
    if strict and best_index is not None:
        assert best_length > second_best_length
    return best_index

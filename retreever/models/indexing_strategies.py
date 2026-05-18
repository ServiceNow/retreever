import os
import json
from abc import ABC, abstractmethod
from annoy import AnnoyIndex
import torch
from collections import OrderedDict
import numpy as np 

import faiss


class IndexingStrategy(ABC):
    """
    Abstract base class for indexing strategies used in retreever.

    Args:
        num_dimensions: (e.g., number of leaves of the tree).
    """

    def __init__(self, num_dimensions, *args):
        self.num_dimensions = num_dimensions

    @abstractmethod
    def index_ctxs(self, context_assignments, context_names, **kwargs):
        """
        Index context embeddings.

        Args:
            context_assignments (torch.Tensor): Context embeddings routed through the tree.
            context_names (torch.Tensor): Identifiers for the contexts.
            **kwargs: Additional parameters for specific indexing strategies.
        """
        pass

    @abstractmethod
    def top_contexts(self, question_assignments, k):
        """
        Retrieve top-k contexts for given query.

        Args:
            question_assignments (torch.Tensor): Query embeddings routed through the tree.
            k (int): Number of top contexts to retrieve.

        Returns:
            list: List of top-k context identifiers and scores per query.
        """
        pass

    @abstractmethod
    def reset_index(self):
        """
        Reset the index, clearing all stored data.
        """
        pass

    @abstractmethod
    def build_index(self):
        """
        Build the index for retrieval, if necessary.
        """
        pass

    @abstractmethod
    def save_index(self, save_path: str = "./tree_index.json"):
        """
        Saves the index as a JSON file at :save_path.

        Args:
            save_path(str): Full path to the JSON file.
        """
        pass

    @abstractmethod
    def size(self):
        """
        Gives size of the index - How many contexts are stored in the index

        Returns:
            int: Number of items stored in the index
        """
        pass

    def is_empty(self):
        """
        Checks if the index is empty or not.

        Returns:
            bool: True if empty index, False otherwise
        """
        return self.size() == 0


class GreedyIndexing(IndexingStrategy):
    """
    Indexing strategy using greedy assignment of contexts to leaves.
    """

    def __init__(self, num_dimensions, *args):
        super().__init__(num_dimensions)
        self.index = None

    def index_ctxs(self, context_assignments, context_names, threshold=0.1, **kwargs):
        # Leaf assignment mask
        c_mask = torch.where(context_assignments > threshold, 1, 0).bool()

        for leaf_id, leaf_ctx_names_scores in self.index.items():
            ctx_to_insert_names = context_names[c_mask[:, leaf_id]]
            ctx_to_insert_scores = context_assignments[:, leaf_id][c_mask[:, leaf_id]]

            # Sort by score
            ctx_to_insert_sorted_scores, ctx_to_insert_sorted_idx = torch.sort(
                ctx_to_insert_scores, descending=True
            )
            ctx_to_insert_sorted_names = ctx_to_insert_names[ctx_to_insert_sorted_idx]

            # Merge with existing data, fails if no context has been assigned to the leaf yet
            try:
                all_leaf_scores = torch.hstack(
                    (leaf_ctx_names_scores[1], ctx_to_insert_sorted_scores)
                )
                sorted_leaf_scores, sorted_leaf_idx = torch.sort(all_leaf_scores, descending=True)
                sorted_leaf_names = torch.hstack(
                    (leaf_ctx_names_scores[0], ctx_to_insert_sorted_names)
                )[sorted_leaf_idx]

                # tuple with the name and the score of each assigned context
                self.index[leaf_id] = (sorted_leaf_names, sorted_leaf_scores)

            except TypeError:
                self.index[leaf_id] = (ctx_to_insert_sorted_names, ctx_to_insert_sorted_scores)

    def top_contexts(self, question_assignments, k):
        num_questions = question_assignments.size(0)

        # dicts of returned context names per question and their scores
        topk = [OrderedDict() for _ in range(num_questions)]

        # sort leaf assignments by score and get leaf ids
        sorted_leaves = torch.argsort(question_assignments, dim=1, descending=True)

        # TODO: avoid double loop (with a tensor index and a tensor topk)

        # Loop over queries
        for q_idx in range(num_questions):
            # loop over leaves in descending order of score for current query
            for leaf_id in sorted_leaves[q_idx].tolist():
                # return as many contexts as possible from the current leaf
                num_ctxs_to_retrieve = min(k - len(topk[q_idx]), len(self.index[leaf_id][0]))

                for c in range(num_ctxs_to_retrieve):
                    # upper bound ctx score by query score for current leaf
                    c_score = min(question_assignments[q_idx, leaf_id], self.index[leaf_id][1][c])

                    # setdefault do not add contexts that are already retrieved
                    topk[q_idx].setdefault(int(self.index[leaf_id][0][c]), float(c_score))

                if len(topk[q_idx]) == k:
                    break

        return topk

    def reset_index(self):
        self.index = {leaf: None for leaf in range(self.num_dimensions)}

    def build_index(self):
        pass

    def save_index(self, save_path: str = "./tree_index.json"):
        # Ensure the directory exists
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        print(f"Saving the tree index to {save_path}")
        # JSON only supports str keys; map int leaf -> str leaf for serialization.
        serializable = {
            str(leaf): (
                [[name, float(score)] for name, score in entries]
                if entries is not None
                else None
            )
            for leaf, entries in self.index.items()
        }
        with open(save_path, "w") as file:
            json.dump(serializable, file)

    def size(self):
        # Gather all unique context names stored in the index in a set
        stored_ctxts = set()
        # Iterate over the index
        for leaf_ctx_names_scores in self.index.values():
            # Its possible that no contexts are assigned to this leaf.
            if leaf_ctx_names_scores is None:
                continue
            # Gather the context names assigned to this leaf, and convert these names to a list
            ctxt_names = leaf_ctx_names_scores[0].detach().cpu().tolist()
            # If there are non zero contexts assigned to this leaf, add their names to the ctxt set
            if len(ctxt_names) > 0:
                stored_ctxts.update(ctxt_names)
        # Return the size of the set consisting of context names stored in the index.
        return len(stored_ctxts)


class TreeRepAnnoyIndexing(IndexingStrategy):
    """
    Indexing strategy using the tree-based representations with Annoy (Approximate Nearest Neighbors Library).
    """

    def __init__(self, num_dimensions, distance="manhattan"):
        super().__init__(num_dimensions)
        self.distance = distance
        self.index = AnnoyIndex(self.num_dimensions, distance)
        self.index_ready = False

    def index_ctxs(self, context_assignments, context_names, **kwargs):
        for i in range(context_names.shape[0]):
            self.index.add_item(
                context_names[i].detach().cpu().numpy(),
                context_assignments[i].detach().cpu().numpy(),
            )

        # Since we added new contexts, index is stale
        self.index_ready = False

    def top_contexts(self, question_assignments, k):
        # Need to call build index if index is not ready
        if not self.index_ready:
            self.build_index()
            self.index_ready = True

        num_questions = question_assignments.size(0)
        topk = []

        for q_idx in range(num_questions):
            neighbors, distances = self.index.get_nns_by_vector(
                question_assignments[q_idx].detach().cpu().numpy(), k, include_distances=True
            )
            topk.append(OrderedDict(zip(neighbors, distances)))
        return topk

    def reset_index(self):
        self.index = AnnoyIndex(self.num_dimensions, self.distance)
        self.index_ready = False

    def build_index(self):
        # Argument is number of trees in the index - which is fixed for now to 10.
        # TODO: Ablate over this to see whats the right value.
        self.index.build(10)

    def size(self):
        return self.index.get_n_items()
    
    def save_index(self, save_path: str = "./tree_index.pckl"):
        pass


class TreeRepFaissIndexing(IndexingStrategy):
    """
    Indexing strategy using tree-based representations with FAISS.
    """

    def __init__(self, num_dimensions, distance="manhattan", to_device="cpu"):
        """
        Args:
            num_dimensions (int): Number of dimensions for the embeddings.
            to_device (str): Device for embeddings ('cpu' or 'cuda').
        """

        self.num_dimensions = num_dimensions
        self.device = to_device

        # Create FAISS index based on the distance metric
        # https://github.com/facebookresearch/faiss/issues/3458
        if distance == "manhattan":
            self.distance = faiss.METRIC_L1
        elif distance == "angular":
            self.distance = faiss.METRIC_INNER_PRODUCT
        else:
            raise NotImplementedError("Unknown distance")

        self.index = faiss.index_factory(num_dimensions, "Flat", self.distance)
        self.index = faiss.IndexIDMap(self.index)  # Add ID mapping for context identifiers
        self.index_ready = False
        if self.device == "cuda":
            self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)
        
        print(f"Created Tree Rep based Faiss Index to store tensors of dim {num_dimensions}")

    def index_ctxs(self, context_assignments, context_names, **kwargs):
        """
        Index context embeddings into the FAISS index.

        Args:
            context_assignments (torch.Tensor): Tensor of shape (num_contexts, num_dimensions).
            context_names (torch.Tensor): Tensor of unique identifiers for each context.
        """
        context_assignments_np = context_assignments.detach().cpu().numpy()
        context_names_np = context_names.detach().cpu().numpy()

        self.index.add_with_ids(
            context_assignments_np.copy().astype(np.float32), 
            context_names_np.copy().astype(np.int64))
        self.index_ready = True

    def top_contexts(self, question_assignments, k):
        """
        Retrieve the top-k closest contexts for each question embedding.

        Args:
            question_assignments (torch.Tensor): Tensor of shape (num_questions, num_dimensions).
            k (int): Number of nearest neighbors to retrieve.

        Returns:
            List[OrderedDict]: A list where each entry contains an OrderedDict of top-k context IDs and distances.
        """
        assert self.index_ready, "Index is not built yet. Call index_ctxs first."

        question_assignments_np = question_assignments.detach().cpu().numpy()
        
        # Ensure C-contiguous and correct data type
        question_assignments_np = question_assignments_np.copy().astype(np.float32)

        # Perform the search
        distances, neighbors = self.index.search(question_assignments_np, k)

        topk = [
            OrderedDict(zip(neighbors[q_idx], distances[q_idx]))
            for q_idx in range(question_assignments_np.shape[0])
        ]

        return topk

    def reset_index(self):
        """
        Reset the FAISS index, clearing all stored embeddings.
        """
        self.index = faiss.index_factory(self.num_dimensions, "Flat", self.distance)
        self.index = faiss.IndexIDMap(self.index)
        if self.device == "cuda":
            self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)
        self.index_ready = False

    def build_index(self):
        pass

    def size(self):
        return self.index.ntotal
    
    def save_index(self, save_path: str = "./tree_index.json"):
        pass


import heapq


class TreeRepMultiIndexFaissIndexing:
    """
    Multi-index strategy: Creates separate FAISS indices per node at a tree level.
    
    - Level 0: 1 index (acts like standard TreeRepFaissIndexing)
    - Level 3: 8 indices (one per node at level 3)
    - Level 10: 1024 indices (one per node at level 10)
    
    At indexing: Each doc assigned to top-L nodes, added to those L indices.
    At query: Query selects top-M nodes, searches those M indices, merges results.
    """

    def __init__(
        self, 
        num_dimensions, 
        distance="manhattan", 
        to_device="cpu",
        index_embeddings=False,
        multi_level_config=None,
    ):
        """
        Args:
            num_dimensions (int): Dimension of embeddings/leaf assignments
            distance (str): 'manhattan' or 'angular'
            to_device (str): 'cpu' or 'cuda'
            enable_multi_level (bool): If True, create multiple indices per node
            index_embeddings (bool): If True, index base embeddings; else leaf assignments
            multi_level_config (dict): {'level': int, 'docs_per_node': int}
        """
        self.num_dimensions = num_dimensions
        self.device = to_device
        self.index_embeddings = index_embeddings

        if distance == "manhattan":
            self.distance = faiss.METRIC_L1
        elif distance == "angular":
            self.distance = faiss.METRIC_INNER_PRODUCT
        else:
            raise NotImplementedError("Unknown distance")

        if multi_level_config is None:
            raise ValueError("multi_level_config required")
        
        self.multi_level = multi_level_config.get('level', 3)
        self.nodes_per_doc = multi_level_config.get('nodes_per_doc', 1)
        self.num_indices = 2 ** self.multi_level
        
        # Create separate FAISS index for each node at this level
        self.indices = []
        for i in range(self.num_indices):
            idx = faiss.index_factory(num_dimensions, "Flat", self.distance)
            idx = faiss.IndexIDMap(idx)
            # if self.device == "cuda":
            #     idx = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, idx)
            self.indices.append(idx)
        
        # Track which indices are ready and their sizes
        self.indices_ready = [False] * self.num_indices
        self.index_doc_counts = [0] * self.num_indices
        
        print(f"Multi-level: created {self.num_indices} FAISS indices at level {self.multi_level}")
        print(f"Each doc assigned to top-{self.nodes_per_doc} node(s)")
        
        index_mode = "embeddings" if index_embeddings else "leaf assignments"
        print(f"Indexing: {index_mode}")

    def index_ctxs(self, context_assignments, context_names, full_tree_assignments=None, context_embeddings=None, **kwargs):
        """
        Index contexts into appropriate FAISS indices.
        
        Args:
            context_assignments (torch.Tensor): Leaf assignments (num_contexts, num_leaves)
            context_names (torch.Tensor): Context UIDs
            full_tree_assignments (torch.Tensor): Full tree (num_contexts, total_nodes)
            context_embeddings (torch.Tensor): Base embeddings (num_contexts, dim)
        """
        # Determine what to index
        if self.index_embeddings:
            if context_embeddings is None:
                raise ValueError("context_embeddings required when index_embeddings=True")
            to_index = context_embeddings.detach().cpu().numpy()
            if to_index.ndim == 3:
                to_index = to_index[:, 0, :]
        else:
            to_index = context_assignments.detach().cpu().numpy()
        
        to_index = to_index.astype(np.float32)
        context_names_np = context_names.detach().cpu().numpy().astype(np.int64)
        

        # Multi-index: route each doc to top-L nodes
        if full_tree_assignments is None:
            raise ValueError("full_tree_assignments required for multi-level indexing")
        
        level_start = (2 ** self.multi_level) - 1
        level_end = (2 ** (self.multi_level + 1)) - 1
        
        # Get node probabilities at this level
        node_probs = full_tree_assignments[:, level_start:level_end].cpu().numpy()
        
        # For each doc, find top-L nodes
        for doc_idx, (doc_vec, doc_uid) in enumerate(zip(to_index, context_names_np)):
            doc_node_probs = node_probs[doc_idx]
            
            # Get top-L nodes
            top_nodes = np.argsort(doc_node_probs)[::-1][:self.nodes_per_doc]
            
            # Add this doc to each of the top-L indices
            for node_idx in top_nodes:
                self.indices[node_idx].add_with_ids(
                    doc_vec.reshape(1, -1), 
                    np.array([doc_uid])
                )
                self.indices_ready[node_idx] = True
                self.index_doc_counts[node_idx] += 1

    def top_contexts(self, question_assignments, k, num_nodes=None, question_embeddings=None, question_tree_assignments=None, **kwargs):
        """
        Retrieve top-k contexts.
        
        Args:
            question_assignments (torch.Tensor): Leaf assignments
            k (int): Number of contexts to retrieve
            num_nodes (int): For multi-level, number of indices (M) to search
            question_embeddings (torch.Tensor): Base embeddings
            question_tree_assignments (torch.Tensor): Full tree assignments
        
        Returns:
            List[OrderedDict]: Top-k contexts per query
        """

        # Multi-index retrieval with M nodes
        if num_nodes is None:
            raise ValueError("num_nodes required for multi-level retrieval")
        if question_tree_assignments is None:
            raise ValueError("question_tree_assignments required for multi-level")
        
        return self._top_contexts_multi_level(
            question_assignments, 
            question_tree_assignments, 
            k, 
            num_nodes,
            question_embeddings
        )

    def _top_contexts_multi_level(self, question_assignments, question_tree_assignments, k, num_nodes, question_embeddings=None):
        """
        Multi-index retrieval: search top-M indices and merge results.
        """
        level_start = (2 ** self.multi_level) - 1
        level_end = (2 ** (self.multi_level + 1)) - 1
        
        # Determine what to search with
        if self.index_embeddings:
            if question_embeddings is None:
                raise ValueError("question_embeddings required")
            to_search = question_embeddings.detach().cpu().numpy()
            if to_search.ndim == 3:
                to_search = to_search[:, 0, :]
        else:
            to_search = question_assignments.detach().cpu().numpy()
        
        to_search = to_search.astype(np.float32)
        
        # Get query node scores
        query_node_probs = question_tree_assignments[:, level_start:level_end].cpu().numpy()
        
        num_queries = to_search.shape[0]
        results = []
        
        for q_idx in range(num_queries):
            query_vec = to_search[q_idx:q_idx+1]
            query_probs = query_node_probs[q_idx]
            
            # Select top-M nodes by query score
            top_m = min(num_nodes, self.num_indices)
            top_node_indices = np.argsort(query_probs)[::-1][:top_m]
            
            # Running max-heap and seen set to avoid duplicates
            max_heap = []
            seen_docs = set()
            
            # Search each of the M selected indices
            for node_idx in top_node_indices:
                if not self.indices_ready[node_idx]:
                    continue
                
                index = self.indices[node_idx]
                k_to_search = min(k, index.ntotal)
                
                if k_to_search == 0:
                    continue
                
                distances, neighbors = index.search(query_vec, k_to_search)
                
                # Process each candidate
                for doc_id, dist in zip(neighbors[0], distances[0]):
                    if doc_id < 0:  # Skip FAISS sentinels
                        continue
                    
                    if doc_id in seen_docs:  # Already processed this doc
                        continue
                    
                    seen_docs.add(doc_id)
                    
                    # Add to heap (using negative distance for max-heap behavior)
                    if len(max_heap) < k:
                        heapq.heappush(max_heap, (-dist, doc_id))
                    elif dist < -max_heap[0][0]:  # Better than worst in heap
                        heapq.heappushpop(max_heap, (-dist, doc_id))
            
            # Extract top-k in ascending order
            top_k = [(doc_id, -neg_dist) for neg_dist, doc_id in sorted(max_heap)]
            results.append(OrderedDict(top_k))
        
        return results

    def reset_index(self):
        """Reset all indices."""
        self.indices = []
        for i in range(self.num_indices):
            idx = faiss.index_factory(self.num_dimensions, "Flat", self.distance)
            idx = faiss.IndexIDMap(idx)
            # if self.device == "cuda":
            #     idx = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, idx)
            self.indices.append(idx)
        
        self.indices_ready = [False] * self.num_indices
        self.index_doc_counts = [0] * self.num_indices

    def is_empty(self):
        return not any(self.indices_ready)

    def build_index(self):
        pass

    def size(self):
            return sum(idx.ntotal for idx in self.indices)

index_strategy_dict = {
    "greedy": GreedyIndexing,
    "tree_rep": TreeRepAnnoyIndexing,
    "faiss_tree_rep": TreeRepFaissIndexing,
    "tree_rep_multi_index_faiss": TreeRepMultiIndexFaissIndexing,
}

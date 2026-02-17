import numpy as np
import torch
import torch.nn as nn

from abc import ABC, abstractmethod
from typing import Tuple
import math
import torch.nn.functional as F

import torch.nn.init as init

import os
import json 
from transformers import AutoModelForCausalLM
import deepspeed
from retreever.models.node_embedding_strategies import get_node_embedding_strategy



class Split(ABC, torch.nn.Module):
    def __init__(self):
        """Abstract class for split functions."""
        super(Split, self).__init__()

        self.split = torch.nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        x = x.to(next(self.split.parameters()).dtype)
        return self.split(x)

    @abstractmethod
    def _init_params(self, start_idx: int = 0):
        """Initialize parameters of split functions from <start_idx> on.

        Args:
            start_idx (int, optional): From which split function index to start the initialization. Defaults to 0.
        """
        pass


class LinearSplit(Split):
    def __init__(self, in_size: Tuple[int], nb_splits: int, *args, **kwargs):
        """Linear split functions.

        Args:
            in_size (tuple): expected size of the input
            nb_splits (int): number of split nodes (or output dimension)
        """
        self.in_size = np.prod(in_size)

        self.layers = [torch.nn.Flatten(), torch.nn.Linear(self.in_size, nb_splits, bias=False)]

        super(LinearSplit, self).__init__()

    def _init_params(self, start_idx: int = 0):
        torch.nn.init.zeros_(self.layers[1].weight[start_idx:])


class MLPSplit(Split):
    def __init__(
        self,
        in_size: Tuple[int],
        nb_out: int,
        hidden_size: int = 128,
        p: float = 0.2,
        *args,
        **kwargs,
    ):
        """MLP-based split functions with ReLU activations.
        Args:
            in_size (tuple): Expected size of the input (excluding batch size).
            nb_out (int): Number of output dimension.
            hidden_size (int, optional): Number of hidden units in the MLP. Defaults to 128.
            p (float, optional): Dropout param p, Defaults to 0.2
        """
        self.in_size = np.prod(in_size)
        self.p = p

        self.layers = [
            torch.nn.Flatten(),  # Flatten input to (batch_size, in_size)
            torch.nn.Linear(self.in_size, hidden_size),
            torch.nn.Dropout(p=self.p),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, nb_out),
        ]

        super(MLPSplit, self).__init__()

    def _init_params(self, start_idx: int = 0):
        """Initialize parameters of the split function."""
        # Initialization based on the first batch of data is not supported
        return


class CrossAttentionSplit(torch.nn.Module):
    """
    This is a split function designed to be working with token-level encodings of queries/documents.
    This learns multiple node embeddings per split node and utilizes cross attention between node
    embeddings and document tokens to learn meaningful node representations.
    """

    def __init__(
        self,
        in_size: int,
        nb_splits: int,
        embedding_dim: int,
        scoring_fn_name: str,
        num_embeddings_per_node: int = 2,
        d_k: int = 128,
        n_heads: int = 2,
        node_embedding_strategy: str = "independent",
        node_embedding_rank: int = 32,  # For low-rank strategies
        tree_depth: int = 10,  # For level-based strategies
        treat_attn_as_residual: bool = False,
        have_ffn_layer: bool = False,
        vq_residual: bool = False,
        use_simvq: bool = False,
        dropout_location: str = "inside_split",
    ):
        """
        Initializes the CrossAttentionSplit module.

        Args:
            in_size (int): Size of the input feature vector - represents dim of encoder.
            nb_splits (int): Number of split nodes in the tree. This denotes the out dim here.
            embedding_dim (int): Dimension of the node embeddings.
            num_embeddings_per_node(int): Number of node embeddings to learn for each split node. Defaults to 10.
            d_k (int, optional): Dimension of the projected query and key vectors. Defaults to 128.
            n_heads (int): Number of attention heads. Defaults to 8.
        """
        super(CrossAttentionSplit, self).__init__()
        self.d_k = d_k
        self.d_emb = embedding_dim
        self.d_enc = in_size
        self.n_e = num_embeddings_per_node
        self.n_t = nb_splits
        self.n_heads = n_heads
        self.head_dim = d_k // n_heads
        self.tree_depth = tree_depth

        self.scoring_fn_name = scoring_fn_name


        assert d_k % n_heads == 0, "d_k must be divisible by n_heads"
            

        # Projections for multi-head attention
        # Queries come from node embeddings. Its possible to avoid this projection if we ensure d_emb == d_k
        self.query_proj = nn.Linear(self.d_emb, d_k)

        # Keys and Values are derived from input context/query
        self.key_proj = nn.Linear(self.d_enc, d_k)
        self.value_proj = nn.Linear(self.d_enc, d_k)

        # Output projection
        self.output_proj = self.get_output_projection()

        # Scaling factor for attention scores
        self.scale = math.sqrt(self.head_dim)

        # ===== Node Embedding Strategy (NEW) =====
        self.node_embedding_module = get_node_embedding_strategy(
            strategy=node_embedding_strategy,
            n_nodes=self.n_t,
            n_emb_per_node=self.n_e,
            d_emb=self.d_emb,
            tree_depth=tree_depth,
            rank=node_embedding_rank,
            is_residual=vq_residual,
            use_simvq=use_simvq,
        )
        
        self.use_level_only = (node_embedding_strategy == "level_only")
        self.use_vq = (node_embedding_strategy == "vector_quantized")
        
        if self.use_level_only or self.use_vq:
            # Precompute node-to-level mapping
            node_levels = torch.zeros(self.n_t, dtype=torch.long)
            for i in range(self.n_t):
                node_levels[i] = int(math.floor(math.log2(i + 1)))
            self.register_buffer("node_levels", node_levels)
        
        # Log parameter count for transparency
        num_params = self.node_embedding_module.get_num_parameters()
        print(f"Node embedding strategy '{node_embedding_strategy}': {num_params:,} parameters")
        
        # Node embeddings for inner nodes (trainable parameters)
        # self.node_embeddings = nn.Parameter(torch.randn(self.n_t, self.n_e, self.d_emb))
        
        #### Adding this 
        self.attn_drpout = nn.Dropout(p=0.1)
        self.dropout_location = dropout_location
        
        self.treat_attn_as_residual = treat_attn_as_residual
        self.have_ffn_layer = have_ffn_layer
        
        wt_matrix_dim = self.d_k
        if treat_attn_as_residual:
            self.out_proj = nn.Linear(d_k, self.d_emb)
            wt_matrix_dim = self.d_emb
            self.norm1 = nn.LayerNorm(self.d_emb)
            
            if have_ffn_layer:
                self.norm2 = nn.LayerNorm(self.d_emb)
        
                # FFN sub-layer
                d_ff = self.d_emb * 4
                self.ffn = nn.Sequential(
                    nn.Linear(self.d_emb, d_ff),
                    nn.GELU(),
                    nn.Dropout(p=0.1),
                    nn.Linear(d_ff, self.d_emb),
                    nn.Dropout(p=0.1)
                )

        # Mimic a separate Linear mapping defined for each tree node separately.
        if self.scoring_fn_name in ["linear_per_nt_then_mean",
                                    "product_propagation"]:
            self.weight_matrix = nn.Parameter(torch.empty(self.n_t, wt_matrix_dim))  # (Nt, Dk)
            self.bias_matrix = nn.Parameter(torch.empty(self.n_t))  # (Nt)

            # Apply the default initialization used in nn.Linear
            init.kaiming_uniform_(self.weight_matrix, a=math.sqrt(5))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_matrix)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias_matrix, -bound, bound)
            
            if self.scoring_fn_name == "tree_based":
                tree_mat = self._get_tree_matrix(nb_splits)
                self.register_buffer("tree_matrix", tree_mat)
            
            if self.scoring_fn_name in ["propagate_sep_mlp"]:
                # Create a 2-layer MLP for node score propagation through ancestor scores only.
                self.levels = int(math.log2(self.n_t + 1))
                self.agg_linear1_wt = nn.Parameter(torch.empty(self.n_t, self.levels, self.levels//2))  # [n_t, levels, levels//2]
                self.agg_linear1_bias = nn.Parameter(torch.empty(self.n_t, self.levels//2)) # [n_t, levels]

                # Second layer weights and biases for each node
                self.agg_linear2_wt = nn.Parameter(torch.empty(self.n_t, self.levels//2, 1)) # [n_t, 1, hidden_dim]
                self.agg_linear2_bias = nn.Parameter(torch.empty(self.n_t)) # [n_t, 1]

                # Dropout
                self.dropout = nn.Dropout(0.1)

                for param in [self.agg_linear1_wt, self.agg_linear2_wt]:
                    nn.init.kaiming_uniform_(param, a=math.sqrt(5))

                for param in [self.agg_linear1_bias, self.agg_linear2_bias]:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.agg_linear1_wt)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(param, -bound, bound)

                anc_idx, anc_msk = self._compute_ancestor_indices()
                self.register_buffer("anc_idx", anc_idx)
                self.register_buffer("anc_msk", anc_msk) 
                
                
            if self.scoring_fn_name == "propagate_single_mlp":
                self.levels = int(math.log2(self.n_t + 1))
                self.hidden_dim = max(1, self.levels // 2)
                self.shared_mlp = nn.Sequential(
                    nn.Linear(self.levels, self.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(self.hidden_dim, 1)
                )    
                anc_idx, anc_msk = self._compute_ancestor_indices()
                self.register_buffer("anc_idx", anc_idx)
                self.register_buffer("anc_msk", anc_msk) 
              
        self.scoring_functions = {
            "mean_then_linear": self._mean_then_linear,
            "linear_then_mean": self._linear_then_mean,
            "flatten_then_linear": self._flatten_then_linear_shared,
            "flatten_then_linear_unshared": self._flatten_then_linear_unshared,
            "linear_per_nt_then_mean": self._linear_per_nt_then_mean,
            "propagate_sep_mlp": self._propagate_sep_mlp,
            "propagate_single_mlp": self._propagate_single_mlp,
            "product_propagation": self._product_propagation,
        }

        self._init_params()
        
    def _get_tree_matrix(self, num_nodes):
        """
        Returns a mask to use with tree based scoring. This mask indicates which nodes
        are ancestors in the tree structure to a given node. 
        """
        tree_matrix = torch.zeros(num_nodes, num_nodes)
        for i in range(num_nodes):
            tree_matrix[i, i] = 1

            if i == 0:
                continue
            parent = (i - 1) // 2
            tree_matrix[i] += tree_matrix[parent]
        return tree_matrix
    
    def _compute_ancestor_indices(self):
        """
        Builds two tensors: anc_idx and anc_msk of shape [n_t, max_len], where:
        - anc_idx[i, 0] = i
        - anc_idx[i, 1] = parent(i)
        - anc_idx[i, 2] = parent(parent(i))
        - ...
        - remaining positions up to max_len are filled with -1 if no more ancestors
        - anc_msk[i, j] = 1 if anc_idx[i, j] is a valid ancestor, else 0
        """
        anc_idx = torch.zeros((self.n_t, self.levels), dtype=torch.long)
        anc_msk = torch.zeros((self.n_t, self.levels), dtype=torch.float)

        # Root node is node 0 (no parent)
        anc_idx[0, 0] = 0
        anc_msk[0, 0] = 1.0

        for i in range(1, self.n_t):
            p = (i - 1) // 2  # parent in a binary heap / binary tree

            anc_idx[i, 1:] = anc_idx[p, :-1]
            anc_msk[i, 1:] = anc_msk[p, :-1]

            # 2) Put the child itself in position 0
            anc_idx[i, 0] = i
            anc_msk[i, 0] = 1.0

        return anc_idx, anc_msk
    
    def forward(self, tokens: torch.Tensor, mask: torch.Tensor):
        """
        Forward pass this module.

        Args:
            tokens (torch.Tensor): Input tokens of shape [B, n_d, d_enc].
            mask (torch.Tensor): Mask for meaningful tokens of shape [B, n_d], with 1 for valid tokens and 0 for padding.

        Returns:
            torch.Tensor: Output tensor of shape [B, n_t].
        """
        if (len(tokens.shape)) == 2:
            tokens = tokens.unsqueeze(1)
        B, n_d, _ = tokens.shape
        
        if mask is None:
            # Create default all-ones mask
            mask = torch.ones(B, n_d, device=tokens.device)
            
        mask = mask[:, :n_d]


        # ===== Get node embeddings (NEW) =====
        # node_embeddings = self.node_embedding_module()  # [n_t, n_e, d_emb]
        vq_losses = {}
        if self.use_vq:
            # Extract CLS tokens: [B, d_enc]
            cls_tokens = tokens[:, 0, :]  # [B, d_enc]
            # Get VQ embeddings: [B, tree_depth+1, d_emb]
            node_embeddings, vq_losses = self.node_embedding_module(cls_tokens)
            # Expand n_e dimension: [B, tree_depth+1, 1, d_emb]
            node_embeddings = node_embeddings.unsqueeze(2)
            self.extra_loss = vq_losses["total_quantization_loss"]
        else:
            node_embeddings = self.node_embedding_module()
                
        # if self.treat_attn_as_residual:
        #     node_embeddings = self.norm1(node_embeddings)
    
        # Project keys, values, and queries
        keys = self.key_proj(tokens).view(
            B, n_d, self.n_heads, self.head_dim
        )  # [B, n_d, n_heads, head_dim]
        values = self.value_proj(tokens).view(
            B, n_d, self.n_heads, self.head_dim
        )  # [B, n_d, n_heads, head_dim]
        # queries = self.query_proj(node_embeddings).view(
        #     self.n_t, self.n_e, self.n_heads, self.head_dim
        # )  # [n_t, n_e, n_heads, head_dim]
        
        if self.use_level_only:
            # embeddings is [n_levels, n_e, d_emb]
            n_levels = node_embeddings.shape[0]
            queries = self.query_proj(node_embeddings).view(
                n_levels, self.n_e, self.n_heads, self.head_dim
            )  # [n_levels, n_e, n_heads, head_dim]
        elif self.use_vq:
            # embeddings is [B, n_levels, n_e, d_emb]
            # Keep batch dimension in queries!
            n_levels = node_embeddings.shape[1]  # tree_depth+1
            queries = self.query_proj(node_embeddings).view(
                B, n_levels, self.n_e, self.n_heads, self.head_dim
            )  # [B, n_levels, n_e, n_heads, head_dim]
        else:
            # Standard path: embeddings is [n_t, n_e, d_emb]
            if self.treat_attn_as_residual:
                node_embeddings = self.norm1(node_embeddings)
            queries = self.query_proj(node_embeddings).view(
                self.n_t, self.n_e, self.n_heads, self.head_dim
            )  # [n_t, n_e, n_heads, head_dim]

        # Compute attention scores
        if self.use_vq:
            # queries has batch dimension: [B, n_levels, n_e, n_heads, head_dim]
            attention_scores = torch.einsum(
                "bnhd,btmhd->bhtmn", keys, queries
            )  # [B, n_heads, n_levels, n_e, n_d]
        else:
            attention_scores = torch.einsum(
                "bnhd,tmhd->bhtmn", keys, queries
            )  # [B, n_heads, n_t, n_e, n_d]
        attention_scores = attention_scores / self.scale

        # Apply mask
        mask = mask[:, None, None, None]  # [B, 1, 1, 1, n_d]
        attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))

        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # [B, n_heads, n_t, n_e, n_d]

        ####### Adding this
        if self.dropout_location in ["inside_split", "both"]:
            attention_weights = self.attn_drpout(attention_weights)
        
        # Aggregate values
        aggregated_values = torch.einsum(
            "bnhd,bhtmn->bhtmd", values, attention_weights
        )  # [B, n_heads, n_t, n_e, head_dim]

        
        # Expand from level to node if using level-only embeddings
        if self.use_level_only or self.use_vq:
            n_levels = aggregated_values.shape[2]
            aggregated_values = torch.einsum("bhtmd->btmdh", aggregated_values).reshape(
                B, n_levels, self.n_e, self.d_k
            )  # [B, n_t, n_e, d_k]
            # Expand: [B, n_levels, n_e, d_k] -> [B, n_t, n_e, d_k]
            aggregated_values = aggregated_values[:, self.node_levels]  # [B, n_t, n_e, d_k]
        else:
            aggregated_values = torch.einsum("bhtmd->btmdh", aggregated_values).reshape(
                B, self.n_t, self.n_e, self.d_k
            )  # [B, n_t, n_e, d_k]
    
        
        if self.treat_attn_as_residual:
            aggregated_values = node_embeddings + self.attn_drpout(self.out_proj(aggregated_values))
            
            if self.have_ffn_layer:
                residual = aggregated_values
                values_norm = self.norm2(aggregated_values)
                ffn_out = self.ffn(values_norm)
                aggregated_values = residual + ffn_out
                

        # Output projection to scores per embedding
        node_scores, _ = self.get_scores_from_attn_out(aggregated_values)  # [B, n_t]
        return node_scores, {}

    def get_scores_from_attn_out(self, attn_out):
        # Get the appropriate function based on scoring_fn_name, or raise an error if not found
        scoring_function = self.scoring_functions.get(self.scoring_fn_name, None)

        if scoring_function is None:
            raise ValueError(f"Unknown scoring function: {self.scoring_fn_name}")

        return scoring_function(attn_out)

    def get_output_projection(self):
        if self.scoring_fn_name == "flatten_then_linear_shared":
            return nn.Linear(self.n_e * self.d_k, 1)

        if self.scoring_fn_name == "flatten_then_linear_unshared":
            return nn.Linear(self.n_t * self.n_e * self.d_k, self.n_t)

        return nn.Linear(self.d_k, 1)

    def _mean_then_linear(self, attn_out: torch.Tensor):
        attn_out = attn_out.mean(2)  # B, n_t, d_k
        return self.output_proj(attn_out).squeeze(-1)  # B, n_t

    def _linear_then_mean(self, attn_out: torch.Tensor):
        attn_out = self.output_proj(attn_out).squeeze(-1)  # B, n_t, n_e
        return attn_out.mean(-1)

    def _flatten_then_linear_shared(self, attn_out: torch.Tensor):
        B = attn_out.shape[0]
        attn_out = attn_out.reshape(B, self.n_t, -1)
        return self.output_proj(attn_out).squeeze(-1)

    def _flatten_then_linear_unshared(self, attn_out: torch.Tensor):
        B = attn_out.shape[0]
        attn_out = attn_out.reshape(B, -1)
        return self.output_proj(attn_out)

    def _linear_per_nt_then_mean(self, attn_out: torch.Tensor):
        B, Nt, Ne, Dk = attn_out.shape

        if self.weight_matrix is None or self.bias_matrix is None:
            raise RuntimeError(
                "Weight and bias matrices have not been initialized. "
                "Ensure scoring_fn_name='linear_per_nt_then_mean'."
            )

        # Apply weights: (B, Nt, Ne, Dk) x (Nt, Dk) -> (B, Nt, Ne)
        linear_out = torch.einsum("btnd,td->btn", attn_out, self.weight_matrix)

        # Add bias: (Nt) -> (B, Nt, Ne)
        linear_out += self.bias_matrix[None, :, None]

        # Mean pool over Ne dimension: (B, Nt)
        return linear_out.mean(dim=2)

    
    def _product_propagation(self, attn_out):
        scores = self._linear_per_nt_then_mean(attn_out)
        # return scores
        B, nt = scores.shape

        # Step 1: Convert split scores to probabilities using the sigmoid function
        probabilities = torch.sigmoid(scores)
        
        
        nb_nodes = 2 * nt + 1  # Total nodes in a full binary tree
        # left nodes
        desc_left = range(1, nb_nodes, 2)

        # right nodes
        desc_right = range(2, nb_nodes, 2)

        # Step 2: Initialize scores for each node to 1 (since we're using multiplicative probabilities)
        propagated_scores = torch.ones((B, nb_nodes), device=attn_out.device)

        # Step 3: Propagate probabilities down the tree
        for _ in range(self.tree_depth):
            # For left children, multiply with the probability of going left
            propagated_scores[:, desc_left] = (
                propagated_scores[:, range(nt)] * probabilities
            )

            # For right children, multiply with the probability of going right (1 - probability)
            propagated_scores[:, desc_right] = propagated_scores[:, range(nt)] * (1 - probabilities)

        return propagated_scores, scores
    
    def _propagate_sep_mlp(self, attn_out):
        scores = self._linear_per_nt_then_mean(attn_out)
        B, nt = scores.shape

        anc_idx_expanded = self.anc_idx.unsqueeze(0).expand(B, -1, -1)  # [B, n_t, max_len]
        anc_scores = torch.gather(scores.unsqueeze(-1).expand(-1, -1, anc_idx_expanded.shape[-1]), dim=1, index=anc_idx_expanded) # [B, n_t, max_len]

        # Apply mask to zero out invalid ancestors: [B, n_t, max_len]
        anc_scores = anc_scores * self.anc_msk.unsqueeze(0)

        # First linear layer (vectorized): [B, n_t, hidden_dim]
        hidden = torch.einsum('btn,tnh->bth', anc_scores, self.agg_linear1_wt)  # [B, n_t, hidden_dim]
        hidden = hidden + self.agg_linear1_bias.unsqueeze(0)                   # Add bias: [B, n_t, hidden_dim]
        hidden = F.relu(hidden)                                                # Apply ReLU
        hidden = self.dropout(hidden)                                          # Apply Dropout

        # Second linear layer (vectorized): [B, n_t, 1]
        out = torch.einsum('btm,tmn->btn', hidden, self.agg_linear2_wt)        # [B, n_t, 1]
        out = out.squeeze(-1)                                                 # Remove last dimension: [B, n_t]
        out = out + self.agg_linear2_bias.unsqueeze(0)             # Add bias: [B, n_t]

        return out, scores

        
    def _propagate_single_mlp(self, attn_out):
        scores = self._linear_per_nt_then_mean(attn_out)
        B, nt = scores.shape

        anc_idx_expanded = self.anc_idx.unsqueeze(0).expand(B, -1, -1)  # [B, n_t, max_len]
        anc_scores = torch.gather(scores.unsqueeze(-1).expand(-1, -1, anc_idx_expanded.shape[-1]), dim=1, index=anc_idx_expanded) # [B, n_t, max_len]

        # Apply mask to zero out invalid ancestors: [B, n_t, max_len]
        anc_scores = anc_scores * self.anc_msk.unsqueeze(0)
        
        # Flatten the node dimension to process all nodes in one batch.
        anc_scores_flat = anc_scores.view(B * self.n_t, self.levels)  # [B*n_t, levels]

        out_flat = self.shared_mlp(anc_scores_flat).squeeze(-1)  # [B*n_t]
        outputs = out_flat.view(B, self.n_t)

        return outputs, scores


    def _init_params(self):
        """
        Doing a custom initialization of the projector layers better suited for attention.
        """
        for layer in [self.key_proj, self.value_proj, self.query_proj, self.output_proj]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)



split_dict = {  # supported split modulestorch.nn.init.zeros_(self.value_proj.bias)
    "linear": LinearSplit,
    "mlp": MLPSplit,
    "cross_attn": CrossAttentionSplit,
}

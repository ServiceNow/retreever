import numpy as np
import torch
import torch.nn as nn

from abc import ABC, abstractmethod
from typing import Tuple
import math
import torch.nn.functional as F

import torch.nn.init as init

from retreever.utils.toolkit_paths import PATH_HF_CACHE_RW
from retreever.utils.scripting import  get_local_rank_and_world_size

import os
import json
from transformers import AutoModelForCausalLM
import deepspeed
from retreever.models.token_mixers import get_token_mixer
from retreever.models.node_embedding_strategies import get_node_embedding_strategy



# ═══════════════════════════════════════════════════════════════════════
# Tree Adapter: modular adapter for frozen CrossAttentionSplit trees
# ═══════════════════════════════════════════════════════════════════════

class LoRALinear(nn.Module):
    """LoRA adapter wrapping a frozen nn.Linear. Zero-init on B so starts as identity."""
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = nn.Linear(base.in_features, r, bias=False)
        self.B = nn.Linear(r, base.out_features, bias=False)
        self.scaling = alpha / r
        nn.init.kaiming_uniform_(self.A.weight)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.base(x) + self.B(self.A(x)) * self.scaling


class TreeAdapter(nn.Module):
    """Modular adapter for frozen CrossAttentionSplit trees.

    Each component is independently toggleable via adapt_* flags:
      - embeddings:  residual delta on level_embeddings
      - projections: LoRA on query_proj / value_proj
      - scoring:     low-rank residual on weight_matrix + bias delta
      - logits:      scalar / linear / mlp adjustment on pre-sigmoid logits

    All components are zero-initialized so the adapted tree starts identical
    to the frozen pretrained tree.
    """
    def __init__(self, split_fn, adapt_embeddings=False, adapt_projections=False,
                 adapt_scoring=False, adapt_logits=False,
                 lora_r=8, lora_alpha=16, logit_adapter_type="scalar"):
        super().__init__()
        self.adapt_embeddings = adapt_embeddings
        self.adapt_projections = adapt_projections
        self.adapt_scoring = adapt_scoring
        self.adapt_logits = adapt_logits

        # ── Embedding adapter: residual delta ──────────────────────
        if adapt_embeddings:
            if hasattr(split_fn.node_embedding_module, 'level_embeddings'):
                self.emb_delta = nn.Parameter(
                    torch.zeros_like(split_fn.node_embedding_module.level_embeddings.data))
            elif hasattr(split_fn.node_embedding_module, 'node_embeddings'):
                self.emb_delta = nn.Parameter(
                    torch.zeros_like(split_fn.node_embedding_module.node_embeddings.data))

        # ── Projection adapter: LoRA on query_proj / value_proj ───
        if adapt_projections:
            split_fn.query_proj = LoRALinear(split_fn.query_proj, r=lora_r, alpha=lora_alpha)
            split_fn.value_proj = LoRALinear(split_fn.value_proj, r=lora_r, alpha=lora_alpha)

        # ── Scoring adapter: low-rank residual on weight_matrix ───
        if adapt_scoring:
            n_t, d_k = split_fn.weight_matrix.shape
            self.score_A = nn.Parameter(torch.zeros(n_t, lora_r))
            self.score_B = nn.Parameter(torch.zeros(lora_r, d_k))
            self.bias_delta = nn.Parameter(torch.zeros(n_t))

        # ── Logit adapter: adjust pre-sigmoid logits ──────────────
        if adapt_logits:
            self.logit_adapter_type = logit_adapter_type
            n_t = split_fn.n_t
            tree_depth = split_fn.tree_depth
            if logit_adapter_type == "scalar":
                self.logit_temp = nn.Parameter(torch.ones(n_t))
                self.logit_bias = nn.Parameter(torch.zeros(n_t))
            elif logit_adapter_type == "linear":
                self.logit_linears = nn.ModuleList()
                for level in range(tree_depth):
                    n = 2 ** level
                    lin = nn.Linear(n, n)
                    nn.init.zeros_(lin.weight)
                    nn.init.zeros_(lin.bias)
                    self.logit_linears.append(lin)
            elif logit_adapter_type == "mlp":
                self.logit_mlp = nn.Sequential(
                    nn.Linear(n_t, n_t // 4),
                    nn.ReLU(),
                    nn.Linear(n_t // 4, n_t),
                )
                nn.init.zeros_(self.logit_mlp[-1].weight)
                nn.init.zeros_(self.logit_mlp[-1].bias)

        # Log param count
        total = sum(p.numel() for p in self.parameters())
        print(f"TreeAdapter: {total:,} trainable parameters "
              f"(emb={adapt_embeddings}, proj={adapt_projections}, "
              f"score={adapt_scoring}, logits={adapt_logits}"
              f"{f'/{logit_adapter_type}' if adapt_logits else ''})")

    def adjust_embeddings(self, node_embeddings):
        """Add residual delta to node embeddings."""
        return node_embeddings + self.emb_delta

    def adjust_scoring(self, weight_matrix, bias_matrix):
        """Return adjusted weight and bias matrices."""
        W = weight_matrix + self.score_A @ self.score_B
        b = bias_matrix + self.bias_delta
        return W, b

    def adjust_logits(self, scores):
        """Adjust pre-sigmoid logits. scores: [B, n_t]."""
        if self.logit_adapter_type == "scalar":
            return scores * self.logit_temp + self.logit_bias
        elif self.logit_adapter_type == "linear":
            adjusted = scores.clone()
            for level, lin in enumerate(self.logit_linears):
                start = 2**level - 1
                end = 2**(level+1) - 1
                adjusted[:, start:end] = scores[:, start:end] + lin(scores[:, start:end])
            return adjusted
        elif self.logit_adapter_type == "mlp":
            return scores + self.logit_mlp(scores)
        return scores


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


class AttentionSplit(torch.nn.Module):
    def __init__(
        self,
        in_size: Tuple[int],
        nb_splits: int,
        embedding_dim: int,
        d_k: int = 128,
        *args,
        **kwargs,
    ):
        """
        Initializes the AttentionSplit module.

        Args:
            in_size (Tuple[int]): Size of the input feature vector.
            nb_splits (int): Number of split nodes in the tree. This denotes the out dim here.
            embedding_dim (int): Dimension of the node embeddings.
            d_k (int, optional): Dimension of the projected query and key vectors. Defaults to 128.
        """
        super(AttentionSplit, self).__init__()
        self.d_k = d_k

        self.query_proj = torch.nn.Linear(in_size, d_k)
        self.key_proj = torch.nn.Linear(embedding_dim, d_k)
        self.value_proj = torch.nn.Linear(embedding_dim, nb_splits)

        # Scaling factor for attention scores
        self.scale = math.sqrt(d_k)

        # Node embeddings for inner nodes (trainable parameters)
        self.node_embeddings = torch.nn.Parameter(torch.randn(nb_splits, embedding_dim))

        self._init_params()

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        Forward pass for the AttentionSplit module.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_size).
            node_embeddings (torch.Tensor): Node embeddings with shape (nb_splits, embedding_dim).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, nb_splits), representing split node scores.
        """
        Q = self.query_proj(x)  # (batch_size, d_k)
        K = self.key_proj(self.node_embeddings)  # (nb_splits, d_k)
        V = self.value_proj(self.node_embeddings)  # (nb_splits, nb_splits)

        scores = (Q @ K.T) / self.scale  # (batch_size, nb_splits)
        attention_weights = torch.softmax(scores, dim=-1)  # (batch_size, nb_splits)
        output = attention_weights @ V  # (batch_size, nb_splits)

        return output

    def _init_params(self):
        """
        Doing a custom initialization of the projector layers better suited for attention.
        """
        torch.nn.init.xavier_uniform_(self.query_proj.weight)
        torch.nn.init.xavier_uniform_(self.key_proj.weight)
        if self.query_proj.bias is not None:
            torch.nn.init.zeros_(self.query_proj.bias)
        if self.key_proj.bias is not None:
            torch.nn.init.zeros_(self.key_proj.bias)
        if self.value_proj.bias is not None:
            torch.nn.init.zeros_(self.value_proj.bias)


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
        token_mixer_type: str = None,
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
        
        if token_mixer_type is not None:
            self.token_mixer = get_token_mixer(
                mixer_type=token_mixer_type,
                d_model=self.d_enc,
            )
            print(f"Obtained token mixer as {token_mixer_type}")
            print("===== Token Mixer: ======")
            print(self.token_mixer)
        else:
            # Default to identity (no mixing)
            self.token_mixer = get_token_mixer(
                mixer_type="none",
                d_model=self.d_enc
            )
            

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
                                    "tree_based", 
                                    "propagate_sep_mlp", 
                                    "propagate_single_mlp", 
                                    "propagate_transformer",
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
                
            if self.scoring_fn_name == "propagate_transformer":
                self.levels = int(math.log2(self.n_t + 1))
                # We use the mean of the node embeddings per node for simplicity.
                self.node_emb_dim = self.d_emb  # original dimension of node embeddings
                # The input for each ancestor will be the concatenation of its score (scalar) and its node embedding.
                self.transformer_input_dim = self.node_emb_dim + 1
                self.proj_dim = 128  
                assert self.proj_dim % 4 == 0, "proj_dim must be divisible by n_heads"
                self.input_proj = nn.Linear(self.transformer_input_dim, self.proj_dim)
                
                # Learnable CLS token for each node (same dimension as transformer input)
                self.cls_token = nn.Parameter(torch.randn(1, 1, self.proj_dim))

                # Define a single transformer encoder layer (with one attention head for simplicity).
                encoder_layer = nn.TransformerEncoderLayer(d_model=self.proj_dim, nhead=4, batch_first=True)
                self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
                # A linear layer to project the pooled transformer output to a scalar.
                self.transformer_out_linear = nn.Linear(self.proj_dim, 1)
                
                anc_idx, anc_msk = self._compute_ancestor_indices()
                self.register_buffer("anc_idx", anc_idx)
                self.register_buffer("anc_msk", anc_msk) 
                
                torch.backends.cuda.enable_mem_efficient_sdp(False)
                torch.backends.cuda.enable_flash_sdp(False)
                torch.backends.cuda.enable_math_sdp(True)

        self.scoring_functions = {
            "mean_then_linear": self._mean_then_linear,
            "linear_then_mean": self._linear_then_mean,
            "flatten_then_linear": self._flatten_then_linear_shared,
            "tree_based": self._tree_based_scoring_fn,
            "flatten_then_linear_unshared": self._flatten_then_linear_unshared,
            "linear_per_nt_then_mean": self._linear_per_nt_then_mean,
            "propagate_sep_mlp": self._propagate_sep_mlp,
            "propagate_level_wise": self._propagate_level_wise,
            "propagate_single_mlp": self._propagate_single_mlp,
            "propagate_transformer": self._propagate_transformer,
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
        
        # ===== Apply token mixer =====
        tokens = self.token_mixer(tokens, mask)  # [B, n_d, d_enc]
        
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

        # Apply embedding adapter if present
        if hasattr(self, 'adapter') and self.adapter is not None and self.adapter.adapt_embeddings:
            node_embeddings = self.adapter.adjust_embeddings(node_embeddings)

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
        node_scores, pre_mlp_scores = self.get_scores_from_attn_out(aggregated_values)  # [B, n_t]
        
        # Attach CLS tokens with node scores 
        # node_scores_with_tokens = torch.hstack([node_scores, tokens[:, 0]])

        # return node_scores, {
        #     "node_scores_post_mlp": node_scores.detach().cpu(),
        #     "node_scores_pre_mlp": pre_mlp_scores.detach().cpu(),
        #     "attention_scores": attention_scores.detach().cpu(),
        # }
        # return node_scores_with_tokens, {}
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
        W = self.weight_matrix
        b = self.bias_matrix
        if hasattr(self, 'adapter') and self.adapter is not None and self.adapter.adapt_scoring:
            W, b = self.adapter.adjust_scoring(W, b)

        linear_out = torch.einsum("btnd,td->btn", attn_out, W)

        # Add bias: (Nt) -> (B, Nt, Ne)
        linear_out += b[None, :, None]

        # Mean pool over Ne dimension: (B, Nt)
        return linear_out.mean(dim=2)

    def _tree_based_scoring_fn(self, attn_out: torch.Tensor):
        B, Nt, Ne, Dk = attn_out.shape

        # Apply weights: (B, Nt, Ne, Dk) x (Nt, Dk) -> (B, Nt, Ne, Nt)
        linear_out = torch.einsum("btnd,sd->btns", attn_out, self.weight_matrix)

        # Add bias: (Nt) -> (B, Nt, Ne, Nt)
        linear_out += self.bias_matrix[None, None, None, :]

        linear_out = linear_out.sum(2)  # (B, Nt, Nt)
        linear_out = linear_out * self.tree_matrix[None]  # (B, Nt, Nt)
        linear_out = linear_out.sum(-1)  # (B, Nt)
        scale = self.tree_matrix.sum(-1) * self.n_e

        return linear_out / scale
    
    def _product_propagation(self, attn_out):
        scores = self._linear_per_nt_then_mean(attn_out)
        # return scores
        B, nt = scores.shape

        # Apply logit adapter if present (before sigmoid)
        if hasattr(self, 'adapter') and self.adapter is not None and self.adapter.adapt_logits:
            scores = self.adapter.adjust_logits(scores)

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

    
    def _propagate_level_wise(self, attn_out):
        scores = self._linear_per_nt_then_mean(attn_out)
        B, nt = scores.shape

        anc_idx_expanded = self.anc_idx.unsqueeze(0).expand(B, -1, -1)  # [B, n_t, max_len]
        anc_scores = torch.gather(scores.unsqueeze(-1).expand(-1, -1, anc_idx_expanded.shape[-1]), dim=1, index=anc_idx_expanded) # [B, n_t, max_len]
        
        
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
    
    def _propagate_transformer(self, attn_out):
        """
        This scoring function uses a transformer encoder to process the sequence of 
        ancestor representations for each node. For each node, we gather its ancestor scores 
        and the corresponding node embeddings (mean over embeddings), concatenate them, and 
        then use a transformer followed by masked pooling and a final projection.
        """
        scores = self._linear_per_nt_then_mean(attn_out)
        B, nt = scores.shape

        anc_idx_expanded = self.anc_idx.unsqueeze(0).expand(B, -1, -1)  # [B, n_t, max_len]
        anc_scores = torch.gather(scores.unsqueeze(-1).expand(-1, -1, anc_idx_expanded.shape[-1]), dim=1, index=anc_idx_expanded) # [B, n_t, max_len]

        # Apply mask to zero out invalid ancestors: [B, n_t, max_len]
        anc_scores = anc_scores * self.anc_msk.unsqueeze(0)
        
        # Now gather ancestor node embeddings.
        # For each node, use the mean of its multiple embeddings.
        # node_emb_mean: [n_t, d_emb]
        node_emb_mean = self.node_embeddings.mean(dim=1)
        # Gather ancestor embeddings: using self.anc_idx (shape [n_t, levels])
        # This yields [n_t, levels, d_emb], then expand to batch dimension.
        anc_emb = node_emb_mean[self.anc_idx]  # [n_t, levels, d_emb]
        anc_emb = anc_emb.unsqueeze(0).expand(B, -1, -1, -1)  # [B, n_t, levels, d_emb]

        # Combine the ancestor scores and embeddings: 
        # Expand scores to [B, n_t, levels, 1] and then concatenate.
        anc_scores_unsq = anc_scores.unsqueeze(-1)  # [B, n_t, levels, 1]
        transformer_input = torch.cat([anc_scores_unsq, anc_emb], dim=-1)  # [B, n_t, levels, 1+d_emb]

        # Prepare for transformer encoder:
        # Flatten batch and node dimensions: new shape [B*n_t, levels, 1+d_emb]
        transformer_input = transformer_input.view(B * self.n_t, self.levels, self.transformer_input_dim)
        
        # Prepend the CLS token to each sequence.
        # Expand the CLS token from shape [1, 1, transformer_input_dim] to [B*n_t, 1, transformer_input_dim].
        cls_tokens = self.cls_token.expand(B * self.n_t, -1, -1)
        
        transformer_input = self.input_proj(transformer_input)  # now shape: [B*n_t, 1+levels, proj_dim]
        transformer_input = torch.cat([cls_tokens, transformer_input], dim=1)  # [B*n_t, 1+levels, transformer_input_dim]

        # Create key padding mask:
        # anc_msk has shape [n_t, levels] -> expand to [B, n_t, levels] then flatten to [B*n_t, levels]
        key_padding = (self.anc_msk.unsqueeze(0).expand(B, -1, -1)
                       .reshape(B * self.n_t, self.levels) == 0)
        # Prepend a column of False for the CLS token (CLS token is always valid)
        key_padding = torch.cat([torch.zeros(B * self.n_t, 1, device=key_padding.device, dtype=torch.bool),
                                 key_padding], dim=1)  # [B*n_t, 1+levels]

        # Pass through the transformer encoder (no causal mask used)
        encoded_seq = self.transformer_encoder(
            transformer_input,
            src_key_padding_mask=key_padding
        )  # [B*n_t, 1+levels, transformer_input_dim]

        # Use the CLS token representation for each node.
        cls_rep = encoded_seq[:, 0, :]  # [B*n_t, transformer_input_dim]
        out_flat = self.transformer_out_linear(cls_rep).squeeze(-1)  # [B*n_t]
        outputs = out_flat.view(B, self.n_t)

        return outputs
        

    def _init_params(self):
        """
        Doing a custom initialization of the projector layers better suited for attention.
        """
        for layer in [self.key_proj, self.value_proj, self.query_proj, self.output_proj]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoderSplit(nn.Module):
    """
    Computes node scores by processing a single sequence per sample composed of:
      - The node embeddings for all nodes (of a tree with nb_splits nodes),
      - Followed by the document (text) token embeddings.
    
    To reduce memory, we work in a lower-dimensional space (d_model) than the original text
    (in_size, e.g. 1024) and node embeddings (node_emb_dim, e.g. 768). We use two linear
    projections to map these into a common d_model space.
    
    In self-attention, we enforce that for a node token (position i, 0<= i < nb_splits)
    the allowed keys among the node tokens are only those corresponding to its ancestors,
    as given by self.anc_idx, while text tokens (positions nb_splits:) are always allowed.
    
    The final node score is read off from the output at the corresponding position.
    """
    def __init__(
        self,
        in_size: int,             # Dimension of document token features (e.g. 1024)
        nb_splits: int,           # Number of nodes in the tree (e.g. 2047)
        node_emb_dim: int = 1024,        # Dimension of node embeddings (e.g. 768)
        max_doc_len: int = 512,   # Maximum document token length.
        n_heads: int = 8,         # Number of attention heads.
        n_layers: int = 2,        # Number of transformer encoder layers.
        d_model: int = 1024,       # Internal transformer dimension.
        ff_multiplier: int = 1    # Factor to set feedforward dimension (e.g. 2*d_model).
    ):
        super(TransformerEncoderSplit, self).__init__()
        self.in_size = in_size
        self.node_emb_dim = node_emb_dim
        self.nb_splits = nb_splits
        self.d_model = d_model
        self.max_doc_len = max_doc_len

        # Compute tree depth (levels) based on nb_splits (assumes binary tree).
        self.levels = int(math.floor(math.log2(nb_splits + 1)))
        
        # Node embeddings are stored in their original dimension.
        self.node_embeddings = nn.Parameter(torch.randn(nb_splits, node_emb_dim))
        
        # Compute ancestor indices and mask.
        anc_idx, anc_msk = self._compute_ancestor_indices(nb_splits, self.levels)
        # anc_idx: [nb_splits, levels]; anc_msk: [nb_splits, levels] with 1 for valid.
        self.register_buffer("anc_idx", anc_idx)
        self.register_buffer("anc_msk", anc_msk)
        
        # Linear projections into d_model space.
        self.token_proj = nn.Identity() # nn.Linear(in_size, d_model)
        self.node_proj = nn.Identity() # nn.Linear(node_emb_dim, d_model)
        
        # Learned positional encoding over the entire sequence.
        # The full sequence length is L_total = nb_splits + max_doc_len.
        self.L_total = nb_splits + max_doc_len
        self.pos_embedding = nn.Embedding(self.L_total, d_model)
        
        # Precompute the custom attention mask for node tokens.
        # We'll create a mask for the node block: shape [nb_splits, nb_splits].
        # For each node i, only positions corresponding to its ancestors (anc_idx[i]) are allowed.
        node_attn_mask = torch.full((nb_splits, nb_splits), float('-inf'))
        for i in range(nb_splits):
            allowed = anc_idx[i]  # allowed indices for node i (a tensor of shape [levels])
            node_attn_mask[i, allowed] = 0.0  # allowed positions get 0 (no mask)
        # This is fixed and will be reused.
        self.register_buffer("node_attn_mask", node_attn_mask)
        
        # Build a full src_mask of shape [L_total, L_total] for the transformer.
        # The sequence order is: [nodes (nb_splits) ; text tokens (max_doc_len)].
        # For rows corresponding to node tokens (0:nb_splits):
        #   - For columns 0:nb_splits, use node_attn_mask.
        #   - For columns nb_splits: L_total, allow full attention (0).
        # For rows corresponding to text tokens, allow full attention.
        full_mask = torch.zeros(self.L_total, self.L_total)
        full_mask[0:nb_splits, 0:nb_splits] = self.node_attn_mask
        # The rest remains 0.
        self.register_buffer("src_mask_full", full_mask)
        
        # Transformer Encoder with batch_first.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_multiplier,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Final projection to scalar score from d_model.
        self.score_proj = nn.Linear(d_model, 1)
        
        self.levels = int(math.log2(self.nb_splits + 1))
        self.hidden_dim = max(1, self.levels // 2)
        self.shared_mlp = nn.Sequential(
            nn.Linear(self.levels, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, 1)
        )    
    
    def _compute_ancestor_indices(self, nb_splits, levels):
        """
        For each node i (0 <= i < nb_splits), compute a list of indices:
            anc_idx[i, 0] = i (the node itself)
            anc_idx[i, 1] = parent(i)
            anc_idx[i, 2] = parent's parent, etc.
        If a node has fewer than 'levels' ancestors, remaining entries are left as 0.
        Also return anc_msk: 1 for valid entries, 0 for padded.
        """
        anc_idx = torch.zeros((nb_splits, levels), dtype=torch.long)
        anc_msk = torch.zeros((nb_splits, levels), dtype=torch.float)
        # For root node:
        anc_idx[0, 0] = 0
        anc_msk[0, 0] = 1.0
        for i in range(1, nb_splits):
            parent = (i - 1) // 2
            anc_idx[i, 1:] = anc_idx[parent, :-1]
            anc_msk[i, 1:] = anc_msk[parent, :-1]
            anc_idx[i, 0] = i
            anc_msk[i, 0] = 1.0
        return anc_idx, anc_msk
    
    def forward(self, tokens: torch.Tensor, token_mask: torch.Tensor):
        """
        Args:
            tokens: [B, n_d, in_size] document token features.
            token_mask: [B, n_d] mask with 1 for valid tokens, 0 for padding.
        Returns:
            scores: [B, nb_splits] node scores.
        """
        B, n_d, _ = tokens.shape
        # Compute actual sequence length for this batch: 
        # L = nb_splits (all node embeddings) + n_d (actual number of text tokens)
        L = self.nb_splits + n_d
            
        # Project text tokens to d_model space.
        tokens_proj = self.token_proj(tokens)  # [B, n_d, d_model]
        # (No need to pad tokens if n_d < max_doc_len, we use the actual n_d.)
        
        # Project node embeddings to d_model.
        # self.node_embeddings: [nb_splits, node_emb_dim]
        nodes_proj = self.node_proj(self.node_embeddings)  # [nb_splits, d_model]
        # Expand to batch: repeat for each example.
        nodes_proj = nodes_proj.unsqueeze(0).expand(B, -1, -1)  # [B, nb_splits, d_model]
        
        # Concatenate node embeddings and text token embeddings.
        # We form a sequence of length L = nb_splits + n_d.
        seq = torch.cat([nodes_proj, tokens_proj], dim=1)  # [B, nb_splits+n_d, d_model]
        
        # Add positional encodings.
        # We use the first L positions from the learned positional encoding.
        pos_ids = torch.arange(L, device=tokens.device).unsqueeze(0)  # [1, L]
        pos_emb = self.pos_embedding(pos_ids)  # [1, L, d_model]
        seq = seq + pos_emb  # [B, L, d_model]
        
        # Build a src_mask for self-attention.
        # Our precomputed full mask (self.src_mask_full) was of shape [L_total, L_total]
        # with L_total = nb_splits + max_doc_len. Now we take the top-left LxL submatrix.
        src_mask = self.src_mask_full[:L, :L]  # [L, L]
        
        # Build key_padding_mask for text tokens.
        # For node tokens (first nb_splits positions): no padding.
        # For text tokens (positions nb_splits:L): use token_mask.
        key_padding = torch.zeros(B, L, dtype=torch.bool, device=tokens.device)
        # For each example, for text tokens beyond actual n_d (if any), mark as padded.
        # Since we don't pad tokens_proj here, this mask should mark only positions nb_splits:n_d (which are valid)
        # and leave any positions beyond n_d (if using a fixed max) as padded.
        # Here, L = nb_splits + n_d, so the text portion is exactly n_d long.
        # Therefore, key_padding[:, nb_splits:] remains all False.
        key_padding[:, self.nb_splits:] = ~token_mask.bool()
        
        # Run the transformer encoder.
        enc_out = self.transformer_encoder(seq, mask=src_mask, src_key_padding_mask=key_padding)
        # enc_out: [B, L, d_model]
        
        # Extract node outputs: positions 0:nb_splits.
        node_out = enc_out[:, :self.nb_splits, :]  # [B, nb_splits, d_model]
        
        # Compute scores.
        scores = self.score_proj(node_out).squeeze(-1)  # [B, nb_splits]
        
        # anc_idx_expanded = self.anc_idx.unsqueeze(0).expand(B, -1, -1)  # [B, n_t, max_len]
        # anc_scores = torch.gather(scores.unsqueeze(-1).expand(-1, -1, anc_idx_expanded.shape[-1]), dim=1, index=anc_idx_expanded) # [B, n_t, max_len]

        # # Apply mask to zero out invalid ancestors: [B, n_t, max_len]
        # anc_scores = anc_scores * self.anc_msk.unsqueeze(0)
        
        # # Flatten the node dimension to process all nodes in one batch.
        # anc_scores_flat = anc_scores.view(B * self.nb_splits, self.levels)  # [B*n_t, levels]

        # out_flat = self.shared_mlp(anc_scores_flat).squeeze(-1)  # [B*n_t]
        # outputs = out_flat.view(B, self.nb_splits)

        # return outputs
        return scores


import torch
import torch.nn as nn
import math
from transformers import AutoModelForCausalLM, AutoTokenizer

class LLMSplit(nn.Module):
    """
    A split function that uses a large language model (LLM) as a backbone.
    The prompt is constructed as follows:
      [text tokens] + [tree start token] + [learned node embeddings] + [tree end token].
      
    The LLM is loaded using its name (with quantization support) and its final head is removed.
    The score for each node is read off from the LLM’s output at the corresponding position.
    
    Options:
      - lora_finetune: if True, apply LoRA (e.g. via PEFT) to the LLM so that the LLM is finetuned.
                       Otherwise, the LLM is frozen and only the node embeddings and final score
                       layer are learned.
    """
    def __init__(
        self,
        in_size: int,             # Dimension of document token features (e.g. 1024)
        nb_splits: int,    
        llm_name: str = "meta-llama/Llama-3.2-1B-Instruct",            # Name or path of the LLM to load.
        in_quantize: bool = True,  # Whether to load in quantized mode (if supported).
        lora_finetune: bool = False, # Whether to apply LoRA finetuning.
        node_emb_dim: int = 768,
        tree_start_token: str = "<tree>",  # Special token marking start of node embeddings.
        tree_end_token: str = "</tree>",   # Special token marking end.
        attend_ne_only_on_text: bool = False,
        use_sigmoid_in_projection: bool = False,
        enable_llm_split_score_prop: bool = False,
    ):
        super(LLMSplit, self).__init__()
        self.nb_splits = nb_splits
        self.n_t = nb_splits
        
        rank_info = get_local_rank_and_world_size()
        local_rank = rank_info.local_rank

        # Load DeepSpeed config
        # ds_config = json.load(open("scripts/config/deepspeed_retreever.json"))

        # Initialize DeepSpeed config (this sets up a global configuration)
        # dschf = HfDeepSpeedConfig(ds_config)

        # Load tokenizer and model.
        # We assume the model is a causal LM.
        # If quantization is desired, use the appropriate flag (here, e.g., load_in_8bit).
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_name,
            cache_dir=PATH_HF_CACHE_RW)
        if in_quantize:
            self.model = AutoModelForCausalLM.from_pretrained(
                llm_name,
                cache_dir=PATH_HF_CACHE_RW,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={"": local_rank}
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(llm_name)
            
        # model_engine, _, _, _ = deepspeed.initialize(
        #         model=self.model,
        #         config=ds_config
        # )
        
        # self.model = model_engine
                
        # Save hidden dimension from the model.
        self.hidden_dim = self.model.config.hidden_size
        
        # Register the special tokens.
        if tree_start_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens([tree_start_token, tree_end_token])
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.tree_start_token = tree_start_token
        self.tree_end_token = tree_end_token
        
        
        # Remove the original LM head. We assume the model has a language modeling head attribute,
        # e.g. model.lm_head or model.score, so we replace it with an identity.
        if hasattr(self.model, "lm_head"):
            self.model.lm_head = nn.Identity()
        elif hasattr(self.model, "score"):
            self.model.score = nn.Identity()
        else:
            raise ValueError("Cannot find LM head to remove.")

        # Freeze model parameters if not finetuning with LoRA.
        if not lora_finetune:
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            from peft import get_peft_model, LoraConfig
            lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=["c_attn"], lora_dropout=0.1)
            self.model = get_peft_model(self.model, lora_config)
            pass

        # Learnable node embeddings: these are "soft prompts" for each node.
        # They are learned in the same hidden space as the LLM.
        self.node_embeddings = nn.Parameter(torch.randn(nb_splits, node_emb_dim))
        
        self.node_emb_proj = nn.Linear(node_emb_dim, self.hidden_dim)
        
        # Final linear layer to produce a scalar score from the LLM output for each node.
        self.score_proj = nn.Linear(self.hidden_dim, 1)
        
        if use_sigmoid_in_projection:
            print("!!!!!!!!!!!!!!!!! Using sigmoid in projection !!!!!!!!!!!!!!!!!!!")
            self.score_proj = nn.Sequential(
                nn.Linear(self.hidden_dim, 1),
                nn.Sigmoid()
            )
        
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
        
        self.attend_ne_only_on_text = attend_ne_only_on_text
        self.enable_llm_split_score_prop = enable_llm_split_score_prop
        if attend_ne_only_on_text:
            self._precompute_attention_templates()
        
    def _precompute_attention_templates(self):
        """
        Precompute static parts of the attention mask that don't depend on L_text.
        Only the node-to-node blocking pattern is truly static.
        """
        # Template: Node-to-node blocking pattern (the only L_text-independent part)
        # This is the key optimization - precompute which nodes block which other nodes
        node_block_pattern = torch.zeros((self.nb_splits, self.nb_splits))
        for i in range(self.nb_splits):
            for j in range(self.nb_splits):
                if i != j:  # Block attention between different nodes
                    node_block_pattern[i, j] = float('-inf')
        self.register_buffer("node_block_pattern", node_block_pattern)
    
      
    def _create_llama_attention_mask_fast(self, B, L_text, device):
        """
        Fast attention mask creation using precomputed node blocking pattern.
        Creates the causal mask on-the-fly since it depends on L_text.
        
        This is faster than the original because:
        1. Uses precomputed node blocking pattern (O(1) assignment vs O(nb_splits^2) loops)
        2. Still creates causal mask on-demand since it's L_text dependent
        """
        seq_len = L_text + 2 + self.nb_splits  # text + tree_start + nodes + tree_end
        
        # Step 1: Create causal mask for this specific sequence length
        # This must be done on-the-fly since seq_len depends on L_text
        attention_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1)
        
        # Step 2: Apply precomputed node blocking pattern
        # Node positions: [L_text+1, L_text+1+nb_splits)
        node_start = L_text + 1
        node_end = L_text + 1 + self.nb_splits
        
        # This is the key optimization: O(1) assignment instead of O(nb_splits^2) loops
        attention_mask[node_start:node_end, node_start:node_end] = self.node_block_pattern
        
        # Step 3: Expand to batch dimension
        # [seq_len, seq_len] -> [B, 1, seq_len, seq_len]
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)
        
        return attention_mask
    
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
    
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        Forward pass.
        
        Args:
            input_ids: [B, L_text] token ids for the text portion.
            attention_mask: [B, L_text] attention mask for text tokens (optional).
            
        Returns:
            scores: [B, nb_splits] node scores.
        """
        B, L_text = input_ids.shape
        
        # Get text token embeddings from the LLM's input embedding layer.
        # Assume the model has an attribute get_input_embeddings().
        text_embeds = self.model.get_input_embeddings()(input_ids)  # [B, L_text, hidden_dim]
        
        # Obtain embeddings for the special tree tokens (from the model's embeddings).
        tree_start_id = self.tokenizer.convert_tokens_to_ids(self.tree_start_token)
        tree_end_id = self.tokenizer.convert_tokens_to_ids(self.tree_end_token)
        tree_start_embed = self.model.get_input_embeddings()(torch.tensor([tree_start_id], device=input_ids.device))  # [1, hidden_dim]
        tree_end_embed = self.model.get_input_embeddings()(torch.tensor([tree_end_id], device=input_ids.device))      # [1, hidden_dim]
        
        # Expand special token embeddings to batch size.
        tree_start_embed = tree_start_embed.unsqueeze(0).expand(B, -1, -1)  # [B, 1, hidden_dim]
        tree_end_embed = tree_end_embed.unsqueeze(0).expand(B, -1, -1)      # [B, 1, hidden_dim]
        
        # The full prompt is:
        #  [text tokens] + [tree start token] + [node embeddings] + [tree end token]
        # The node embeddings are learned parameters (same for all examples).
        node_embeds = self.node_emb_proj(self.node_embeddings).unsqueeze(0).expand(B, -1, -1)  # [B, nb_splits, hidden_dim]
        
        # Concatenate all parts.
        prompt_embeds = torch.cat([text_embeds, tree_start_embed, node_embeds, tree_end_embed], dim=1)
        # The sequence length is L_text + 1 + nb_splits + 1.
        
        
        if self.attend_ne_only_on_text:
             # Create custom 4D attention mask for node isolation (fast version)
            custom_attention_mask_4d = self._create_llama_attention_mask_fast(B, L_text, input_ids.device)
            prompt_mask = custom_attention_mask_4d.to(prompt_embeds.dtype)

        else:
            if attention_mask is None:
                text_mask = torch.ones(B, L_text, device=input_ids.device, dtype=torch.bool)
            else:
                text_mask = attention_mask.bool()  # [B, L_text]
            # For the tree start token, node tokens, and tree end token, assume they are all valid.
            extra_mask = torch.ones(B, 2 + self.nb_splits, device=input_ids.device, dtype=torch.bool)
            prompt_mask = torch.cat([text_mask, extra_mask], dim=1)
            
        
        # Run the prompt through the LLM.
        # Because the model is causal, each token attends only to previous tokens.
        # We use the standard causal attention mask built into the model.
        outputs = self.model(
            inputs_embeds=prompt_embeds,
            attention_mask=prompt_mask,
            output_hidden_states=True,
            return_dict=True
        )

        # Get hidden states from the last layer.
        hidden_states = outputs.hidden_states[-1]  # [B, L_prompt, hidden_dim]
        
        # The node outputs are at positions: L_text+1 to L_text+1+nb_splits.
        start_idx = L_text + 1
        end_idx = start_idx + self.nb_splits
        node_outputs = hidden_states[:, start_idx:end_idx, :]  # [B, nb_splits, hidden_dim]
        
        # Compute scores for each node.
        scores = self.score_proj(node_outputs).squeeze(-1)  # [B, nb_splits]
        
        if self.enable_llm_split_score_prop:
            B, nt = scores.shape

            anc_idx_expanded = self.anc_idx.unsqueeze(0).expand(B, -1, -1)  # [B, n_t, max_len]
            anc_scores = torch.gather(scores.unsqueeze(-1).expand(-1, -1, anc_idx_expanded.shape[-1]), dim=1, index=anc_idx_expanded) # [B, n_t, max_len]

            # Apply mask to zero out invalid ancestors: [B, n_t, max_len]
            anc_scores = anc_scores * self.anc_msk.unsqueeze(0)
            
            # Flatten the node dimension to process all nodes in one batch.
            anc_scores_flat = anc_scores.view(B * self.n_t, self.levels)  # [B*n_t, levels]

            out_flat = self.shared_mlp(anc_scores_flat).squeeze(-1)  # [B*n_t]
            outputs = out_flat.view(B, self.n_t)

            return outputs.float()
        
        else:
            return scores.float()



split_dict = {  # supported split modulestorch.nn.init.zeros_(self.value_proj.bias)
    "linear": LinearSplit,
    "mlp": MLPSplit,
    "attn": AttentionSplit,
    "cross_attn": CrossAttentionSplit,
    "transformer_encoder": TransformerEncoderSplit,
    "llm_split": LLMSplit,
}

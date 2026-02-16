"""
Node embedding strategies for tree-based retrieval.

Provides different ways to initialize and parameterize node embeddings,
from fully independent embeddings to highly parameter-efficient schemes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class IndependentNodeEmbeddings(nn.Module):
    """
    Baseline: Each node has completely independent embedding.
    
    Parameters: n_nodes × n_emb × d_emb
    
    Use case: Maximum flexibility, no sharing. Current default.
    """
    def __init__(self, n_nodes: int, n_emb_per_node: int, d_emb: int, **kwargs):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_emb_per_node = n_emb_per_node
        self.d_emb = d_emb
        
        # Each node gets its own embedding(s)
        self.node_embeddings = nn.Parameter(
            torch.randn(n_nodes, n_emb_per_node, d_emb)
        )
    
    def forward(self):
        """
        Returns:
            [n_nodes, n_emb_per_node, d_emb]
        """
        return self.node_embeddings
    
    def get_num_parameters(self):
        return self.n_nodes * self.n_emb_per_node * self.d_emb


class LevelBasedEmbeddings(nn.Module):
    """
    Level-based: Nodes at same level share embeddings.
    Position within level distinguished by sinusoidal encoding (no params).
    
    Parameters: n_levels × n_emb × d_emb
    
    Use case: Hierarchical structure, massive parameter reduction.
    """
    def __init__(
        self, 
        n_nodes: int, 
        n_emb_per_node: int, 
        d_emb: int,
        tree_depth: int = 10,
        **kwargs
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_emb_per_node = n_emb_per_node
        self.d_emb = d_emb
        self.n_levels = tree_depth + 1  # 0 to depth inclusive
        
        # Embeddings per level (shared by all nodes at that level)
        self.level_embeddings = nn.Parameter(
            torch.randn(self.n_levels, n_emb_per_node, d_emb)
        )
        
        # Precompute tree structure
        self.register_buffer("node_levels", self._compute_node_levels())
        self.register_buffer("node_positions", self._compute_node_positions())
        self.register_buffer("sinusoidal_encodings", self._get_all_sinusoidal_encodings())
    
    def _compute_node_levels(self):
        """Compute which level each node belongs to."""
        levels = torch.zeros(self.n_nodes, dtype=torch.long)
        for i in range(self.n_nodes):
            levels[i] = int(math.floor(math.log2(i + 1)))
        return levels
    
    def _compute_node_positions(self):
        """Compute position of each node within its level."""
        positions = torch.zeros(self.n_nodes, dtype=torch.long)
        for i in range(self.n_nodes):
            level = int(math.floor(math.log2(i + 1)))
            positions[i] = i - (2**level - 1)
        return positions
    
    def forward(self):
        """
        Vectorized version - NO for loops!
        
        Returns:
            [n_nodes, n_emb_per_node, d_emb]
        """
        # Get level embeddings for all nodes at once
        level_embs = self.level_embeddings[self.node_levels]  # [n_nodes, n_emb, d_emb]
        
        # Broadcast and add: [n_nodes, n_emb, d_emb] + [n_nodes, 1, d_emb]
        node_embeddings = level_embs + self.sinusoidal_encodings.unsqueeze(1)
        
        return node_embeddings.to(self.level_embeddings.dtype)

    def _get_all_sinusoidal_encodings(self):
        """
        Compute sinusoidal encodings for all nodes at once.
        
        Returns:
            [n_nodes, d_emb]
        """
        # Compute angles for all nodes: position / num_positions_at_level * 2π
        num_positions_at_level = 2 ** self.node_levels  # [n_nodes]
        angles = 2 * math.pi * self.node_positions.float() / num_positions_at_level.float()  # [n_nodes]
        
        # Create frequency terms: [d_emb//2]
        half_dim = self.d_emb // 2
        freqs = torch.exp(
            torch.arange(0, self.d_emb, 2, device=self.level_embeddings.device).float() *
            (-math.log(10000.0) / self.d_emb)
        )  # [d_emb//2]
        
        # Compute sin/cos for all nodes and all frequencies: [n_nodes, 1] * [1, d_emb//2]
        angles_expanded = angles.unsqueeze(1)  # [n_nodes, 1]
        freqs_expanded = freqs.unsqueeze(0)    # [1, d_emb//2]
        
        angle_freqs = angles_expanded * freqs_expanded  # [n_nodes, d_emb//2]
        
        # Interleave sin and cos
        encodings = torch.zeros(self.n_nodes, self.d_emb, device=self.level_embeddings.device)
        encodings[:, 0::2] = torch.sin(angle_freqs)
        encodings[:, 1::2] = torch.cos(angle_freqs)
        
        return encodings
        
    
    def get_num_parameters(self):
        return self.n_levels * self.n_emb_per_node * self.d_emb


class LowRankNodeEmbeddings(nn.Module):
    """
    Level embeddings + low-rank node-specific updates.
    
    node_emb[i] = level_emb[L] + (A[i] @ B)
    
    Parameters: n_levels × n_emb × d_emb + n_nodes × rank + rank × d_emb
    
    Use case: Balance between flexibility and parameter efficiency.
    """
    def __init__(
        self,
        n_nodes: int,
        n_emb_per_node: int,
        d_emb: int,
        tree_depth: int = 10,
        rank: int = 32,
        **kwargs
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_emb_per_node = n_emb_per_node
        self.d_emb = d_emb
        self.n_levels = tree_depth + 1
        self.rank = rank
        
        # Level embeddings (shared)
        self.level_embeddings = nn.Parameter(
            torch.randn(self.n_levels, n_emb_per_node, d_emb)
        )
        
        # Low-rank update matrices
        self.A = nn.Parameter(torch.randn(n_nodes, rank) * 0.01)  # Small init
        self.B = nn.Parameter(torch.randn(rank, d_emb) * 0.01)
        
        # Precompute tree structure
        self.register_buffer("node_levels", self._compute_node_levels())
    
    def _compute_node_levels(self):
        levels = torch.zeros(self.n_nodes, dtype=torch.long)
        for i in range(self.n_nodes):
            levels[i] = int(math.floor(math.log2(i + 1)))
        return levels
    
    def forward(self):
        """
        Returns:
            [n_nodes, n_emb_per_node, d_emb]
        """
        # Get level embeddings for all nodes
        level_embs = self.level_embeddings[self.node_levels]  # [n_nodes, n_emb, d_emb]
        
        # Compute low-rank updates for all nodes
        updates = self.A @ self.B  # [n_nodes, d_emb]
        updates = updates.unsqueeze(1)  # [n_nodes, 1, d_emb]
        
        # Add updates to level embeddings
        node_embeddings = level_embs + updates  # Broadcast
        
        return node_embeddings
    
    def get_num_parameters(self):
        level_params = self.n_levels * self.n_emb_per_node * self.d_emb
        lowrank_params = self.n_nodes * self.rank + self.rank * self.d_emb
        return level_params + lowrank_params


class FixedWithLowRankUpdates(nn.Module):
    """
    Fixed random initialization + learned low-rank updates.
    
    node_emb[i] = fixed_emb[i] + (A[i] @ B)
    
    Parameters: n_nodes × rank + rank × d_emb
    
    Use case: Explore minimal updates to random initialization.
    """
    def __init__(
        self,
        n_nodes: int,
        n_emb_per_node: int,
        d_emb: int,
        rank: int = 32,
        **kwargs
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_emb_per_node = n_emb_per_node
        self.d_emb = d_emb
        self.rank = rank
        
        # Fixed random embeddings (NOT learned)
        self.register_buffer(
            "fixed_embeddings",
            torch.randn(n_nodes, n_emb_per_node, d_emb)
        )
        
        # Low-rank update matrices (learned)
        self.A = nn.Parameter(torch.randn(n_nodes, rank) * 0.01)
        self.B = nn.Parameter(torch.randn(rank, d_emb) * 0.01)
    
    def forward(self):
        """
        Returns:
            [n_nodes, n_emb_per_node, d_emb]
        """
        # Compute low-rank updates
        updates = self.A @ self.B  # [n_nodes, d_emb]
        updates = updates.unsqueeze(1)  # [n_nodes, 1, d_emb]
        
        # Add to fixed embeddings
        node_embeddings = self.fixed_embeddings + updates
        
        return node_embeddings
    
    def get_num_parameters(self):
        # Only low-rank matrices are learned
        return self.n_nodes * self.rank + self.rank * self.d_emb


class LevelOnlyEmbeddings(nn.Module):
    """
    Learns one embedding per tree level instead of per node.
    Returns [num_levels, n_emb_per_node, d_emb]
    """
    def __init__(self, n_nodes: int, n_emb_per_node: int, d_emb: int, tree_depth: int = 10, **kwargs):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_emb_per_node = n_emb_per_node
        self.d_emb = d_emb
        self.n_levels = tree_depth + 1
        
        # One embedding per level
        self.level_embeddings = nn.Parameter(
            torch.randn(self.n_levels, n_emb_per_node, d_emb)
        )
    
    def forward(self):
        return self.level_embeddings  # [n_levels, n_emb_per_node, d_emb]
    
    def get_num_parameters(self):
        return self.n_levels * self.n_emb_per_node * self.d_emb
    
    
class VectorQuantizedLevelEmbeddings(nn.Module):
    """
    Vector Quantized Level Embeddings.
    
    Uses codebook learning: each level has a codebook of K vectors.
    Given input encoding (e.g., CLS token), selects nearest codebook vector per level.
    
    Two modes:
    - Parallel: Each level independently selects from its codebook based on input (VECTORIZED)
    - Residual: Level i selects based on residual from levels 0..i-1 (RQ-VAE style)
    
    Parameters: n_levels × codebook_size × d_emb
    
    Use case: Discrete, data-dependent level embeddings optimized via VQ.
    """
    def __init__(
        self,
        n_nodes: int,
        n_emb_per_node: int,
        d_emb: int,
        tree_depth: int = 10,
        codebook_size: int = 256,
        is_residual: bool = False,
        commitment_cost: float = 0.25,
        use_simvq: bool = True,
        **kwargs
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_emb_per_node = 1
        self.d_emb = d_emb
        self.tree_depth = tree_depth
        # Codebooks for ACTUAL routing levels only (skip root)
        self.n_levels = tree_depth  # Levels 1, 2, ..., tree_depth
        self.codebook_size = codebook_size
        self.is_residual = is_residual
        self.commitment_cost = commitment_cost
        
        # Codebooks: [tree_depth, K, d_emb]
        # codebooks[0] corresponds to level 1 (first split)
        # codebooks[tree_depth-1] corresponds to level tree_depth (leaves)
        self.codebooks = nn.Parameter(
            torch.randn(self.n_levels, codebook_size, d_emb)
        )
        
        # Root embedding: single learned parameter (not data-dependent)
        # All documents get the same root embedding
        self.root_embedding = nn.Parameter(torch.randn(1, d_emb) * 0.5)
        
        # with torch.no_grad():
        #     for level in range(self.n_levels):
        #         self.codebooks[level] = F.normalize(
        #             self.codebooks[level], dim=-1
        #         ) * 0.5
        self.use_simvq = use_simvq
        if use_simvq:
            # Freeze codebooks, only optimize W
            self.codebooks.requires_grad = False
            # Learnable linear transformation matrix
            self.W = nn.Parameter(torch.randn(d_emb, d_emb))
        else:
            self.W = None

    def forward(self, z: torch.Tensor):
        """Returns: [B, tree_depth+1, d_emb] with level 0 = root, levels 1-D = VQ"""
        B, d = z.shape
        
        if self.is_residual:
            quantized_1_to_D, losses = self._forward_residual(z)  # [B, n_levels, d_emb]
        else:
            quantized_1_to_D, losses = self._forward_parallel(z)  # [B, n_levels, d_emb]
        
        # Expand root embedding for all batch elements: [B, 1, d_emb]
        root = self.root_embedding.unsqueeze(0).expand(B, -1, -1)  # [B, 1, d_emb]
        
        # Concatenate: [B, tree_depth+1, d_emb]
        quantized = torch.cat([root, quantized_1_to_D], dim=1)
        
        return quantized, losses

    def get_num_parameters(self):
        """Total learnable parameters."""
        if self.use_simvq:
            # Only W is learnable
            return self.d_emb * self.d_emb + self.d_emb  # W + root
        else:
            # Codebooks + root
            codebook_params = self.n_levels * self.codebook_size * self.d_emb
            root_params = self.d_emb
            return codebook_params + root_params

    def _forward_parallel(self, z: torch.Tensor):
        """
        Parallel VQ: Each level independently quantizes the input.
        FULLY VECTORIZED - NO FOR LOOPS!
        
        All levels see the same input z and independently select from their codebook.
        
        Args:
            z: [B, d_emb]
        
        Returns:
            quantized: [n_levels, 1, d_emb]
            losses: dict
        """
        B = z.shape[0]
        
        # Compute distances for all levels at once
        # z: [B, d_emb]
        # codebooks: [n_levels, K, d_emb]
        # Want: [B, n_levels, K] distances
        
        if self.use_simvq:
            # Transform codebooks: C @ W
            codebooks = torch.einsum('lkd,de->lke', self.codebooks, self.W)
        else:
            codebooks = self.codebooks
        
        # ||z||² for each level: [B, 1]
        z_sq = (z ** 2).sum(dim=1, keepdim=True)
        
        # Expand z for all levels: [B, n_levels, d_emb]
        z_expanded = z.unsqueeze(1).expand(B, self.n_levels, self.d_emb)
        
        # ||c||² for each codebook: [n_levels, K]
        c_sq = (codebooks ** 2).sum(dim=2)  # [n_levels, K]
        c_sq = c_sq.unsqueeze(0)  # [1, n_levels, K]
        
        # Dot products: [B, n_levels, d_emb] @ [n_levels, d_emb, K] → [B, n_levels, K]
        # Need to reshape for bmm
        z_flat = z.unsqueeze(1).expand(B, self.n_levels, self.d_emb).reshape(B * self.n_levels, self.d_emb)  # [B*n_levels, d_emb]
        codebooks_flat = codebooks.permute(0, 2, 1)  # [n_levels, d_emb, K]
        codebooks_expanded = codebooks_flat.unsqueeze(0).expand(B, -1, -1, -1)  # [B, n_levels, d_emb, K]
        codebooks_for_bmm = codebooks_expanded.reshape(B * self.n_levels, self.d_emb, self.codebook_size)
        
        dot_products = torch.bmm(
            z_flat.unsqueeze(1),  # [B*n_levels, 1, d_emb]
            codebooks_for_bmm     # [B*n_levels, d_emb, K]
        ).squeeze(1)  # [B*n_levels, K]
        dot_products = dot_products.reshape(B, self.n_levels, self.codebook_size)  # [B, n_levels, K]
        
        # Compute distances: ||z - c||² = ||z||² + ||c||² - 2⟨z, c⟩
        distances = z_sq.unsqueeze(1) + c_sq - 2 * dot_products  # [B, n_levels, K]
        
        # Find nearest codebook vector for each level: [B, n_levels]
        indices = torch.argmin(distances, dim=-1)
        
        # Gather quantized vectors: [B, n_levels, d_emb]
        # codebooks: [n_levels, K, d_emb]
        # indices: [B, n_levels]
        # For each batch b and level l, want codebooks[l, indices[b, l], :]
        
        # Expand codebooks: [1, n_levels, K, d_emb]
        codebooks_expanded = codebooks.unsqueeze(0).expand(B, -1, -1, -1)
        
        # Use gather: need to expand indices to [B, n_levels, 1, d_emb]
        indices_expanded = indices.unsqueeze(-1).unsqueeze(-1).expand(B, self.n_levels, 1, self.d_emb)
        quantized = torch.gather(
            codebooks_expanded,  # [B, n_levels, K, d_emb]
            dim=2,               # gather along K dimension
            index=indices_expanded  # [B, n_levels, 1, d_emb]
        ).squeeze(2)  # [B, n_levels, d_emb]
        
        # Straight-through estimator: [B, n_levels, d_emb]
        quantized_st = z_expanded + (quantized - z_expanded).detach()
        
        # Compute losses (vectorized)
        # VQ loss: update codebook toward encoder outputs
        vq_loss = F.mse_loss(z_expanded.detach(), quantized)
        
        # Commitment loss: encourage encoder to commit to codebook
        commit_loss = F.mse_loss(z_expanded, quantized.detach())
        
        # # Average over batch to get final embeddings: [n_levels, d_emb]
        # quantized_mean = quantized_st.mean(dim=0)  # [n_levels, d_emb]
        
        # # Add dimension for compatibility: [n_levels, 1, d_emb]
        # quantized_output = quantized_mean.unsqueeze(1)
        
        losses = {
            'vq_loss': vq_loss,
            'commit_loss': commit_loss,
            'total_quantization_loss': vq_loss + self.commitment_cost * commit_loss
        }
        
        return quantized_st, losses
    
    def _forward_residual(self, z: torch.Tensor):
        """
        Residual VQ (RQ-VAE style): Each level quantizes residual from previous levels.
        
        z = q₀ + q₁ + q₂ + ... where qᵢ comes from codebook i
        
        Args:
            z: [B, d_emb]
        
        Returns:
            quantized: [n_levels, 1, d_emb]
            losses: dict
        """
        B = z.shape[0]
        
        # ADD THIS:
        if self.use_simvq:
            codebooks = torch.einsum('lkd,de->lke', self.codebooks, self.W)
        else:
            codebooks = self.codebooks
            
        residual = z.clone()  # Start with full input
        quantized_levels = []
        total_vq_loss = 0.0
        total_commit_loss = 0.0
        
        for level in range(self.n_levels):
            codebook = codebooks[level]  # [K, d_emb]
            
            # Compute distances from residual: [B, K]
            r_sq = (residual ** 2).sum(dim=1, keepdim=True)  # [B, 1]
            c_sq = (codebook ** 2).sum(dim=1, keepdim=True).T  # [1, K]
            distances = r_sq + c_sq - 2 * (residual @ codebook.T)  # [B, K]
            
            # Find nearest
            indices = torch.argmin(distances, dim=-1)  # [B]
            
            # Lookup
            quantized = codebook[indices]  # [B, d_emb]
            
            # Straight-through
            quantized_st = residual + (quantized - residual).detach()
            
            # Losses
            vq_loss = F.mse_loss(residual.detach(), quantized)
            commit_loss = F.mse_loss(residual, quantized.detach())
            
            total_vq_loss += vq_loss
            total_commit_loss += commit_loss
            
            quantized_levels.append(quantized_st)  # [B, d_emb]
            
            # Update residual for next level
            residual = residual - quantized_st
        
        # Stack: [n_levels, 1, d_emb]
        quantized = torch.stack(quantized_levels, dim=1)  # [B, n_levels, d_emb]

        # Average losses
        total_vq_loss /= self.n_levels
        total_commit_loss /= self.n_levels
        
        losses = {
            'vq_loss': total_vq_loss,
            'commit_loss': total_commit_loss,
            'total_quantization_loss': total_vq_loss + self.commitment_cost * total_commit_loss
        }
        
        return quantized, losses
    
# ===== Factory function =====

def get_node_embedding_strategy(
    strategy: str,
    n_nodes: int,
    n_emb_per_node: int,
    d_emb: int,
    tree_depth: int = 10,
    rank: int = 32,
    **kwargs
):
    """
    Factory function to get node embedding strategy.
    
    Args:
        strategy: One of ["independent", "level_based", "hyperbolic_level",
                         "low_rank", "fixed_low_rank", "hyperbolic_low_rank"]
        n_nodes: Number of nodes in tree
        n_emb_per_node: Number of embeddings per node
        d_emb: Embedding dimension
        tree_depth: Depth of tree (for level-based strategies)
        rank: Rank for low-rank strategies
    
    Returns:
        Node embedding strategy module
    """
    strategy_dict = {
        "independent": IndependentNodeEmbeddings,
        "level_based": LevelBasedEmbeddings,
        "low_rank": LowRankNodeEmbeddings,
        "fixed_low_rank": FixedWithLowRankUpdates,
        "level_only": LevelOnlyEmbeddings,
        "vector_quantized": VectorQuantizedLevelEmbeddings,
    }
    
    if strategy not in strategy_dict:
        raise ValueError(
            f"Unknown strategy: {strategy}. "
            f"Choose from {list(strategy_dict.keys())}"
        )
    
    strategy_class = strategy_dict[strategy]
    
    return strategy_class(
        n_nodes=n_nodes,
        n_emb_per_node=n_emb_per_node,
        d_emb=d_emb,
        tree_depth=tree_depth,
        rank=rank,
        **kwargs
    )
"""
Token mixing strategies to adapt frozen encoder outputs for token-level use.

These adapters help models like BGE (trained for sentence-level embeddings) 
produce meaningful token-level representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class IdentityTokenMixer(nn.Module):
    """
    Identity mixer - passes tokens through unchanged.
    
    Use case: Baseline / ablation studies. Preserves original frozen encoder outputs.
    Useful for maintaining backward compatibility with existing models.
    """
    def __init__(self, d_model: int, **kwargs):
        super().__init__()
        # No parameters needed - just pass through
        # But we accept d_model for interface consistency
        pass
    
    def forward(self, tokens: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            tokens: [B, L, d_model]
            mask: [B, L] - Not used, kept for interface consistency
        Returns:
            tokens: [B, L, d_model] - Unchanged
        """
        return tokens


class LinearTokenMixer(nn.Module):
    """
    Simple linear transformation applied to each token independently.
    No residual connection - learns a full linear transformation.
    
    Use case: Simplest learnable adapter. Fast, minimal parameters.
    Good for when tokens just need a linear re-projection.
    """
    def __init__(self, d_model: int, use_bias: bool = True, **kwargs):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model, bias=use_bias)
        
        # Initialize with identity-like weights (near-identity transformation)
        # This makes early training more stable
        nn.init.eye_(self.linear.weight)
        if use_bias:
            nn.init.zeros_(self.linear.bias)
    
    def forward(self, tokens: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            tokens: [B, L, d_model]
            mask: [B, L] - Not used, kept for interface consistency
        Returns:
            adapted_tokens: [B, L, d_model]
        """
        return self.linear(tokens)


class TokenMLPAdapter(nn.Module):
    """
    MLP adapter that processes each token independently.
    Similar to the FFN in transformers, but specifically for adapting frozen encoder outputs.
    
    Use case: Simplest adapter with non-linearity, adds capacity without token communication.
    """
    def __init__(self, d_model: int, expansion_factor: int = 2, dropout: float = 0.1, **kwargs):
        super().__init__()
        hidden_dim = d_model * expansion_factor
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, tokens: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            tokens: [B, L, d_model]
            mask: [B, L] - Optional, not used but kept for interface consistency
        Returns:
            adapted_tokens: [B, L, d_model]
        """
        # Pre-norm + MLP + residual
        residual = tokens
        tokens = self.norm(tokens)
        tokens = self.mlp(tokens)
        return residual + tokens


class TokenSelfAttention(nn.Module):
    """
    Self-attention layer that allows tokens to communicate and refine their representations.
    This helps frozen encoder tokens become more meaningful for token-level tasks.
    
    Use case: Single self-attention layer with residual and norm.
    Note: This is just the attention sub-layer, not a full transformer block.
    """
    def __init__(
        self, 
        d_model: int, 
        n_heads: int = 8, 
        dropout: float = 0.1,
        use_positional_encoding: bool = False,
        max_seq_len: int = 512,
        num_out_tokens: int = 10,
        **kwargs
    ):
        super().__init__()
        
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.num_out_tokens = num_out_tokens
        
        # Standard self-attention projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
        # Layer norm (pre-norm style)
        self.norm = nn.LayerNorm(d_model)
        
        # Optional: Learned positional embeddings
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            self.pos_embedding = nn.Embedding(max_seq_len, d_model)
    
    def forward(self, tokens: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            tokens: [B, L, d_model] - Encoder token embeddings
            mask: [B, L] - Attention mask (1 for valid, 0 for padding)
        Returns:
            refined_tokens: [B, L, d_model]
        """
        B, L, d_model = tokens.shape
        residual = tokens
        
        # Add positional encoding if enabled
        if self.use_positional_encoding:
            positions = torch.arange(L, device=tokens.device).unsqueeze(0).expand(B, -1)
            tokens = tokens + self.pos_embedding(positions)
        
        # Pre-norm
        tokens = self.norm(tokens)
        
        # Project Q, K, V
        Q = self.q_proj(tokens).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(tokens).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(tokens).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        # Q, K, V: [B, n_heads, L, head_dim]
        
        # Compute attention scores
        attn_scores = (Q @ K.transpose(-2, -1)) / self.scale  # [B, n_heads, L, L]
        
        # Apply mask if provided
        if mask is not None:
            # mask: [B, L] → [B, 1, 1, L]
            mask_expanded = mask[:, None, None, :]
            attn_scores = attn_scores.masked_fill(mask_expanded == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, n_heads, L, L]
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_out = attn_weights @ V  # [B, n_heads, L, head_dim]
        
        # Concatenate heads
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, d_model)
        
        # Output projection
        attn_out = self.out_proj(attn_out)
        attn_out = self.dropout(attn_out)
        
        # Residual connection
        residual = residual + attn_out
        return residual[:, :self.num_out_tokens, :]


class TokenTransformerBlock(nn.Module):
    """
    Full transformer encoder block with self-attention + FFN.
    This is a complete building block that can be stacked.
    
    Use case: Single transformer layer with both attention and FFN sub-layers.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_ff: int = None,  # If None, defaults to 4 * d_model
        dropout: float = 0.1,
        use_positional_encoding: bool = False,
        max_seq_len: int = 512,
        **kwargs
    ):
        super().__init__()
        
        if d_ff is None:
            d_ff = d_model * 4
        
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        # Self-attention sub-layer
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.attn_out_proj = nn.Linear(d_model, d_model)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
        # Layer norm after attention
        self.norm1 = nn.LayerNorm(d_model)
        
        # FFN sub-layer
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer norm after FFN
        self.norm2 = nn.LayerNorm(d_model)
        
        # Optional: Learned positional embeddings
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            self.pos_embedding = nn.Embedding(max_seq_len, d_model)
    
    def forward(self, tokens: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            tokens: [B, L, d_model]
            mask: [B, L] - Attention mask (1 for valid, 0 for padding)
        Returns:
            output: [B, L, d_model]
        """
        B, L, d_model = tokens.shape
        
        # Add positional encoding if enabled (only in first block ideally)
        if self.use_positional_encoding:
            positions = torch.arange(L, device=tokens.device).unsqueeze(0).expand(B, -1)
            tokens = tokens + self.pos_embedding(positions)
        
        # ===== Self-Attention Sub-layer =====
        residual = tokens
        
        # Pre-norm
        tokens_norm = self.norm1(tokens)
        
        # Project Q, K, V
        Q = self.q_proj(tokens_norm).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(tokens_norm).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(tokens_norm).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attn_scores = (Q @ K.transpose(-2, -1)) / self.scale
        
        # Apply mask
        if mask is not None:
            mask_expanded = mask[:, None, None, :]
            attn_scores = attn_scores.masked_fill(mask_expanded == 0, float('-inf'))
        
        # Attention weights and output
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_out = attn_weights @ V
        
        # Concatenate heads and project
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, d_model)
        attn_out = self.attn_out_proj(attn_out)
        attn_out = self.attn_dropout(attn_out)
        
        # Residual connection
        tokens = residual + attn_out
        
        # ===== FFN Sub-layer =====
        residual = tokens
        
        # Pre-norm
        tokens_norm = self.norm2(tokens)
        
        # FFN
        ffn_out = self.ffn(tokens_norm)
        
        # Residual connection
        tokens = residual + ffn_out
        
        return tokens


class TokenTransformerAdapter(nn.Module):
    """
    Stack of multiple transformer encoder blocks.
    This is the most powerful adapter - essentially adds transformer layers
    on top of frozen encoder outputs.
    
    Use case: When you need maximum representational power for token adaptation.
    Can stack N layers for deeper processing.
    """
    def __init__(
        self,
        d_model: int,
        n_layers: int = 2,
        n_heads: int = 8,
        d_ff: int = None,
        dropout: float = 0.1,
        use_positional_encoding: bool = False,
        max_seq_len: int = 512,
        num_out_tokens: int = 10,
        **kwargs
    ):
        """
        Args:
            d_model: Model dimension
            n_layers: Number of transformer blocks to stack
            n_heads: Number of attention heads per block
            d_ff: FFN hidden dimension (default: 4 * d_model)
            dropout: Dropout probability
            use_positional_encoding: Whether to add learned positional embeddings
            max_seq_len: Maximum sequence length for positional embeddings
        """
        super().__init__()
        
        self.n_layers = n_layers
        self.num_out_tokens = num_out_tokens
        
        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TokenTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                # Only add positional encoding in first block
                use_positional_encoding=(use_positional_encoding and i == 0),
                max_seq_len=max_seq_len
            )
            for i in range(n_layers)
        ])
        
        # Final layer norm (standard in transformers)
        self.final_norm = nn.LayerNorm(d_model)
    
    def forward(self, tokens: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            tokens: [B, L, d_model] - Frozen encoder outputs
            mask: [B, L] - Attention mask (1 for valid, 0 for padding)
        Returns:
            adapted_tokens: [B, L, d_model] - Refined token representations
        """
        # Pass through all transformer blocks
        for block in self.blocks:
            tokens = block(tokens, mask)
        
        # Final normalization
        tokens = self.final_norm(tokens)
        
        return tokens[:, :self.num_out_tokens, :]


# ===== Factory function for easy instantiation =====

def get_token_mixer(
    mixer_type: str,
    d_model: int,
    n_heads: int = 8,
    n_layers: int = 1,
    dropout: float = 0.1,
    expansion_factor: int = 2,
    use_positional_encoding: bool = False,
    max_seq_len: int = 512,
    **kwargs
):
    """
    Factory function to get the appropriate token mixer.
    
    Args:
        mixer_type: One of ["none", "linear", "mlp", "self_attn", "transformer"]
        d_model: Model dimension
        n_heads: Number of attention heads (for attention-based mixers)
        n_layers: Number of layers (for transformer mixer)
        dropout: Dropout probability
        expansion_factor: Hidden dim multiplier for MLP
        use_positional_encoding: Whether to add positional embeddings
        max_seq_len: Maximum sequence length
    
    Returns:
        Token mixer module
    """
    if mixer_type == "none" or mixer_type == "identity":
        return IdentityTokenMixer(d_model=d_model)
    
    elif mixer_type == "linear":
        return LinearTokenMixer(d_model=d_model, use_bias=True)
    
    elif mixer_type == "mlp":
        return TokenMLPAdapter(
            d_model=d_model,
            expansion_factor=expansion_factor,
            dropout=dropout
        )
    
    elif mixer_type == "self_attn":
        return TokenSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            use_positional_encoding=use_positional_encoding,
            max_seq_len=max_seq_len
        )
    
    elif mixer_type == "transformer":
        d_ff = kwargs.get('d_ff', d_model * 4)
        return TokenTransformerAdapter(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            use_positional_encoding=use_positional_encoding,
            max_seq_len=max_seq_len
        )
    
    else:
        raise ValueError(
            f"Unknown mixer_type: {mixer_type}. "
            f"Choose from ['none', 'linear', 'mlp', 'self_attn', 'transformer']"
        )


# ===== Convenience dict for mapping strings to classes =====
TOKEN_MIXER_DICT = {
    "none": IdentityTokenMixer,
    "identity": IdentityTokenMixer,
    "linear": LinearTokenMixer,
    "mlp": TokenMLPAdapter,
    "self_attn": TokenSelfAttention,
    "transformer": TokenTransformerAdapter,
}
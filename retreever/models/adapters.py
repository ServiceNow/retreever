"""
Adapter modules for encoder finetuning in ReTreever.

Supports three strategies:
1. shared_mlp_zero_init_norm: Shared MLP adapter with zero init and normalization
2. shared_linear_zero_init_norm: Shared linear adapter with zero init and normalization  
3. mrl: Matryoshka Representation Learning adapter
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPAdapterWithZeroInitNorm(nn.Module):
    """MLP adapter with 4x expansion, zero init, and L2 normalization.
    
    Used for shared_mlp_zero_init_norm strategy.
    Starts as identity function and learns gradually.
    """
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        hidden_dim = input_dim * 4
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # Zero initialization - starts as identity
        torch.nn.init.zeros_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)
        
    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        # L2 normalization of output
        return F.normalize(residual + x, p=2, dim=-1)


class LinearAdapterWithZeroInitNorm(nn.Module):
    """Linear adapter with zero init and L2 normalization.
    
    Used for shared_linear_zero_init_norm strategy.
    Lighter than MLP, starts as identity function.
    """
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
        
        # Zero initialization - starts as identity
        torch.nn.init.zeros_(self.projection.weight)
        torch.nn.init.zeros_(self.projection.bias)
        
    def forward(self, x):
        # L2 normalization of output
        return F.normalize(x + self.projection(x), p=2, dim=-1)


class MRLAdapter(nn.Module):
    """Adapter for Matryoshka Representation Learning (MRL).
    
    Used for mrl strategy.
    Simple linear projection without residual connection.
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.projection(x)


# Supported adapter types
ADAPTER_REGISTRY = {
    "shared_mlp_zero_init_norm": MLPAdapterWithZeroInitNorm,
    "shared_linear_zero_init_norm": LinearAdapterWithZeroInitNorm,
    "mrl": MRLAdapter,
}


def get_adapter(strategy: str, input_dim: int, output_dim: int, dropout: float = 0.1):
    """Factory function to create adapter based on strategy.
    
    Args:
        strategy: One of "shared_mlp_zero_init_norm", "shared_linear_zero_init_norm", "mrl"
        input_dim: Input dimension
        output_dim: Output dimension
        dropout: Dropout rate (ignored for mrl)
        
    Returns:
        Adapter module instance
        
    Raises:
        ValueError: If strategy is not supported
    """
    if strategy not in ADAPTER_REGISTRY:
        raise ValueError(
            f"Unsupported finetune strategy: {strategy}. "
            f"Supported strategies: {list(ADAPTER_REGISTRY.keys())}"
        )
    
    adapter_class = ADAPTER_REGISTRY[strategy]
    
    # MRLAdapter doesn't take dropout
    if strategy == "mrl":
        return adapter_class(input_dim, output_dim)
    else:
        return adapter_class(input_dim, output_dim, dropout)

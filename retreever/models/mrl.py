"""Matryoshka Representation Learning (MRL) model using ReTreever architecture."""

import torch
from omegaconf import OmegaConf
from typing import Callable, Optional

from retreever.models.retreever import ReTreever


def load_from_ckpt(
    ckpt_path: str,
    cfg_path: str,
    cache_dir: str = None,
    **kwargs
):
    """
    Load MRL model from checkpoint.

    Args:
        ckpt_path: Path to .bin checkpoint
        cfg_path: Path to .yaml configuration file
        cache_dir: Path to cache directory (None uses HF defaults)
        **kwargs: Additional keyword arguments

    Returns:
        Loaded MRL model and config
    """
    # Load checkpoint
    checkpoint = torch.load(ckpt_path)

    # Remove loss parameters not needed for inference
    checkpoint.pop("loss.criterion.temp_param.temp_coef", None)
    checkpoint.pop("loss.temp_param.temp_coef", None)

    # Load configuration
    cfg = OmegaConf.load(cfg_path)

    # Set evaluation representation size
    eval_level = kwargs.get("rep_level", None)
    if eval_level is None:
        eval_level = 10

    # Instantiate model
    model = MRL(
        loss=None,
        encoder_type=cfg.model.encoder_type,
        freeze_encoder=cfg.model.freeze_encoder,
        cache_dir=cache_dir,
        dual_model=cfg.model.dual_model,
        tree_split_fn=cfg.model.tree_split_fn,
        encoder_token_level=cfg.model.encoder_token_level,
        encoder_normalize=cfg.model.encoder_normalize,
        encoder_context_length=cfg.model.encoder_context_length,
        embedding_dim=cfg.model.get("split_fn_embedding_dim", 768),
        num_embeddings_per_node=cfg.model.get("num_embeddings_per_node", 1),
        scoring_fn_name=cfg.model.get("cross_attn_scoring_fn_name", "linear_then_mean"),
        d_k=cfg.model.get("split_fn_d_k", 768),
        n_heads=cfg.model.get("split_fn_n_heads", 12),
        eval_depth=eval_level,
    )
    model.load_state_dict(checkpoint)

    return model, cfg


class MRL(ReTreever):
    """
    Matryoshka Representation Learning model.
    
    Uses ReTreever architecture with a flattened tree and non-probabilistic outputs.
    Allows learning representations at multiple granularities.
    """

    def __init__(
        self,
        encoder_type: str = "bge",
        loss: Callable = None,
        cache_dir: str = None,
        dual_model: bool = False,
        freeze_encoder: bool = True,
        emb_size: int = None,
        **module_params,
    ):
        """
        Initialize MRL model.
        
        Args:
            encoder_type: Type of encoder ('bge', 'distilbert', etc.)
            loss: Loss function
            cache_dir: Cache directory for models
            dual_model: Whether to use separate query/context encoders
            freeze_encoder: Whether to freeze encoder weights
            emb_size: Embedding size (auto-detected if None)
            **module_params: Additional model parameters
        """
        module_params["tree_depth"] = 10
        module_params["index_distance"] = "angular"

        super(MRL, self).__init__(
            encoder_type=encoder_type,
            tree_type="no_tree",
            loss=loss,
            cache_dir=cache_dir,
            dual_model=dual_model,
            freeze_encoder=freeze_encoder,
            emb_size=emb_size,
            eval_strategy="faiss_tree_rep",
            **module_params,
        )

    def encode_sentences(
        self,
        sentences: torch.Tensor,
        tag: str = "query",
        rep_level: Optional[int] = None,
        device: str = "cpu",
        *args,
        **kwargs,
    ):
        """
        Encode sentences at specified representation level.
        
        Used for MTEB evaluation and other downstream tasks.
        
        Args:
            sentences: Input sentences to encode
            tag: 'query' or 'context'
            rep_level: Representation level (truncates embedding to 2^rep_level dims)
            device: Device to use
            
        Returns:
            Encoded representations
        """
        encodings = super().encode_sentences(sentences, tag, None, device)

        if rep_level is not None:
            return encodings[:, : 2**rep_level]

        return encodings

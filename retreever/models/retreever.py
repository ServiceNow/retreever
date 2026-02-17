import torch

from omegaconf import OmegaConf
from typing import Callable, Optional

from retreever import config
from retreever.models.encoders import get_encoders
from retreever.models.trees import tree_dict
from retreever.models.indexing_strategies import index_strategy_dict
import torch.nn.functional as F


def freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False

def load_from_ckpt(ckpt_path: str, cfg_path: str, cache_dir: str = None, **kwargs):
    """Loads a Retreever model given a checkpoint and a configuration.

    Args:
        ckpt_path (str): path to .bin checkpoint
        cfg_path (str): path to .yaml configuration file
        cache_dir (str, optional): path to cache directory. Defaults to None (uses HuggingFace default cache).
        **kwargs: Additional keyword arguments.
            - force_topk_strategy (bool): If provided and True, this is used to overwrite the model's
            existing evaluation strategy with the `topk_strategy` provided here (or 'greedy' as default)


    Returns:
        ReTreever, OmegaConf: Loaded model instance, and the corresponding config
    """

    # load checkpoint
    checkpoint = torch.load(ckpt_path)

    checkpoint.pop(
        "loss.temp_param.temp_coef", None
    )  # as Loss is not instantiated, keeping this parameter would throw an error

    checkpoint.pop(
        "loss.criterion.temp_param.temp_coef", None
    )  # Needed when trained with MRL loss.

    checkpoint.pop("loss.corr_scale", None)

    checkpoint.pop("loss.temp_param-temp_coef", None)

    checkpoint.pop("loss.criterion-temp_param-temp_coef", None)
    
    checkpoint.pop("loss.siglip_bias", None)
    
    to_pop = []
    for k in checkpoint:
        if "temp_coef" in k:
            if k not in ["context_tree.temp_coeff", "query_tree.temp_coeff"]:
                to_pop.append(k)
            
    for k in to_pop:
        print(f"Removing this temp_coef key from checkpoint: {k}")
        checkpoint.pop(k, None)
    
    # rename old keys to new expected names
    rename_map = {
        "context_tree.split.node_embeddings": "context_tree.split.node_embedding_module.node_embeddings",
        "query_tree.split.node_embeddings": "query_tree.split.node_embedding_module.node_embeddings",
    }
    for old, new in rename_map.items():
        if old in checkpoint:
            checkpoint[new] = checkpoint.pop(old)

    # load configuration
    cfg = OmegaConf.load(cfg_path)

    # Decide on eval strategy
    eval_strategy = cfg.model.get("eval_strategy", "greedy")
    print(f"Eval strategy from config: {eval_strategy}")
    if kwargs.get("force_topk_strategy"):
        eval_strategy = kwargs.get("topk_strategy", "greedy")
        print(f"Eval strategy from config replaced by : {eval_strategy}")

    # set evaluation representation size
    eval_depth = kwargs.get("rep_level", None)
    if eval_depth is None:
        eval_depth = cfg.model.tree_depth

    # instantiate model
    cache_dir = cache_dir or config.HF_CACHE_DIR
    model = ReTreever(
        loss=None,
        encoder_type=cfg.model.encoder_type,
        freeze_encoder=cfg.model.freeze_encoder,
        tree_type=cfg.model.tree_type,
        cache_dir=cache_dir,
        dual_model=cfg.model.dual_model,
        tree_depth=cfg.model.tree_depth,
        tree_split_fn=cfg.model.tree_split_fn,
        encoder_token_level=cfg.model.encoder_token_level,
        encoder_normalize=cfg.model.encoder_normalize,
        encoder_context_length=cfg.model.encoder_context_length,
        # TODO: Remove default values when backward compatibility not required anymore
        embedding_dim=cfg.model.get("split_fn_embedding_dim", 768),
        num_embeddings_per_node=cfg.model.get("num_embeddings_per_node", 1),
        scoring_fn_name=cfg.model.get("cross_attn_scoring_fn_name", "linear_then_mean"),
        d_k=cfg.model.get("split_fn_d_k", 768),
        n_heads=cfg.model.get("split_fn_n_heads", 12),
        eval_strategy=eval_strategy,
        eval_depth=eval_depth,
        train_full_tree_rep=cfg.model.get("train_full_tree_rep", False),
        use_sigmoid_in_projection=cfg.model.get("use_sigmoid_in_projection", False),
        enable_llm_split_score_prop=cfg.model.get("enable_llm_split_score_prop", False),
        attend_ne_only_on_text=cfg.model.get("attend_ne_only_on_text", False),
        token_mixer_type=cfg.model.get("token_mixer_type", None),
        node_embedding_strategy=cfg.model.get("node_embedding_strategy", "independent"),
        finetuned_encoder_path=cfg.model.get("finetuned_encoder_path", None),
        treat_attn_as_residual=cfg.model.get("treat_attn_as_residual", False),
        have_ffn_layer=cfg.model.get("have_ffn_layer", False),
        lock_in_query_encoder=cfg.model.get("lock_in_query_encoder", False),
        lock_in_context_encoder=cfg.model.get("lock_in_context_encoder", False),
        vq_residual=cfg.model.get("vq_residual", False),
        use_simvq=cfg.model.get("use_simvq", False),
        dropout_location=cfg.model.get("dropout_location", "inside_split"),
        encoder_finetune_strategy=cfg.model.get("encoder_finetune_strategy", "none"),
        finetuned_encoder_checkpoint=cfg.model.get("finetuned_encoder_checkpoint", None),
        freeze_finetuned_components=cfg.model.get("freeze_finetuned_components", True),
    )
    model.load_state_dict(checkpoint)

    return model, cfg

class LinearAdapter(torch.nn.Module):
    """Linear projection with residual connection."""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.projection = torch.nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return x + self.projection(x)  # Residual connection
    
class LinearAdapterZeroInit(torch.nn.Module):
    """Linear projection with residual connection."""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.projection = torch.nn.Linear(input_dim, output_dim)
        
        # --- FIX: Zero Initialization ---
        # This ensures the adapter starts as an identity function (output = x + 0)
        torch.nn.init.zeros_(self.projection.weight)
        torch.nn.init.zeros_(self.projection.bias)
        
    def forward(self, x):
        return x + self.projection(x)


class MLPAdapter(torch.nn.Module):
    """MLP adapter with 4x expansion and residual connection."""
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        hidden_dim = input_dim * 4
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return residual + x  # Residual connection
    
class MLPAdapterWithZeroInit(torch.nn.Module):
    """MLP adapter with 4x expansion and residual connection."""
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        hidden_dim = input_dim * 4
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

        # --- FIX: Zero Initialization for the LAST layer only ---
        torch.nn.init.zeros_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)
        
    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return residual + x
    
class MLPAdapterWithZeroInitNorm(torch.nn.Module):
    """MLP adapter with 4x expansion and residual connection."""
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        hidden_dim = input_dim * 4
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

        # --- FIX: Zero Initialization for the LAST layer only ---
        torch.nn.init.zeros_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)
        
    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return F.normalize(residual + x, p=2, dim=-1)
    
class LinearAdapterWithZeroInitNorm(torch.nn.Module):
    """MLP adapter with 4x expansion and residual connection."""
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.projection = torch.nn.Linear(input_dim, output_dim)
        
        # This ensures the adapter starts as an identity function (output = x + 0)
        torch.nn.init.zeros_(self.projection.weight)
        torch.nn.init.zeros_(self.projection.bias)
        
    def forward(self, x):
        return F.normalize(x + self.projection(x), p=2, dim=-1)
    

class MRLAdapter(torch.nn.Module):
    """Adapter for MRL learning"""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.projection = torch.nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.projection(x)
    
    
class MLPAdapterWithoutResidual(torch.nn.Module):
    """MLP adapter with 4x expansion and residual connection."""
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        hidden_dim = input_dim * 4
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return  x  # Residual connection


class BottleneckAdapter(torch.nn.Module):
    """Houlsby-style bottleneck adapter with residual."""
    def __init__(self, input_dim: int, bottleneck_dim: int = 64):
        super().__init__()
        self.down_project = torch.nn.Linear(input_dim, bottleneck_dim)
        self.activation = torch.nn.ReLU()
        self.up_project = torch.nn.Linear(bottleneck_dim, input_dim)
        
    def forward(self, x):
        residual = x
        x = self.down_project(x)
        x = self.activation(x)
        x = self.up_project(x)
        return residual + x  # Residual connection (Houlsby-style)
    
class OptimizedBottleneckAdapter(torch.nn.Module):
    """Low-rank bottleneck with zero init."""
    def __init__(self, input_dim: int, reduction_factor: int = 8):
        super().__init__()
        bottleneck_dim = input_dim // reduction_factor
        
        self.down_project = torch.nn.Linear(input_dim, bottleneck_dim)
        self.activation = torch.nn.GELU()
        self.up_project = torch.nn.Linear(bottleneck_dim, input_dim)
        
        # Init Strategy:
        # 1. Kaiming/Xavier for the down projection (to preserve variance)
        torch.nn.init.kaiming_normal_(self.down_project.weight)
        
        # 2. ZERO init for the up projection (to start as identity)
        torch.nn.init.zeros_(self.up_project.weight)
        torch.nn.init.zeros_(self.up_project.bias)
        
    def forward(self, x):
        residual = x
        x = self.down_project(x)
        x = self.activation(x)
        x = self.up_project(x)
        return residual + x

class PreNormAdapter(torch.nn.Module):
    """Applies LayerNorm before the adapter, keeping residual clean."""
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = torch.nn.LayerNorm(input_dim)
        
        hidden_dim = input_dim * 4
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.activation = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        
        # Zero Init the last layer
        torch.nn.init.zeros_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        residual = x
        
        # Norm only affects the adapter branch
        out = self.norm(x)
        out = self.fc1(out)
        out = self.activation(out)
        out = self.fc2(out)
        
        return residual + out

class ReTreever(torch.nn.Module):
    def __init__(
        self,
        encoder_type: str = "bge",
        tree_type: str = "qr_tree",
        loss: Callable = None,
        token_loss_term: Callable = None,
        cache_dir: str = None,
        dual_model: bool = False,
        freeze_encoder: bool = True,
        encoder_finetune_strategy: str = "none",  # Options: "none", "last_layer", "linear", "mlp", "adapter", "bitfit", "layernorm"
        emb_size: int = None,
        eval_strategy: str = "greedy",
        train_full_tree_rep: bool = False,
        **module_params,
    ):
        super(ReTreever, self).__init__()

        self.loss = loss
        self.token_loss_term = token_loss_term
        self.dual_model = dual_model

        self.encoder_type = encoder_type
        self.tree_type = tree_type
        self.train_full_tree_rep = train_full_tree_rep

        # get tree parameters
        self.tree_depth = module_params.pop("tree_depth", 10)
        tree_split_fn = module_params.pop("tree_split_fn", "linear")
        tree_split_fn_params = {}
        if tree_split_fn == "cross_attn":
            tree_split_fn_params = {
                "embedding_dim": module_params.pop("embedding_dim", 512),
                "num_embeddings_per_node": module_params.pop("num_embeddings_per_node", 10),
                "d_k": module_params.pop("d_k", 768),
                "n_heads": module_params.pop("n_heads", 12),
                "scoring_fn_name": module_params.pop("scoring_fn_name", "linear_then_mean"),
                "token_mixer_type": module_params.pop("token_mixer_type", None),
                "node_embedding_strategy": module_params.pop("node_embedding_strategy", "independent"),
                "treat_attn_as_residual": module_params.pop("treat_attn_as_residual", False),
                "have_ffn_layer": module_params.pop("have_ffn_layer", False),
                "vq_residual": module_params.pop("vq_residual", False),
                "use_simvq": module_params.pop("use_simvq", False),
                "dropout_location": module_params.pop("dropout_location", "inside_split"),
            }
        if tree_split_fn == "llm_split":
             tree_split_fn_params = {
                 "use_sigmoid_in_projection": module_params.pop("use_sigmoid_in_projection", False),
                 "attend_ne_only_on_text": module_params.pop("attend_ne_only_on_text", False),
                 "enable_llm_split_score_prop": module_params.pop("enable_llm_split_score_prop", False),
             }
             
        # Adapter/finetuning hyperparameters
        adapter_bottleneck_dim = module_params.pop("adapter_bottleneck_dim", 64)
        mlp_dropout = module_params.pop("mlp_dropout", 0.1)

        # get encoder parameters
        self.token_level_enc = module_params.pop("encoder_token_level", False)
        normalize_emb = module_params.pop("encoder_normalize", False)
        max_length = module_params.pop("encoder_context_length", 512)

        # The only split_fn that supports self.token_level_enc is cross_attn
        if self.token_level_enc and tree_split_fn not in  ["cross_attn", "transformer_encoder", "llm_split"]:
            raise ValueError(f"Token level encoding is not supported for {tree_split_fn} splits")

        # instantiate separate context encoder and tree
        cache_dir = cache_dir or config.HF_CACHE_DIR
        self.query_encoder, self.context_encoder = get_encoders(
            self.encoder_type,
            cache_dir=cache_dir,
            token_level=self.token_level_enc,
            normalize=normalize_emb,
            max_length=max_length,
        )
        
        # lockin flags
        self.lock_in_query_encoder = module_params.pop("lock_in_query_encoder", False)
        self.lock_in_context_encoder = module_params.pop("lock_in_context_encoder", False)

        if emb_size is None:
            emb_size = self.context_encoder.output_size
            
        # Store checkpoint loading info for later (after projection layers created)
        finetuned_encoder_checkpoint = module_params.pop("finetuned_encoder_checkpoint", None)
        freeze_finetuned_components = module_params.pop("freeze_finetuned_components", True)
        
        if finetuned_encoder_checkpoint is not None:
            self._pending_checkpoint_load = (finetuned_encoder_checkpoint, freeze_finetuned_components)
        else:
            self._pending_checkpoint_load = None

        self.context_tree = tree_dict[self.tree_type](
            input_size=emb_size,
            depth=self.tree_depth,
            split_fn=tree_split_fn,
            **tree_split_fn_params,
        )

        if dual_model:
            # train one tree for routing questions and one for contexts
            self.query_tree = tree_dict[self.tree_type](
                input_size=emb_size,
                depth=self.tree_depth,
                split_fn=tree_split_fn,
                **tree_split_fn_params,
            )
        else:
            # use same tree for question and context
            self.query_tree = self.context_tree

        # if freeze_encoder:
        #     freeze_module(self.query_encoder)
        #     freeze_module(self.context_encoder)
        # else:
        #     # Freeze entire encoder first
        #     freeze_module(self.query_encoder)
        #     freeze_module(self.context_encoder)
            
        #     # Unfreeze only the last transformer layer
        #     encoders_to_finetune = [self.context_encoder, self.query_encoder]
        #     if self.lock_in_query_encoder:
        #         encoders_to_finetune = [self.context_encoder]
        #     elif self.lock_in_context_encoder:
        #         encoders_to_finetune = [self.query_encoder]
                
        #     for encoder in encoders_to_finetune:
        #         if hasattr(encoder.model, 'encoder') and hasattr(encoder.model.encoder, 'layer'):
        #             # For BERT/BGE/DPR-style models
        #             for layer in encoder.model.encoder.layer[-3:]:
        #                 for param in layer.parameters():
        #                     param.requires_grad = True
        
        # Initialize projection/adapter layers
        self.query_projection = None
        self.context_projection = None

        if freeze_encoder:
            # Freeze encoders completely
            freeze_module(self.query_encoder)
            freeze_module(self.context_encoder)
            
        else:
            # freeze_encoder=False: Apply finetuning strategy
            
            
            encoders_to_finetune = [self.query_encoder, self.context_encoder]
            if self.lock_in_query_encoder:
                encoders_to_finetune = [self.context_encoder]
            elif self.lock_in_context_encoder:
                encoders_to_finetune = [self.query_encoder]
            
            if encoder_finetune_strategy == "none":
                # No finetuning strategy specified - raise error
                raise ValueError(
                    "freeze_encoder=False but no encoder_finetune_strategy specified. "
                    "Choose from: 'last_layer', 'linear', 'mlp', 'adapter', 'bitfit', 'layernorm'"
                )
            
            elif encoder_finetune_strategy == "last_layer":
                # Freeze entire encoder first
                freeze_module(self.query_encoder)
                freeze_module(self.context_encoder)
                    
                for encoder in encoders_to_finetune:
                    # Check if this is ResNet
                    if hasattr(encoder, '__class__') and 'ResNet' in encoder.__class__.__name__:
                        # For ResNet: unfreeze the last sequential block
                        # ResNet is wrapped as nn.Sequential, so access the last child
                        last_layer = list(encoder.model.children())[-1]
                        for param in last_layer.parameters():
                            param.requires_grad = True
                        print(f"Unfroze last layer for ResNet encoder")
                        continue
                    
                    # Try common transformer layer paths
                    layers = None
                    
                    # BERT/RoBERTa/BGE/Contriever/SimCSE: model.encoder.layer
                    if hasattr(encoder.model, 'encoder') and hasattr(encoder.model.encoder, 'layer'):
                        layers = encoder.model.encoder.layer
                    # DistilBERT: model.transformer.layer
                    elif hasattr(encoder.model, 'transformer') and hasattr(encoder.model.transformer, 'layer'):
                        layers = encoder.model.transformer.layer
                    # Direct access: model.layer (some custom wrappers)
                    elif hasattr(encoder.model, 'layer'):
                        layers = encoder.model.layer
                    
                    if layers is not None:
                        # Unfreeze last layer
                        for param in layers[-1].parameters():
                            param.requires_grad = True
                        print(f"Unfroze last layer for {encoder.__class__.__name__}")
                    else:
                        raise ValueError(
                            f"Could not find transformer layers in {encoder.__class__.__name__}. "
                            f"Model structure: {type(encoder.model)}"
                        )
                    
            elif encoder_finetune_strategy == "linear":
                # Freeze encoders, add linear adapter on top
                freeze_module(self.query_encoder)
                freeze_module(self.context_encoder)
                
                self.query_projection = LinearAdapter(emb_size, emb_size)
                self.context_projection = LinearAdapter(emb_size, emb_size)

            elif encoder_finetune_strategy == "linear_zero_init":
                # Freeze encoders, add linear adapter on top
                freeze_module(self.query_encoder)
                freeze_module(self.context_encoder)
                
                self.query_projection = LinearAdapterZeroInit(emb_size, emb_size)
                self.context_projection = LinearAdapterZeroInit(emb_size, emb_size)
                
            elif encoder_finetune_strategy == "mlp":
                # Freeze encoders, add MLP adapter on top
                freeze_module(self.query_encoder)
                freeze_module(self.context_encoder)
                
                self.query_projection = MLPAdapter(emb_size, emb_size, dropout=mlp_dropout)
                self.context_projection = MLPAdapter(emb_size, emb_size, dropout=mlp_dropout)
                
            elif encoder_finetune_strategy == "mlp_no_residual":
                # Freeze encoders, add MLP adapter on top
                freeze_module(self.query_encoder)
                freeze_module(self.context_encoder)
                
                self.query_projection = MLPAdapterWithoutResidual(emb_size, emb_size, dropout=mlp_dropout)
                self.context_projection = MLPAdapterWithoutResidual(emb_size, emb_size, dropout=mlp_dropout)
                
            elif encoder_finetune_strategy == "mlp_zero_init":
                # Freeze encoders, add MLP adapter on top
                freeze_module(self.query_encoder)
                freeze_module(self.context_encoder)
                
                self.query_projection = MLPAdapterWithZeroInit(emb_size, emb_size, dropout=mlp_dropout)
                self.context_projection = MLPAdapterWithZeroInit(emb_size, emb_size, dropout=mlp_dropout)
                
            elif encoder_finetune_strategy == "shared_mlp_zero_init":
                # Freeze encoders, add MLP adapter on top
                freeze_module(self.query_encoder)
                freeze_module(self.context_encoder)
                adapter = MLPAdapterWithZeroInit(emb_size, emb_size, dropout=mlp_dropout)
                
                self.query_projection = adapter
                self.context_projection = adapter    
                            
            elif encoder_finetune_strategy == "shared_mlp_zero_init_norm":
                # Freeze encoders, add MLP adapter on top
                freeze_module(self.query_encoder)
                freeze_module(self.context_encoder)
                adapter = MLPAdapterWithZeroInitNorm(emb_size, emb_size, dropout=mlp_dropout)
                
                self.query_projection = adapter
                self.context_projection = adapter
                
            elif encoder_finetune_strategy == "shared_linear_zero_init_norm":
                # Freeze encoders, add MLP adapter on top
                freeze_module(self.query_encoder)
                freeze_module(self.context_encoder)
                adapter = LinearAdapterWithZeroInitNorm(emb_size, emb_size, dropout=mlp_dropout)
                
                self.query_projection = adapter
                self.context_projection = adapter
                
            elif encoder_finetune_strategy == "pre_norm_mlp":
                # Freeze encoders, add MLP adapter on top
                freeze_module(self.query_encoder)
                freeze_module(self.context_encoder)
                
                self.query_projection = PreNormAdapter(emb_size, emb_size, dropout=mlp_dropout)
                self.context_projection = PreNormAdapter(emb_size, emb_size, dropout=mlp_dropout)
                
            elif encoder_finetune_strategy == "mrl":
                # Freeze encoders, add MLP adapter on top
                freeze_module(self.query_encoder)
                freeze_module(self.context_encoder)
                
                adapter = MRLAdapter(emb_size, emb_size)
                
                self.query_projection = adapter
                self.context_projection = adapter
                
            elif encoder_finetune_strategy == "bottleneck":
                # Freeze encoders, add MLP adapter on top
                freeze_module(self.query_encoder)
                freeze_module(self.context_encoder)
                
                self.query_projection = OptimizedBottleneckAdapter(emb_size)
                self.context_projection = OptimizedBottleneckAdapter(emb_size)
                
            elif encoder_finetune_strategy == "adapter":
                # Freeze encoders, add Houlsby-style bottleneck adapter
                freeze_module(self.query_encoder)
                freeze_module(self.context_encoder)
                
                self.query_projection = BottleneckAdapter(emb_size, bottleneck_dim=adapter_bottleneck_dim)
                self.context_projection = BottleneckAdapter(emb_size, bottleneck_dim=adapter_bottleneck_dim)
                
            elif encoder_finetune_strategy == "bitfit":
                # Freeze all except bias parameters
                freeze_module(self.query_encoder)
                freeze_module(self.context_encoder)
                
                for encoder in encoders_to_finetune:
                    for name, param in encoder.named_parameters():
                        if 'bias' in name:
                            param.requires_grad = True
                            
            elif encoder_finetune_strategy == "layernorm":
                # Freeze all except LayerNorm parameters
                freeze_module(self.query_encoder)
                freeze_module(self.context_encoder)
                
                for encoder in encoders_to_finetune:
                    for name, param in encoder.named_parameters():
                        if 'LayerNorm' in name or 'layer_norm' in name:
                            param.requires_grad = True
                            
            else:
                raise ValueError(f"Unknown encoder_finetune_strategy: {encoder_finetune_strategy}")
            
        # Load from checkpoint if provided (must happen after LoRA is applied)
        if self._pending_checkpoint_load is not None:
            checkpoint_path, freeze_components = self._pending_checkpoint_load
            
            # # Validate strategy consistency
            # if encoder_finetune_strategy not in ["lora", "last_layer"]:
            #     raise ValueError(
            #         f"finetuned_encoder_checkpoint only supports 'lora' or 'last_layer', "
            #         f"got encoder_finetune_strategy='{encoder_finetune_strategy}'"
            #     )
            
            # loaded_strategy = self._load_finetuned_encoder_from_checkpoint(
            #     checkpoint_path, freeze_components
            # )
            
            # # Warn if strategies don't match
            # if loaded_strategy != encoder_finetune_strategy:
            #     print(f"WARNING: encoder_finetune_strategy='{encoder_finetune_strategy}' "
            #           f"but checkpoint contains '{loaded_strategy}' finetuning")
            if encoder_finetune_strategy in ["shared_mlp_zero_init_norm"]:
                self.load_and_freeze_shared_mlp(checkpoint_path)
            
            del self._pending_checkpoint_load
            
        # if self.query_projection is not None:
        #     self.query_projection = self.query_projection.to(torch.float32)
        # if self.context_projection is not None:
        #     self.context_projection = self.context_projection.to(torch.float32)
                            
        # Print trainable parameter counts
        for encoder_name, encoder in [('Query', self.query_encoder), ('Context', self.context_encoder)]:
            total_params = sum(p.numel() for p in encoder.parameters())
            trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
            print(f"{encoder_name} Encoder: {trainable_params:,} / {total_params:,} trainable parameters ({100 * trainable_params / total_params:.2f}%)")

        self.eval_strategy = eval_strategy

        # Instantiate the appropriate indexing strategy
        index_distance = module_params.pop("index_distance", "manhattan")
        self.eval_depth = module_params.pop("eval_depth", self.tree_depth)
        self.indexing_strategy = index_strategy_dict[self.eval_strategy](
            2**self.eval_depth if self.tree_type != "identity_tree" else emb_size,
            index_distance,
        )

        # structure to index contexts. init to empty
        self.reset_index()
        
    def _apply_projection(self, embeddings, projection_layer):
        """Apply projection/adapter layer if it exists."""
        if projection_layer is not None:
            # Handle both [batch, hidden] and [batch, seq, hidden] shapes
            if next(projection_layer.parameters()).dtype != embeddings.dtype:
                projection_layer = projection_layer.to(embeddings.dtype)
            original_shape = embeddings.shape
            if len(original_shape) == 3:
                # Token-level: [batch, seq, hidden]
                batch, seq, hidden = original_shape
                embeddings = embeddings.reshape(-1, hidden)
                embeddings = projection_layer(embeddings)
                out_dim = embeddings.shape[-1]
                embeddings = embeddings.reshape(batch, seq, out_dim)
            else:
                # Sentence-level: [batch, hidden]
                embeddings = projection_layer(embeddings)
        return embeddings
    
    def _load_finetuned_encoder_from_checkpoint(self, checkpoint_path, freeze_components):
        """Load encoder finetuning from checkpoint. Supports only LoRA and last_layer."""
        
        print(f"Loading finetuned encoder from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Detect finetuning strategy from checkpoint keys
        has_lora = any('lora' in key.lower() for key in checkpoint.keys())
        
        if has_lora:
            detected_strategy = "lora"
            print("Detected LoRA finetuning in checkpoint")
            
            if freeze_components:
                # Merge LoRA weights into base model
                print("Merging LoRA weights into base encoder...")
                self.query_encoder, self.context_encoder = merge_lora_weights(
                    checkpoint_path, self.query_encoder, self.context_encoder
                )
                # Freeze everything
                freeze_module(self.query_encoder)
                freeze_module(self.context_encoder)
                print("Merged and froze LoRA weights")
            else:
                # Keep as LoRA model for continued training
                from peft import PeftModel
                
                # Load full encoder state (includes LoRA adapters)
                query_encoder_state = {k.replace('query_encoder.', ''): v 
                                    for k, v in checkpoint.items() 
                                    if k.startswith('query_encoder.')}
                context_encoder_state = {k.replace('context_encoder.', ''): v 
                                        for k, v in checkpoint.items() 
                                        if k.startswith('context_encoder.')}
                
                if query_encoder_state:
                    self.query_encoder.model.load_state_dict(query_encoder_state, strict=False)
                if context_encoder_state:
                    self.context_encoder.model.load_state_dict(context_encoder_state, strict=False)
                
                # SAFE APPROACH: Freeze everything first
                freeze_module(self.query_encoder)
                freeze_module(self.context_encoder)
                
                # Then unfreeze ONLY LoRA parameters
                for encoder in [self.query_encoder, self.context_encoder]:
                    for name, param in encoder.named_parameters():
                        if 'lora' in name.lower():
                            param.requires_grad = True
                print("LoRA weights loaded, base encoder frozen, LoRA parameters unfrozen for training")
        
        else:
            # Last layer finetuning
            detected_strategy = "last_layer"
            print("Detected last_layer finetuning in checkpoint")
            
            # Load encoder weights
            query_encoder_state = {k.replace('query_encoder.', ''): v 
                                for k, v in checkpoint.items() 
                                if k.startswith('query_encoder.')}
            context_encoder_state = {k.replace('context_encoder.', ''): v 
                                    for k, v in checkpoint.items() 
                                    if k.startswith('context_encoder.')}
            
            encoders_loaded = []
            if query_encoder_state:
                self.query_encoder.load_state_dict(query_encoder_state, strict=True)
                encoders_loaded.append(self.query_encoder)
            if context_encoder_state:
                self.context_encoder.load_state_dict(context_encoder_state, strict=True)
                encoders_loaded.append(self.context_encoder)
            
            # SAFE APPROACH: Freeze everything first
            freeze_module(self.query_encoder)
            freeze_module(self.context_encoder)
            
            if not freeze_components:
                # Unfreeze ONLY the last layer
                for encoder in encoders_loaded:
                    layers = None
                    
                    # Try common transformer layer paths
                    if hasattr(encoder.model, 'encoder') and hasattr(encoder.model.encoder, 'layer'):
                        layers = encoder.model.encoder.layer
                    elif hasattr(encoder.model, 'transformer') and hasattr(encoder.model.transformer, 'layer'):
                        layers = encoder.model.transformer.layer
                    elif hasattr(encoder.model, 'layer'):
                        layers = encoder.model.layer
                    
                    if layers is not None:
                        for param in layers[-1].parameters():
                            param.requires_grad = True
                        print(f"Loaded last_layer checkpoint, unfroze last layer of {encoder.__class__.__name__}")
                print("Base encoder frozen, last layer unfrozen for continued training")
            else:
                print("Loaded last_layer checkpoint, everything frozen")
        
        return detected_strategy

    def load_and_freeze_shared_mlp(self, checkpoint_path: str):
        """Load shared MLP projection from checkpoint and freeze it."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle both checkpoint dict and direct state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        mlp_state_dict = {k: v for k, v in state_dict.items() 
                        if 'query_projection' in k or 'context_projection' in k}
        self.load_state_dict(mlp_state_dict, strict=False)
        
        if hasattr(self, 'query_projection') and self.query_projection is not None:
            for param in self.query_projection.parameters():
                param.requires_grad = False
            print(f"Loaded and froze query_projection/context_projection from {checkpoint_path}")
            
    def encode_sentences(
        self,
        sentences: torch.Tensor,
        tag: str = "query",
        rep_level: Optional[int] = None,
        device: str = "cpu",
    ):
        """Needed for MTEB evaluation."""

        if rep_level is None:
            rep_level = self.tree_depth

        if tag == "query":
            tokens = self.query_encoder.tokenizer(
                [self.query_encoder.prefix + s for s in sentences],
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)

            embeddings = self.query_encoder(
                tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                output_attentions=False,
            )
            embeddings = self._apply_projection(embeddings, self.query_projection)
            assignments = self.query_tree(embeddings, tokens["attention_mask"], rep_level)

        elif tag == "passage":
            tokens = self.context_encoder.tokenizer(
                [self.context_encoder.prefix + s for s in sentences],
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)

            embeddings = self.context_encoder(
                tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                output_attentions=False,
            )
            embeddings = self._apply_projection(embeddings, self.context_projection)
            assignments = self.context_tree(embeddings, tokens["attention_mask"], rep_level)

        return assignments

    def forward(
        self,
        question_ids: torch.Tensor,
        context_ids: torch.Tensor,
        question_attn_mask: torch.Tensor,
        context_attn_mask: torch.Tensor,
        enc_output_attentions: bool = False,
        depth: Optional[int] = None,
        return_loss: bool = True,
        **kwargs,
    ):
        """Encode queries and contexts, route them through the tree to get soft leaf assignments, and compute loss based on these assignments.

        Args:
            question_ids (torch.Tensor): query tokens to route and assign to the leaves, of shape (num_questions, seq_length)
            context_ids (torch.Tensor): context tokens to route and assign to the leaves, of shape (num_contexts, seq_length)
            question_attn_mask (torch.Tensor): mask indicating which query tokens to ignore, of shape (num_questions, seq_length)
            context_attn_mask (torch.Tensor): mask indicating which context tokens to ignore, of shape (num_contexts, seq_length)
            enc_output_attentions (bool, optional): whether encoder should output attentions (needed for token-level representations). Defaults to False.
            depth (int, optional): specify the tree depth at which the routings and loss should be computed. Defaults to leaf level. -1 for all tree nodes, at any depth.
            return_loss (bool, optional): whether the loss is returned or not. Defaults to True.

        Returns:
            (torch.float, torch.Tensor, torch.Tensor): loss, query assignments, context assignments
        """
        if depth is None:
            depth = self.tree_depth

        q_embeddings = self.query_encoder(
            question_ids,
            attention_mask=question_attn_mask,
            output_attentions=enc_output_attentions,
        )
        
        if question_attn_mask is None:
            # Image mode: create all-ones mask matching embedding shape
            batch_size = q_embeddings.shape[0]
            seq_len = q_embeddings.shape[1] if len(q_embeddings.shape) == 3 else 1
            question_attn_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=q_embeddings.device)
            
    
        q_embeddings = self._apply_projection(q_embeddings, self.query_projection)
        q_assignments = self.query_tree(q_embeddings, question_attn_mask, -1)
        
        # extra_q_losses = getattr(self.query_tree.split, 'extra_loss', torch.tensor(0.0, device=question_ids.device))

        c_embeddings = self.context_encoder(
            context_ids,
            attention_mask=context_attn_mask,
            output_attentions=enc_output_attentions,
        )
        if context_attn_mask is None:
            # Image mode: create all-ones mask matching embedding shape
            batch_size = c_embeddings.shape[0]
            seq_len = c_embeddings.shape[1] if len(c_embeddings.shape) == 3 else 1
            context_attn_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=c_embeddings.device)
            
        c_embeddings = self._apply_projection(c_embeddings, self.context_projection)
        c_assignments = self.context_tree(c_embeddings, context_attn_mask, -1)
        
        # extra_c_losses = getattr(self.context_tree.split, 'extra_loss', torch.tensor(0.0, device=question_ids.device))

        labels = kwargs.get("label", None)
        
        token_loss = 0
        if self.token_loss_term is not None and return_loss:
            if len(q_embeddings.shape) == 3:
                q_embeddings = q_embeddings.mean(dim=1)
                c_embeddings = c_embeddings.mean(dim=1)
            token_loss = self.token_loss_term(q_embeddings, c_embeddings, torch.tensor(depth, device=question_ids.device), labels=labels)
        
        if return_loss:
            if labels is not None:
                loss = self.loss(q_assignments, c_assignments, torch.tensor(depth, device=question_ids.device), labels=labels) # + extra_q_losses + extra_c_losses
                return loss + token_loss, q_assignments, c_assignments, torch.tensor(depth, device=question_ids.device), labels
            else:
                loss = self.loss(q_assignments, c_assignments, torch.tensor(depth, device=question_ids.device)) # + extra_q_losses + extra_c_losses
                return loss + token_loss, q_assignments, c_assignments, torch.tensor(depth, device=question_ids.device)

        else:
            if labels is not None:
                return q_assignments, c_assignments, torch.tensor(depth, device=question_ids.device), labels
            return q_assignments, c_assignments, torch.tensor(depth, device=question_ids.device)

    def index_ctxs(
        self,
        context_ids: torch.Tensor,
        context_attn_mask: torch.Tensor,
        context_names: torch.Tensor,
        threshold: float = 0.1,
        index_embeddings: bool = False,
    ):
        """Populates dictionary that maps leaf id to the contexts assigned to it (their names and their scores).
        Supports iterative calls to populate index batch by batch.

        Args:
            context_ids (torch.Tensor): tokens to route and assign to the leaves
            context_attn_mask (torch.Tensor): mask indicating which tokens to ignore
            context_names (torch.Tensor): unique context identifiers
            threshold (float): score value above which a context is assigned to a leaf
            level (int): at which depth or representation level to build the index
        """

        # TODO: implement index as a matrix of shape (num_leaves, num_contexts) and with elements the context's leaf assignment scores. The problem with this is that context names might not be integers nor consecutive, and by indexing batch by batch we would need to handle the mapping name to id somehow

        with torch.no_grad():
            # route contexts
            c_embeddings = self.context_encoder(
                context_ids,
                attention_mask=context_attn_mask,
            )
            c_embeddings = self._apply_projection(c_embeddings, self.context_projection)
            c_assignments = self.context_tree(
                c_embeddings,
                context_attn_mask,
                depth=self.eval_depth,
            ).detach()

            self.lowest_score = min(self.lowest_score, torch.min(c_assignments))
            self.highest_score = max(self.highest_score, torch.max(c_assignments))

        to_embed = F.normalize(c_embeddings[:, 0, :], dim=-1) if index_embeddings else c_assignments
        self.indexing_strategy.index_ctxs(to_embed, context_names, threshold=threshold)

    def top_contexts(
        self,
        question_ids: torch.Tensor,
        question_attn_mask: torch.Tensor,
        k: int = 100,
        index_embeddings: bool = False,
    ):
        """Based on the index, returns the top-k contexts with a chosen strategy.

        Args:
            question_ids (_type_): tensor of tokens to route and assign to the leaves, of shape (num_questions, seq_length)
            question_attn_mask (_type_): mask indicating which tokens to ignore, of shape (num_questions, seq_length)
            k (int, optional): number of contexts to return. Defaults to 100.

        Returns:
            (list): list of per question's set of top-k contexts.
        """

        assert (
            not self.indexing_strategy.is_empty()
        ), "Need to populate index with contexts first by calling self.index_ctxs()."

        assert k > 0, "invalid number of top contexts"

        with torch.no_grad():
            # route questions
            q_embeddings = self.query_encoder(
                question_ids,
                attention_mask=question_attn_mask,
            )
            q_embeddings = self._apply_projection(q_embeddings, self.query_projection)
            q_assignments = self.query_tree(
                q_embeddings, question_attn_mask, depth=self.eval_depth
            ).detach()
            
        to_look = F.normalize(q_embeddings[:, 0, :], dim=-1) if index_embeddings else q_assignments

        return self.indexing_strategy.top_contexts(to_look, k)

    def reset_index(self):
        """Reset index. All indexed contexts will be lost."""
        self.lowest_score = 1
        self.highest_score = 0
        return self.indexing_strategy.reset_index()

    def _init_tree_params(
        self,
        question_ids: torch.Tensor,
        context_ids: torch.Tensor,
        question_attn_mask: torch.Tensor,
        context_attn_mask: torch.Tensor,
        enc_output_attentions: bool = False,
        depth=0,
        **kwargs,
    ):
        c_embeddings = self.context_encoder(
            context_ids,
            attention_mask=context_attn_mask,
            output_attentions=enc_output_attentions,
        )
        c_embeddings = self._apply_projection(c_embeddings, self.context_projection)

        q_embeddings = self.query_encoder(
            question_ids,
            attention_mask=question_attn_mask,
            output_attentions=enc_output_attentions,
        )
        q_embeddings = self._apply_projection(q_embeddings, self.query_projection)

        if self.dual_model:
            self.query_tree._init_split_fn(q_embeddings, depth)
            self.context_tree._init_split_fn(c_embeddings, depth)

        elif not self.token_level_enc:
            # if single tree for contexts and queries, concatenate contexts and queries for initialization
            embeddings = torch.vstack((c_embeddings, q_embeddings))
            self.context_tree._init_split_fn(embeddings, depth)

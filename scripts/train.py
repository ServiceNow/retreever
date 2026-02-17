"""Training script for ReTreever models.

Supports:
- Text retrieval (NQ, HotpotQA, RepliQA, TopiocQA)
- Image retrieval (ImageNet1k)  
- Audio retrieval (VoxCeleb2)
- Text-image retrieval (COCO, Flickr30k)

Loss functions:
- Contrastive loss
- Multi-label contrastive loss
- MRL (Matryoshka Representation Learning) loss
"""

import logging
import hydra
import sys
import os
import datasets

from omegaconf import OmegaConf
from pathlib import Path

from retreever.models.retreever import ReTreever
from retreever.models.mrl import MRL
from retreever.training.trainer import get_trainer
from retreever.training.get_training_args import get_training_args
from retreever.utils.neural import (
    ContrastiveLoss,
    MultiLabelContrastiveLoss, 
    MRLLoss,
    CompositeLoss,
)
from retreever.data.collators import (
    SupervisedCollator,
    ImageSupervisedCollator,
    AudioSupervisedCollator,
    TextImageSupervisedCollator,
)
from retreever.data.imagenet_dataset import ImageNetRetrievalDataset
from retreever.data.voxceleb_dataset import VoxCeleb2RetrievalDataset
from retreever.data.text_image_dataset import TextImageRetrievalDataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def set_random_seed(seed=42):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_local_rank_and_world_size():
    """Get local rank and world size for distributed training."""
    import torch.distributed as dist
    
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def print_rank_0(print_fn, message):
    """Print only on rank 0."""
    local_rank, _ = get_local_rank_and_world_size()
    if local_rank == 0:
        print_fn(message)


@hydra.main(version_base=None, config_path="config/", config_name="train.yaml")
def train(cfg: OmegaConf):
    """Main training function."""
    
    if cfg.get("seed", None) is not None:
        set_random_seed(cfg.seed)
    else:
        set_random_seed()

    local_rank, world_size = get_local_rank_and_world_size()
    multi_node = int(world_size > 1)

    print_rank_0(logger.info, OmegaConf.to_yaml(cfg))

    # Hack from: https://github.com/huggingface/datasets/issues/1785
    datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory=".": True

    # ========== Load Dataset ==========
    if cfg.dataset == "imagenet1k":
        training_data = ImageNetRetrievalDataset(
            data_dir=cfg.data_dir,
            split='train',
            num_contexts=100
        )
        validation_data = ImageNetRetrievalDataset(
            data_dir=cfg.data_dir,
            split='val',
            subset=50,
            for_eval=True,
        )
    elif cfg.dataset == "voxceleb2":
        training_data = VoxCeleb2RetrievalDataset(
            data_dir=cfg.data_dir,
            split='train',
            num_contexts=2,
            for_eval=False,
            sample_rate=16000,
            audio_ext='m4a',
        )
        validation_data = VoxCeleb2RetrievalDataset(
            data_dir=cfg.data_dir,
            split='val',
            for_eval=True,
            sample_rate=16000,
            audio_ext='m4a',
            subset=20,
        )
    elif cfg.dataset in ["flickr30k", "coco"]:
        data_path = Path(cfg.data_dir) / "dataset"
        dataset_dict = datasets.load_from_disk(data_path)
        
        training_data = TextImageRetrievalDataset(
            data=dataset_dict['train'],
            images_base_dir=cfg.data_dir
        )
        validation_data = TextImageRetrievalDataset(
            data=dataset_dict['val'],
            images_base_dir=cfg.data_dir,
            subset_size=1000
        )
    else:
        # Text datasets: NQ, HotpotQA, RepliQA, TopiocQA
        data = datasets.load_from_disk(cfg.data_dir)
        training_data = data["train"]
        validation_data = data["val"]
        
        # HotpotQA has large validation set, use subset
        if cfg.dataset == "hotpotqa":
            validation_data = validation_data.select(range(2000))

    # ========== Instantiate Loss ==========
    if cfg.model.loss_type == "contrastive":
        print("Initiating contrastive loss")
        loss = ContrastiveLoss(
            local_loss=not multi_node,
            init_tmp=cfg.train.tmp_scale / (cfg.train.train_batch_size * world_size),
            freeze_tmp=cfg.train.freeze_tmp,
            sim_measure=cfg.model.sim_measure,
        )
        eval_loss = ContrastiveLoss(
            local_loss=not multi_node,
            init_tmp=1.0,
            freeze_tmp=True,
            sim_measure=cfg.model.sim_measure,
        )
        
    elif cfg.model.loss_type == "multi_label_contrastive":
        print("Initiating multi-label contrastive loss")
        loss = MultiLabelContrastiveLoss(
            local_loss=not multi_node,
            init_tmp=cfg.train.tmp_scale / (cfg.train.train_batch_size * world_size),
            freeze_tmp=cfg.train.freeze_tmp,
            sim_measure=cfg.model.sim_measure,
        )
        eval_loss = MultiLabelContrastiveLoss(
            local_loss=not multi_node,
            init_tmp=1.0,
            freeze_tmp=True,
            sim_measure=cfg.model.sim_measure,
        )
        
    elif cfg.model.loss_type == "mrl":
        print("Initiating MRL loss")
        loss = MRLLoss(
            encoder_dim=cfg.model.encoder_dim,
            local_loss=not multi_node,
            init_tmp=cfg.train.tmp_scale / (cfg.train.train_batch_size * world_size),
            freeze_tmp=cfg.train.freeze_tmp,
            sim_measure=cfg.model.sim_measure,
        )
        eval_loss = MRLLoss(
            encoder_dim=cfg.model.encoder_dim,
            local_loss=not multi_node,
            init_tmp=1.0,
            freeze_tmp=True,
            sim_measure=cfg.model.sim_measure,
        )
    else:
        raise NotImplementedError(f"Unknown loss_type: {cfg.model.loss_type}")

    # Wrap in composite loss (allows adding more terms later)
    loss = CompositeLoss([loss], [1.0])
    eval_loss = CompositeLoss([eval_loss], [1.0])

    # ========== Instantiate Model ==========
    if cfg.model_type == "retreever":
        model = ReTreever(
            loss=loss,
            encoder_type=cfg.model.encoder_type,
            freeze_encoder=cfg.model.freeze_encoder,
            tree_type=cfg.model.tree_type,
            cache_dir=None,  # Use HF defaults
            dual_model=cfg.model.dual_model,
            tree_depth=cfg.model.tree_depth,
            tree_split_fn=cfg.model.tree_split_fn,
            encoder_token_level=cfg.model.encoder_token_level,
            encoder_normalize=cfg.model.encoder_normalize,
            encoder_context_length=cfg.model.encoder_context_length,
            embedding_dim=cfg.model.split_fn_embedding_dim,
            num_embeddings_per_node=cfg.model.num_embeddings_per_node,
            scoring_fn_name=cfg.model.cross_attn_scoring_fn_name,
            d_k=cfg.model.split_fn_d_k,
            n_heads=cfg.model.split_fn_n_heads,
            eval_strategy=cfg.model.eval_strategy,
            train_full_tree_rep=cfg.model.train_full_tree_rep,
            node_embedding_strategy=cfg.model.get("node_embedding_strategy", "independent"),
        )
    elif cfg.model_type == "mrl":
        model = MRL(
            loss=loss,
            encoder_type=cfg.model.encoder_type,
            freeze_encoder=cfg.model.freeze_encoder,
            cache_dir=None,
            dual_model=cfg.model.dual_model,
            tree_split_fn=cfg.model.tree_split_fn,
            encoder_token_level=cfg.model.encoder_token_level,
            encoder_normalize=cfg.model.encoder_normalize,
            encoder_context_length=cfg.model.encoder_context_length,
            embedding_dim=cfg.model.split_fn_embedding_dim,
            num_embeddings_per_node=cfg.model.num_embeddings_per_node,
            scoring_fn_name=cfg.model.cross_attn_scoring_fn_name,
            d_k=cfg.model.split_fn_d_k,
            n_heads=cfg.model.split_fn_n_heads,
        )
    else:
        raise NotImplementedError(f"Unknown model_type: {cfg.model_type}")

    # ========== Instantiate Collator ==========
    if cfg.dataset == "imagenet1k":
        collator = ImageSupervisedCollator(
            query_processor=model.query_encoder.processor,
            ctx_processor=model.context_encoder.processor,
        )
    elif cfg.dataset == "voxceleb2":
        collator = AudioSupervisedCollator(
            model.query_encoder.processor,
            model.context_encoder.processor,
        )
    elif cfg.dataset in ["flickr30k", "coco"]:
        collator = TextImageSupervisedCollator(
            query_tokenizer=model.query_encoder.tokenizer,
            ctx_processor=model.context_encoder.processor,
        )
    else:
        # Text datasets
        collator = SupervisedCollator(
            model.query_encoder.tokenizer,
            model.context_encoder.tokenizer,
            model.query_encoder.prefix,
            model.context_encoder.prefix,
            context_field="contexts_list" if cfg.dataset == "hotpotqa" else "context",
        )

    # ========== Create Save Directory ==========
    if not os.path.exists(cfg.savedir) and local_rank == 0:
        os.makedirs(cfg.savedir)

    # ========== Get Training Arguments ==========
    training_args = get_training_args(cfg, cfg.savedir, local_rank, world_size)

    # ========== Print Model Stats ==========
    if cfg.get("debug", False):
        print_rank_0(logger.info, str(training_args))
        print_rank_0(logger.info, str(model))
        print_rank_0(logger.info, f"Training data size: {len(training_data)}")
        print_rank_0(logger.info, f"Validation data size: {len(validation_data)}")

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print_rank_0(
            logger.info,
            f"Total parameters: {total_params}\nTrainable parameters: {trainable_params}"
        )

    # ========== Get Trainer ==========
    trainer = get_trainer(
        local_rank=local_rank,
        model=model,
        data_collator=collator,
        cfg=cfg,
        training_args=training_args,
        training_data=training_data,
        validation_data=validation_data,
        additional_ctxs_per_device=len(validation_data) * cfg.logging.factor_val_irrelevant_ctxs // world_size,
        additional_ctxs=len(validation_data) * cfg.logging.factor_val_irrelevant_ctxs,
        contrastive_loss_tmp1=eval_loss,
    )

    # ========== Resume from Checkpoint ==========
    if cfg.get("ckpt", None) is not None:
        resume_ckpt = cfg.ckpt
    else:
        resume_ckpt = any(d.startswith("checkpoint") for d in os.listdir(cfg.savedir))

    # ========== Save Config ==========
    OmegaConf.save(config=cfg, f=os.path.join(cfg.savedir, "config.yaml"))

    # ========== Train ==========
    trainer.train(resume_from_checkpoint=resume_ckpt)

    logger.info("Training complete!")


if __name__ == "__main__":
    train()

import logging
import hydra
import sys
import os

import torch
import datasets

from omegaconf import OmegaConf

from retreever.training.get_training_args import get_training_args
from retreever.utils.scripting import set_random_seed, print_rank_0, get_local_rank_and_world_size

from retreever.models.retreever import ReTreever
from retreever.training.trainer import get_trainer
from retreever.utils.neural import (
    ContrastiveLoss,
    L1Regularization,
    CompositeLoss,
    EntropyMinimizationLoss,
    MaxProbabilityLoss,
    TreeConsistencyRegularization,
    PathConsistencyLoss,
    TreeLocalityLoss,
    MultiLabelContrastiveLoss,
    TripletLoss,
    HardAssignmentLoss,
    PathConsistencySkipConnectionsLoss,
)
from retreever.utils.toolkit_paths import PATH_HF_CACHE_RW
from local_paths import DATA_PATHS as KNOWN_DATASETS
from retreever.data.collators import (
    SupervisedCollator,
    ImageSupervisedCollator,
    AudioSupervisedCollator,
)

from retreever.data.imagenet_dataset import ImageNetRetrievalDataset
from retreever.data.voxceleb_dataset import VoxCeleb2RetrievalDataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

set_random_seed()

CONFIG_NAME = "train.yaml"

# Supported dataset names and their on-disk locations are configured in
# ``local_paths.DATA_PATHS`` (re-exported above as ``KNOWN_DATASETS``).


"""
Supported combinations:

    1. ReTreever + contrastive loss             : model_type = "retreever", loss_type = "contrastive"
    2. ReTreever + multi-label contrastive loss : model_type = "retreever", loss_type = "multi_label_contrastive"
    3. ReTreever + triplet loss                 : model_type = "retreever", loss_type = "triplet"
"""
@hydra.main(version_base=None, config_path="config/", config_name=CONFIG_NAME)
def train(cfg: OmegaConf):
    """cfg: namespace of experiment arguments"""

    if cfg.get("seed", None) is not None:
        set_random_seed(cfg.seed)

    # if deepspeed passed as command argument, experiment is run with multiple gpus
    multi_node = cfg.get("deepspeed", False)

    local_rank, world_size = get_local_rank_and_world_size()
    multi_node = int(world_size > 1)

    print_rank_0(logger.info, cfg)

    # load dataset
    # hack from: https://github.com/huggingface/datasets/issues/1785
    datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory=".": True

    if cfg.dataset == "imagenet1k":
        training_data = ImageNetRetrievalDataset(
            data_dir=KNOWN_DATASETS[cfg.dataset],
            split="train",
            num_contexts=100,
        )
        # In eval mode there is only one context per query to avoid indexing the
        # same image multiple times.
        validation_data = ImageNetRetrievalDataset(
            data_dir=KNOWN_DATASETS[cfg.dataset],
            split="val",
            subset=50,
            for_eval=True,
        )
    elif cfg.dataset == "voxceleb2":
        training_data = VoxCeleb2RetrievalDataset(
            data_dir=KNOWN_DATASETS[cfg.dataset],
            split="train",
            num_contexts=2,
            for_eval=False,
            sample_rate=16000,
            audio_ext="m4a",
        )
        validation_data = VoxCeleb2RetrievalDataset(
            data_dir=KNOWN_DATASETS[cfg.dataset],
            split="val",
            for_eval=True,
            sample_rate=16000,
            audio_ext="m4a",
            subset=20,
        )
    else:
        if "+" in cfg.dataset:
            ds_names = cfg.dataset.split("+")
            train_parts = []
            for dn in ds_names:
                train_parts.append(datasets.load_from_disk(KNOWN_DATASETS[dn])["train"])
            training_data = datasets.concatenate_datasets(train_parts)
            validation_data = datasets.load_from_disk(KNOWN_DATASETS[ds_names[0]])["val"]
        else:
            data = datasets.load_from_disk(KNOWN_DATASETS[cfg.dataset])
            training_data = data["train"]
            validation_data = data["val"]

    if cfg.dataset == "hotpotqa" or "+" in cfg.dataset:
        # hotpotqa has a massive validation set; use a smaller subset for faster
        # eval during training.
        validation_data = validation_data.select(range(2000))

    # Instantiate the main loss
    if cfg.model.loss_type == "contrastive":
        print("Initiating contrastive loss")
        loss = ContrastiveLoss(
            local_loss=not multi_node,
            init_tmp=cfg.train.tmp_scale / (cfg.train.train_batch_size * world_size),
            freeze_tmp=cfg.train.freeze_tmp,
            sim_measure=cfg.model.sim_measure,
            optimize_prior_levels_too=cfg.train.optimize_prior_levels_too,
            use_siglip_loss=cfg.train.get("use_siglip_loss", False),
            max_tmp=cfg.train.get("max_tmp", 30.0),
            use_separate_temps=cfg.train.get("use_separate_temps", False),
            max_tmp_qc=cfg.train.get("max_tmp_qc", None),
            max_tmp_cq=cfg.train.get("max_tmp_cq", None),
            use_depth_wise_temps=cfg.train.get("use_depth_wise_temps", False),
        )

        contrastive_loss_tmp1 = ContrastiveLoss(
            local_loss=not multi_node,
            init_tmp=1.0,
            freeze_tmp=True,
            sim_measure=cfg.model.sim_measure,
            optimize_prior_levels_too=cfg.train.optimize_prior_levels_too,
        )
    elif cfg.model.loss_type == "multi_label_contrastive":
        print("Initiating multi label contrastive loss")
        loss = MultiLabelContrastiveLoss(
            local_loss=not multi_node,
            init_tmp=cfg.train.tmp_scale / (cfg.train.train_batch_size * world_size),
            freeze_tmp=cfg.train.freeze_tmp,
            sim_measure=cfg.model.sim_measure,
            optimize_prior_levels_too=cfg.train.optimize_prior_levels_too,
            max_tmp=cfg.train.get("max_tmp", 30.0),
            optimize_whole_emb=cfg.train.get("optimize_whole_emb", False),
        )

        contrastive_loss_tmp1 = MultiLabelContrastiveLoss(
            local_loss=not multi_node,
            init_tmp=1.0,
            freeze_tmp=True,
            sim_measure=cfg.model.sim_measure,
            optimize_prior_levels_too=cfg.train.optimize_prior_levels_too,
            optimize_whole_emb=cfg.train.get("optimize_whole_emb", False),
        )
    elif cfg.model.loss_type == "triplet":
        print("Initiating triplet loss")
        loss = TripletLoss(
            local_loss=not multi_node,
            freeze_tmp=cfg.train.freeze_tmp,
            margin=cfg.model.margin,
            sim_measure=cfg.model.sim_measure,
            max_tmp=cfg.train.get("max_tmp", 30.0),
            use_dynamic_margin=cfg.model.use_dynamic_margin,
        )

        contrastive_loss_tmp1 = TripletLoss(
            local_loss=not multi_node,
            freeze_tmp=True,
            margin=cfg.model.margin,
            sim_measure=cfg.model.sim_measure,
            max_tmp=cfg.train.get("max_tmp", 30.0),
            use_dynamic_margin=cfg.model.use_dynamic_margin,
        )
    else:
        raise NotImplementedError(f"Unknown model.loss_type {cfg.model.loss_type}")

    # Collect all loss terms and their lambdas
    loss_terms = [loss]
    loss_lambdas = [1.0]
    contrastive_terms = [contrastive_loss_tmp1]
    contrastive_lambdas = [1.0]

    # L1 regularization
    if cfg.train.l1_lambda > 0:
        print(f"Adding L1Regularization loss with lambda {cfg.train.l1_lambda}")
        l1_reg_term = L1Regularization()
        loss_terms.append(l1_reg_term)
        loss_lambdas.append(cfg.train.l1_lambda)
        contrastive_terms.append(l1_reg_term)
        contrastive_lambdas.append(cfg.train.l1_lambda)

    if hasattr(cfg.train, "entropy_lambda") and cfg.train.entropy_lambda > 0:
        print(f"Adding EntropyMinimizationLoss loss with lambda {cfg.train.entropy_lambda}")
        entropy_loss_term = EntropyMinimizationLoss()
        loss_terms.append(entropy_loss_term)
        loss_lambdas.append(cfg.train.entropy_lambda)
        contrastive_terms.append(entropy_loss_term)
        contrastive_lambdas.append(cfg.train.entropy_lambda)

    if hasattr(cfg.train, "max_prob_lambda") and cfg.train.max_prob_lambda > 0:
        print(f"Adding MaxProbabilityLoss loss with lambda {cfg.train.max_prob_lambda}")
        max_prob_loss_term = MaxProbabilityLoss()
        loss_terms.append(max_prob_loss_term)
        loss_lambdas.append(cfg.train.max_prob_lambda)
        contrastive_terms.append(max_prob_loss_term)
        contrastive_lambdas.append(cfg.train.max_prob_lambda)

    if hasattr(cfg.train, "const_temp_contrastive_lambda") and len(cfg.train.const_temp_contrastive_lambda) > 0:
        print(f"Adding Const Temp Contrastive losses with lambda {cfg.train.const_temp_contrastive_lambda}")
        print(f"Using temperatures: {cfg.train.const_temps}")
        for temp, lam in zip(cfg.train.const_temps, cfg.train.const_temp_contrastive_lambda):
            loss_term = ContrastiveLoss(
                local_loss=not multi_node,
                init_tmp=cfg.train.tmp_scale / (cfg.train.train_batch_size * world_size),
                freeze_tmp=True,
                sim_measure=cfg.model.sim_measure,
                optimize_prior_levels_too=cfg.train.optimize_prior_levels_too,
                max_tmp=temp,
            )
            loss_terms.append(loss_term)
            loss_lambdas.append(lam)
            contrastive_terms.append(loss_term)
            contrastive_lambdas.append(lam)
        loss_terms = loss_terms[1:]
        loss_lambdas = loss_lambdas[1:]
        contrastive_terms = contrastive_terms[1:]
        contrastive_lambdas = contrastive_lambdas[1:]

    if hasattr(cfg.train, "tree_consistency_reg_lambda") and cfg.train.tree_consistency_reg_lambda > 0:
        print(f"Adding Temp Consistency Reg with lambda {cfg.train.tree_consistency_reg_lambda}")
        loss_term = TreeConsistencyRegularization(max_depth=cfg.model.tree_depth)
        loss_terms.append(loss_term)
        loss_lambdas.append(cfg.train.tree_consistency_reg_lambda)
        contrastive_terms.append(loss_term)
        contrastive_lambdas.append(cfg.train.tree_consistency_reg_lambda)

    if hasattr(cfg.train, "path_consistency_lambda") and cfg.train.path_consistency_lambda > 0:
        print(f"Adding Path Consistency Reg with lambda {cfg.train.path_consistency_lambda}")
        loss_term = PathConsistencyLoss(
            max_depth=cfg.model.tree_depth,
            temperature=cfg.model.get("path_loss_tmp", 0.1),
            compute_at_all_levels=cfg.model.get("path_loss_all_levels", False),
        )
        loss_terms.append(loss_term)
        loss_lambdas.append(cfg.train.path_consistency_lambda)
        contrastive_terms.append(loss_term)
        contrastive_lambdas.append(cfg.train.path_consistency_lambda)

    if hasattr(cfg.train, "path_consistency_skip_connection_lambda") and cfg.train.path_consistency_skip_connection_lambda > 0:
        print(f"Adding Path Consistency Reg with lambda {cfg.train.path_consistency_skip_connection_lambda}")
        loss_term = PathConsistencySkipConnectionsLoss(
            max_depth=cfg.model.tree_depth,
            temperature=cfg.model.get("path_loss_tmp", 0.1),
        )
        loss_terms.append(loss_term)
        loss_lambdas.append(cfg.train.path_consistency_skip_connection_lambda)
        contrastive_terms.append(loss_term)
        contrastive_lambdas.append(cfg.train.path_consistency_skip_connection_lambda)

    if hasattr(cfg.train, "tree_locality_reg_lambda") and cfg.train.tree_locality_reg_lambda > 0:
        print(f"Adding Tree Locality Reg with lambda {cfg.train.tree_locality_reg_lambda}")
        loss_term = TreeLocalityLoss(max_depth=cfg.model.tree_depth)
        loss_terms.append(loss_term)
        loss_lambdas.append(cfg.train.tree_locality_reg_lambda)
        contrastive_terms.append(loss_term)
        contrastive_lambdas.append(cfg.train.tree_locality_reg_lambda)

    if hasattr(cfg.train, "hard_assignment_reg_lambda") and cfg.train.hard_assignment_reg_lambda > 0:
        print(f"Adding Hard Assignment Reg with lambda {cfg.train.hard_assignment_reg_lambda}")
        loss_term = HardAssignmentLoss()
        loss_terms.append(loss_term)
        loss_lambdas.append(cfg.train.hard_assignment_reg_lambda)
        contrastive_terms.append(loss_term)
        contrastive_lambdas.append(cfg.train.hard_assignment_reg_lambda)

    token_loss_term = None
    if hasattr(cfg.train, "token_train_lambda") and cfg.train.token_train_lambda > 0:
        print(f"Adding Token train loss with lambda {cfg.train.token_train_lambda}")
        if cfg.model.loss_type == "contrastive":
            token_loss_term = ContrastiveLoss(
                local_loss=not multi_node,
                init_tmp=cfg.train.tmp_scale / (cfg.train.train_batch_size * world_size),
                freeze_tmp=cfg.train.freeze_tmp,
                sim_measure="cos_sim",
                optimize_whole_emb=True,
                max_tmp=cfg.train.get("max_tmp", 30.0),
            )
        elif cfg.model.loss_type == "multi_label_contrastive":
            token_loss_term = MultiLabelContrastiveLoss(
                local_loss=not multi_node,
                init_tmp=cfg.train.tmp_scale / (cfg.train.train_batch_size * world_size),
                freeze_tmp=cfg.train.freeze_tmp,
                sim_measure="cos_sim",
                optimize_whole_emb=True,
                max_tmp=cfg.train.get("max_tmp", 30.0),
            )

    loss = CompositeLoss(loss_terms, loss_lambdas)
    contrastive_loss_tmp1 = CompositeLoss(contrastive_terms, contrastive_lambdas)

    # instantiate model
    if cfg.model_type == "retreever":
        model = ReTreever(
            loss=loss,
            token_loss_term=token_loss_term,
            encoder_type=cfg.model.encoder_type,
            freeze_encoder=cfg.model.freeze_encoder,
            tree_type=cfg.model.tree_type,
            cache_dir=PATH_HF_CACHE_RW,
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
            use_sigmoid_in_projection=cfg.model.use_sigmoid_in_projection,
            attend_ne_only_on_text=cfg.model.attend_ne_only_on_text,
            enable_llm_split_score_prop=cfg.model.enable_llm_split_score_prop,
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
            num_cross_attn_queries=cfg.model.get("num_cross_attn_queries", 10),
            # Tree adapter for cross-dataset transfer
            tree_adapter_targets=cfg.model.get("tree_adapter_targets", []),
            tree_adapter_lora_r=cfg.model.get("tree_adapter_lora_r", 8),
            tree_adapter_lora_alpha=cfg.model.get("tree_adapter_lora_alpha", 16),
            tree_adapter_logit_type=cfg.model.get("tree_adapter_logit_type", "scalar"),
        )

    else:
        raise NotImplementedError(f"Unknown model_type {cfg.model_type}")

    # --- Load pretrained weights for finetuning (weights only, fresh training state) ---
    # Usage: set pretrained_weights=/path/to/pytorch_model.bin in config
    if cfg.get("pretrained_weights") is not None:
        logger.info(f"Loading pretrained weights from {cfg.pretrained_weights}")
        state_dict = torch.load(cfg.pretrained_weights, map_location="cpu", weights_only=True)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("loss.")}

        # Remap keys if tree adapter LoRA has wrapped projection layers.
        # e.g. context_tree.split.query_proj.weight -> context_tree.split.query_proj.base.weight
        tree_adapter_targets = cfg.model.get("tree_adapter_targets", [])
        if isinstance(tree_adapter_targets, str):
            tree_adapter_targets = [t.strip() for t in tree_adapter_targets.split(",") if t.strip()]
        if "projections" in tree_adapter_targets:
            remapped = {}
            for k, v in state_dict.items():
                new_k = k
                for proj_name in ["query_proj", "value_proj"]:
                    prefix = f"split.{proj_name}."
                    if prefix in k and ".base." not in k:
                        new_k = k.replace(prefix, f"split.{proj_name}.base.")
                remapped[new_k] = v
            state_dict = remapped

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.info(f"  Missing keys (expected for adapter/loss params): {missing}")
        if unexpected:
            logger.warning(f"  Unexpected keys: {unexpected}")
        logger.info("  Pretrained weights loaded.")

    # instantiate collator
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
    else:
        # topiocqa packs the full conversation history into the
        # question (most recent turn last), so we want to truncate from the
        # left to keep the most recent context.
        query_truncate_from_left = (
            "topiocqa" in cfg.get("dataset", "")
            or "+" in cfg.get("dataset", "")
        )
        collator = SupervisedCollator(
            model.query_encoder.tokenizer,
            model.context_encoder.tokenizer,
            model.query_encoder.prefix,
            model.context_encoder.prefix,
            context_field="contexts_list" if cfg.dataset == "hotpotqa" else "context",
            query_truncate_from_left=query_truncate_from_left,
        )

    # instantiate HF dataobject for trainer's configuration
    if not os.path.exists(cfg.savedir) and local_rank == 0:
        os.makedirs(cfg.savedir)
    training_args = get_training_args(cfg, cfg.savedir, local_rank, world_size)

    if cfg.debug:
        print_rank_0(logger.info, training_args)
        print_rank_0(logger.info, model)
        print_rank_0(logger.info, f"Training data size: {len(training_data)}")
        print_rank_0(logger.info, f"Validation data size: {len(validation_data)}")

        total_params = 0
        trainable_params = 0
        for _, v in model.named_parameters():
            total_params += v.data.numel()
            if v.requires_grad:
                trainable_params += v.data.numel()
        print_rank_0(
            logger.info,
            f"Total parameters: {total_params}\nTrainable parameters: {trainable_params}",
        )

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
        contrastive_loss_tmp1=contrastive_loss_tmp1,
    )

    # resume training from specified checkpoint path
    if cfg.ckpt is not None:
        resume_ckpt = cfg.ckpt
    else:
        # or from the last checkpoint saved in savedir; if none, train from scratch
        resume_ckpt = any(d.startswith("checkpoint") for d in os.listdir(cfg.savedir))

    OmegaConf.save(config=cfg, f=os.path.join(cfg.savedir, "config.yaml"))

    trainer.train(resume_from_checkpoint=resume_ckpt)

    logger.info("Experiment done\n")


if __name__ == "__main__":
    train()

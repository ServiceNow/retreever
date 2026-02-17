"""Training arguments configuration for ReTreever."""

import os
from omegaconf import DictConfig
from transformers import TrainingArguments


def get_training_args(
    cfg: DictConfig,
    savedir: str,
    local_rank: int = 0,
    world_size: int = 1
) -> TrainingArguments:
    """
    Prepare training arguments from config.

    Args:
        cfg: Experiment configuration
        savedir: Directory to save experiment data
        local_rank: Rank of executing node (0 for main)
        world_size: Number of nodes

    Returns:
        TrainingArguments for HuggingFace Trainer
    """
    deepspeed_cfg = cfg.get("deepspeed", None)

    training_args = TrainingArguments(
        # Needs to be False since custom dataset passes fields processed in collator
        remove_unused_columns=False,
        
        # Optimization
        per_device_train_batch_size=cfg.train.train_batch_size,
        per_device_eval_batch_size=cfg.train.test_batch_size,
        max_steps=cfg.train.steps,
        learning_rate=cfg.train.learning_rate,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        warmup_ratio=cfg.train.warmup_ratio,
        adam_beta1=cfg.train.adam_beta1,
        adam_beta2=cfg.train.adam_beta2,
        adam_epsilon=cfg.train.adam_epsilon,
        weight_decay=cfg.train.weight_decay,
        max_grad_norm=cfg.train.max_grad_norm,
        gradient_accumulation_steps=cfg.train.skip_steps,
        dataloader_num_workers=max(os.cpu_count() // world_size, 1),
        fp16=cfg.train.fp16,
        bf16=cfg.train.bf16,
        
        # Logging and checkpointing
        report_to="wandb",
        logging_dir=os.path.join(savedir, "logs"),
        logging_strategy="steps",
        logging_steps=cfg.logging.log_every,
        logging_first_step=True,
        output_dir=savedir,
        log_on_each_node=False,  # Only rank 0 writes logs
        save_strategy="steps",
        save_steps=cfg.logging.log_every,
        load_best_model_at_end=True,
        
        # Evaluation
        eval_steps=cfg.logging.log_every,
        evaluation_strategy="steps",
        
        # Distributed training
        ddp_find_unused_parameters=True,
        deepspeed=deepspeed_cfg,
        local_rank=local_rank,
        save_safetensors=False,
    )

    return training_args

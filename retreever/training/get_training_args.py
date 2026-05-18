import os
from omegaconf import DictConfig
from transformers import TrainingArguments


def get_training_args(
    cfg: DictConfig, savedir: str, local_rank: int = 0, world_size: int = 1
) -> TrainingArguments:
    """Prepare training args given exp dict and command line args.

    Args:
        cfg: Experiment namespace
        savedir: Folder where to save experiment data.
        local_rank: rank of executing node. 0 for principal.
        world_size: number of nodes

    Returns:
        TrainingArguments: Training arguments for HF trainer.
    """
    deepspeed_cfg = cfg.get("deepspeed", None)

    training_args = TrainingArguments(
        remove_unused_columns=False,  # This needs to be False since our custom dataset passes fields that are processed in the collator.
        # optimization
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
        # logging and checkpointing
        report_to="wandb",
        logging_dir=os.path.join(savedir, "logs"),
        logging_strategy="steps",
        logging_steps=cfg.logging.log_every,
        logging_first_step=True,
        output_dir=savedir,
        log_on_each_node=False,  # Only rank 0 will write logs.
        # push_to_hub=cfg.push_to_hub,
        # save_total_limit=5,  # save at most 5 checkpoints (last 4 + best)
        save_strategy="steps",
        save_steps=cfg.logging.log_every,
        load_best_model_at_end=False,
        # evaluation
        eval_steps=cfg.logging.log_every,
        evaluation_strategy="steps",
        # distributed
        ddp_find_unused_parameters=True,
        deepspeed=deepspeed_cfg,
        local_rank=local_rank,
        save_safetensors=False,
    )

    return training_args

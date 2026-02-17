"""Custom trainer for ReTreever models."""

import math
import wandb
import torch
from typing import Optional, Dict

from omegaconf import DictConfig
from datasets import Dataset
from transformers import DataCollator, Trainer, TrainerCallback, TrainingArguments

from retreever.models.retreever import ReTreever
from retreever.evaluation.retreeval_metrics import retreeval_metrics
from retreever.evaluation.evaluator import RetEvaluator
from retreever.training.depth_schedulers import KNOWN_SCHEDULERS, RandomDepthScheduler


class EvaluateCallback(TrainerCallback):
    """Callback to evaluate before starting training and log retrieval metrics."""

    def __init__(
        self,
        wandb_run,
        additional_ctxs_per_device=0,
        additional_ctxs=0,
        eval_dataset=None,
        train_dataset=None,
        eval_batch_size=64,
        **tracked_metrics
    ):
        super().__init__()

        self._wandb = wandb_run
        self.num_additional_ctxs = additional_ctxs_per_device
        self.additional_ctxs = additional_ctxs
        self.evaluator = RetEvaluator(
            additional_ctxs_per_device=additional_ctxs_per_device,
            additional_ctxs=additional_ctxs,
            **tracked_metrics
        )
        self.unsharded_train_loader = None
        self.unsharded_eval_dataset = eval_dataset
        self.unsharded_train_dataset = train_dataset
        self.eval_batch_size = eval_batch_size

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step == 0:
            control.should_evaluate = True
            
    def on_train_begin(self, args, state, control, model, train_dataloader, **kwargs):
        """Initialize unsharded data loaders for evaluation."""
        self.unsharded_eval_loader = torch.utils.data.DataLoader(
            self.unsharded_eval_dataset,
            batch_size=self.eval_batch_size,
            sampler=torch.utils.data.SequentialSampler(self.unsharded_eval_dataset),
            collate_fn=train_dataloader.collate_fn,
            num_workers=train_dataloader.num_workers,
            pin_memory=train_dataloader.pin_memory
        )
        
        self.unsharded_train_loader = torch.utils.data.DataLoader(
            self.unsharded_train_dataset,
            batch_size=self.eval_batch_size,
            sampler=torch.utils.data.SequentialSampler(self.unsharded_train_dataset),
            collate_fn=train_dataloader.collate_fn,
            num_workers=train_dataloader.num_workers,
            pin_memory=train_dataloader.pin_memory
        )
    
    def on_evaluate(self, args, state, control, **kwargs):
        """Only evaluate on main process."""
        if not state.is_world_process_zero:
            return
        
        model = kwargs.pop("model")
        eval_dataloader = kwargs.pop("eval_dataloader")

        # Use unsharded train loader as additional contexts if requested
        if self.num_additional_ctxs > 0:
            additional_dataloader = self.unsharded_train_loader
        else:
            additional_dataloader = None

        with torch.cuda.amp.autocast(cache_enabled=False):
            additional_metrics = self.evaluator(
                model, 
                self.unsharded_eval_loader,  # Use unsharded eval loader
                additional_dataloader
            )

        if state.is_world_process_zero:
            print(additional_metrics)
            self._wandb.log(additional_metrics, step=state.global_step + 1)


class RetrievalTrainer(Trainer):
    """Custom trainer class for training on retrieval task."""

    def __init__(
        self,
        model: ReTreever,
        data_collator: DataCollator,
        args: TrainingArguments,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        depth_scheduler=None,
        use_custom_scheduler=False,
        callbacks: Optional[TrainerCallback] = None,
    ):
        """
        Initialize retrieval trainer.
        
        Args:
            model: ReTreever model to train
            data_collator: Data collator for batching
            args: Training arguments
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            depth_scheduler: Optional depth curriculum scheduler
            use_custom_scheduler: Whether to use custom LR scheduler
            callbacks: Optional callback functions
        """
        super(RetrievalTrainer, self).__init__(
            model=model,
            data_collator=data_collator,
            compute_metrics=retreeval_metrics,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=callbacks,
        )

        self.can_return_loss = True  # Key override to log loss
        self.depth_scheduler = depth_scheduler
        self.depth = model.tree_depth
        self.use_custom_scheduler = use_custom_scheduler

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None):
        """
        Log metrics including temperature and tree-specific metrics.
        
        Args:
            logs: Dictionary of metrics to log
            start_time: Optional start time for timing metrics
        """
        # Track temperature of contrastive loss
        loss_tmp = [
            param.item()
            for name, param in self.model.loss.named_parameters()
            if "temp_coef" in name
        ]

        if len(loss_tmp) > 0:
            logs |= {
                "loss_temperature": math.exp(loss_tmp[0]),
            }

        # Track tree biases and current depth
        logs |= {
            "question_tree_bias_l2": torch.norm(self.model.query_tree.get_bias(), p=2).item(),
            "context_tree_bias_l2": torch.norm(self.model.context_tree.get_bias(), p=2).item(),
            "loss_depth": self.depth,
        }

        # Default logging
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Perform a training step with depth scheduling.
        
        Args:
            model: Model to train
            inputs: Input batch
            num_items_in_batch: Optional number of items in batch
            
        Returns:
            Loss tensor
        """
        if self.state.global_step == 0:
            self.model._init_tree_params(**inputs)

        if self.depth_scheduler is not None:
            depth = self.depth_scheduler.get_depth(self.state.global_step)

            if not isinstance(self.depth_scheduler, RandomDepthScheduler) and self.depth < depth:
                # When changing loss depth, reinit tree params from parents on
                self.model._init_tree_params(**inputs, depth=depth - 1)

            self.depth = depth
            
        # Override depth if train_full_tree_rep set
        if self.model.train_full_tree_rep:
            self.depth = -1

        # Append loss depth to model inputs (needed to compute loss)
        inputs |= {"depth": self.depth}
        return super().training_step(model, inputs)
    
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Create learning rate scheduler (custom or default).
        
        Args:
            num_training_steps: Total number of training steps
            optimizer: Optimizer to schedule
            
        Returns:
            Learning rate scheduler
        """
        if not self.use_custom_scheduler:
            # Use default HuggingFace scheduler (backward compatible)
            return super().create_scheduler(num_training_steps, optimizer)
        
        # Use custom scheduler
        if optimizer is None:
            optimizer = self.optimizer
        
        def get_custom_lr_lambda(
            warmup_steps=5000,
            decay_steps=50000,
            min_lr=5e-7,
            max_lr=1e-4
        ):
            """
            Custom LR scheduler:
            - Steps 0-5000: Linear warmup to max_lr
            - Steps 5000-55000: Linear decay to min_lr
            - Steps 55000+: Constant at min_lr
            """
            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    # Linear warmup
                    return float(current_step) / float(max(1, warmup_steps))
                elif current_step < (warmup_steps + decay_steps):
                    # Linear decay
                    progress = float(current_step - warmup_steps) / float(max(1, decay_steps))
                    return max(min_lr / max_lr, 1.0 - progress * (1.0 - min_lr / max_lr))
                else:
                    # Constant at min_lr
                    return min_lr / max_lr
            
            return lr_lambda
                
        lr_lambda = get_custom_lr_lambda()
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return self.lr_scheduler


def get_trainer(
    local_rank: int,
    model: ReTreever,
    data_collator: DataCollator,
    cfg: DictConfig,
    training_args: TrainingArguments,
    training_data: Dataset,
    validation_data: Dataset,
    additional_ctxs_per_device: int = 0,
    additional_ctxs: int = 0,
    **tracked_metrics,
) -> Trainer:
    """
    Instantiate Trainer object.

    Args:
        local_rank: Local process rank for distributed training
        model: Model to be trained
        data_collator: Data collator
        cfg: Configuration object
        training_args: Trainer's arguments
        training_data: Training dataset
        validation_data: Validation dataset
        additional_ctxs_per_device: Additional contexts per device for evaluation
        additional_ctxs: Total additional contexts for evaluation
        tracked_metrics: Additional metrics to track at evaluation

    Returns:
        Configured trainer
    """
    if local_rank == 0:
        wandb_run = wandb.init(
            name=cfg.logging.wandb_run,
            project=cfg.logging.wandb_project,
            mode="disabled" if cfg.debug else None,  # No logging while debugging
            config=dict(cfg),
        )
    else:
        wandb_run = None

    depth_scheduler = None

    if cfg.train.hierarchical:
        max_steps = (
            cfg.train.steps
            if cfg.train.depth_scheduler_type == "random"
            else math.ceil(cfg.train.depth_warmup_ratio * cfg.train.steps)
        )

        depth_scheduler = KNOWN_SCHEDULERS[cfg.train.depth_scheduler_type](
            min_value=0,
            max_value=cfg.model.tree_depth,
            max_steps=max_steps,
        )

    trainer = RetrievalTrainer(
        model,
        data_collator,
        args=training_args,
        train_dataset=training_data,
        eval_dataset=validation_data,
        depth_scheduler=depth_scheduler,
        use_custom_scheduler=cfg.train.get("use_custom_scheduler", False),
        callbacks=[
            EvaluateCallback(
                wandb_run,
                additional_ctxs_per_device=additional_ctxs_per_device,
                additional_ctxs=additional_ctxs,
                eval_dataset=validation_data,
                train_dataset=training_data,
                eval_batch_size=cfg.train.test_batch_size,
                **tracked_metrics
            )
        ],
    )

    return trainer

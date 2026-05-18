import math
import wandb
import torch

from omegaconf import DictConfig

from datasets import Dataset
from transformers import DataCollator, Trainer, TrainerCallback, TrainingArguments

from typing import Optional, Dict

from retreever.models.retreever import ReTreever
from retreever.evaluation.retreeval_metrics import retreeval_metrics
from retreever.evaluation.evaluator import RetEvaluator
from retreever.training.depth_schedulers import KNOWN_SCHEDULERS, RandomDepthScheduler
from retreever.utils.jobs import omegaconf_to_dict


class EvaluateCallback(TrainerCallback):
    """Callback to evaluate before starting training and log retrieval metrics"""

    def __init__(self,
                 wandb_run,
                 additional_ctxs_per_device=0,
                 additional_ctxs=0,
                 eval_dataset=None,
                 train_dataset=None,
                 eval_batch_size=64,
                 **tracked_metrics):
        super().__init__()

        self._wandb = wandb_run
        self.num_additional_ctxs = additional_ctxs_per_device
        self.additional_ctxs = additional_ctxs
        self.evaluator = RetEvaluator(
            additional_ctxs_per_device=additional_ctxs_per_device, additional_ctxs=additional_ctxs, **tracked_metrics
        )
        self.unsharded_train_loader = None
        self.unsharded_eval_dataset = eval_dataset
        self.unsharded_train_dataset = train_dataset
        self.eval_batch_size = eval_batch_size

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step == 0:
            control.should_evaluate = True
            
    def on_train_begin(self, args, state, control, model, train_dataloader, **kwargs):
            # """Initialize an unsharded train loader for the main process only."""
            # if state.is_world_process_zero and self.num_additional_ctxs > 0:
            # Get the full training dataset (not sharded)
            # train_dataset = train_dataloader.dataset
            
            # # If using DistributedSampler, it's wrapping the original dataset
            # if hasattr(train_dataset, 'dataset'):
            #     train_dataset = train_dataset.dataset
                
            self.unsharded_eval_loader = torch.utils.data.DataLoader(
                self.unsharded_eval_dataset,
                batch_size=self.eval_batch_size,
                sampler=torch.utils.data.SequentialSampler(self.unsharded_eval_dataset),
                collate_fn=train_dataloader.collate_fn,
                num_workers=train_dataloader.num_workers,
                pin_memory=train_dataloader.pin_memory
            )
            
            # Create a new dataloader with a sequential sampler (no sharding, no shuffling)
            self.unsharded_train_loader = torch.utils.data.DataLoader(
                self.unsharded_train_dataset,
                batch_size=self.eval_batch_size,
                sampler=torch.utils.data.SequentialSampler(self.unsharded_train_dataset),  # Sequential = no shuffling
                collate_fn=train_dataloader.collate_fn,
                num_workers=train_dataloader.num_workers,
                pin_memory=train_dataloader.pin_memory
            )

    # def on_evaluate(self, args, state, control, **kwargs):
    #     # only main process log generation metrics
    #     model = kwargs.pop("model")
    #     eval_dataloader = kwargs.pop("eval_dataloader")

    #     if self.num_additional_ctxs > 0:
    #         if state.is_world_process_zero and self.unsharded_train_loader:
    #             # Use the unsharded loader on main process
    #             additional_dataloader = self.unsharded_train_loader
    #         else:
    #             # For non-main processes, we won't use this anyway
    #             additional_dataloader = kwargs.pop("train_dataloader")
    #     else:
    #         additional_dataloader = None

    #     with torch.cuda.amp.autocast(cache_enabled=False):
    #         additional_metrics = self.evaluator(model, eval_dataloader, additional_dataloader)

    #     if state.is_world_process_zero:
    #         print(additional_metrics)
    #         self._wandb.log(additional_metrics, step=state.global_step + 1)
    
    def on_evaluate(self, args, state, control, **kwargs):
        # only evaluate on main process
        # if not state.is_world_process_zero:
        #     # Other ranks just wait at a barrier
        #     if torch.distributed.is_initialized():
        #         torch.distributed.barrier()
        #     return
        
        # using our unsharded loader
        model = kwargs.pop("model")
        
        # Only main process does evaluation
        # if state.is_world_process_zero:
            
        # UNWRAP DDP/DeepSpeed model for evaluation on single GPU
        # if hasattr(model, 'module'):
        #     model = model.module
        
        eval_dataloader = self.unsharded_eval_loader
        additional_dataloader = self.unsharded_train_loader if self.num_additional_ctxs > 0 else None

        with torch.cuda.amp.autocast(cache_enabled=False):
            additional_metrics = self.evaluator(model, eval_dataloader, additional_dataloader)

        # Safe, as we're already only on main process
        if state.is_world_process_zero:
            print(additional_metrics)
            self._wandb.log(additional_metrics, step=state.global_step + 1)
            
        
        # All processes synchronize here before continuing
        if torch.distributed.is_initialized():
            torch.distributed.barrier()


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
        # when passing a tokenizer to the trainer, a DataCollatorWithPadding is automatically instantiated
        super(RetrievalTrainer, self).__init__(
            model=model,
            data_collator=data_collator,
            compute_metrics=retreeval_metrics,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=callbacks,
        )

        self.can_return_loss = True  # key override to log loss
        self.depth_scheduler = depth_scheduler
        self.depth = model.tree_depth
        self.use_custom_scheduler = use_custom_scheduler

    def _save_checkpoint(self, model, trial, **kwargs):
        super()._save_checkpoint(model, trial, **kwargs)
        if self.depth_scheduler is not None:
            import json, os
            ckpt_dir = self._get_output_dir(trial)
            path = os.path.join(ckpt_dir, "depth_scheduler_state.json")
            with open(path, "w") as f:
                json.dump({"depth_scheduler": self.depth_scheduler.state_dict(),
                           "trainer_depth": self.depth}, f)

    def _load_from_checkpoint(self, resume_from_checkpoint, **kwargs):
        super()._load_from_checkpoint(resume_from_checkpoint, **kwargs)
        if self.depth_scheduler is not None:
            import json, os
            path = os.path.join(resume_from_checkpoint, "depth_scheduler_state.json")
            if os.path.exists(path):
                with open(path) as f:
                    state = json.load(f)
                self.depth_scheduler.load_state_dict(state["depth_scheduler"])
                self.depth = state["trainer_depth"]
                print(f"Restored depth scheduler: depth={self.depth}")

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None):
        """
        Log `logs` on the various objects watching training.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """

        # customized logging. track temperature of contrastive loss and avg center of split functions
        loss_tmp = [
            param.item()
            for name, param in self.model.loss.named_parameters()
            if "temp_coef" in name
        ]

        if len(loss_tmp) > 0:
            logs |= {
                "loss_temperature": math.exp(loss_tmp[0]),
            }

        logs |= {
            "question_tree_bias_l2": torch.norm(self.model.query_tree.get_bias(), p=2).item(),
            "context_tree_bias_l2": torch.norm(self.model.context_tree.get_bias(), p=2).item(),
            "loss_depth": self.depth,
        }

        # default logging
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def training_step(self, model, inputs, num_items_in_batch=None):
        if self.state.global_step == 0:
            self.model._init_tree_params(**inputs)

        if self.depth_scheduler is not None:
            depth = self.depth_scheduler.get_depth(self.state.global_step)

            if not isinstance(self.depth_scheduler, RandomDepthScheduler) and self.depth < depth:
                # when changing loss depth, reinit tree params from parents on.
                self.model._init_tree_params(**inputs, depth=depth - 1)

            self.depth = depth
            
        # Override depth if train_full_tree_rep set
        if self.model.train_full_tree_rep:
            self.depth = -1

        # append loss depth to model inputs, needed to compute loss
        inputs |= {"depth": self.depth}
        return super().training_step(model, inputs)
    
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """Override to optionally use custom LR scheduler"""
        if not self.use_custom_scheduler:
            # Use default HuggingFace scheduler (backward compatible)
            return super().create_scheduler(num_training_steps, optimizer)
        
        # Use custom scheduler
        if optimizer is None:
            optimizer = self.optimizer
        
        # max_lr = optimizer.param_groups[0]['lr']
        
        def get_custom_lr_lambda(warmup_steps=5000, decay_steps=50000, min_lr=5e-7, max_lr=1e-4):
            """
            Custom LR scheduler:
            - Steps 0-5000: Linear warmup to max_lr
            - Steps 5000-105000: Linear decay to min_lr
            - Steps 105000+: Constant at min_lr
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
    """Intanstiates Trainer object.

    Args:
        model (PreTrainedModel): Model to be trained.
        data_collator (DataCollator): Data collator.
        cfg (DictConfig): Command line arguments.
        training_args (TrainingArguments): trainer's argument.
        training_data (Dataset): Training dataset.
        validation_data (Dataset): Validation dataset.
        tracked_metrics (Optional): Additional metrics to track at evaluation.

    Returns:
        Trainer: Configured trainer.
    """

    if local_rank == 0:
        wandb_run = wandb.init(
            name=cfg.logging.wandb_run,
            project=cfg.logging.wandb_project,
            mode="disabled" if cfg.debug else None,  # no logging while debugging
            config=omegaconf_to_dict(cfg),
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
            min_value=cfg.train.get("depth_scheduler_min_value", 1),
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


def plot_lr_schedule(trainer, num_steps=None, save_path=None):
    """
    Plot the learning rate schedule for debugging.
    
    Args:
        trainer: The RetrievalTrainer instance
        num_steps: Number of steps to plot (default: total training steps)
        save_path: Optional path to save the plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    if num_steps is None:
        num_steps = trainer.args.max_steps if trainer.args.max_steps > 0 else len(trainer.train_dataset) // trainer.args.per_device_train_batch_size * trainer.args.num_train_epochs
    
    # Get the base LR
    base_lr = trainer.optimizer.param_groups[0]['lr']
    
    # Collect LR values for each step
    lrs = []
    steps = list(range(num_steps))
    
    for step in steps:
        # Get the LR multiplier from scheduler
        if hasattr(trainer.lr_scheduler, 'lr_lambdas'):
            # LambdaLR scheduler
            lr_mult = trainer.lr_scheduler.lr_lambdas[0](step)
            lr = base_lr * lr_mult
        else:
            # Other schedulers - step through manually
            # This is a bit hacky but works for visualization
            trainer.lr_scheduler.last_epoch = step - 1
            lr = trainer.lr_scheduler.get_last_lr()[0]
        
        lrs.append(lr)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(steps, lrs, linewidth=2)
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Learning Rate Schedule', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale often better for LR
    
    # Add vertical lines at key points
    if trainer.use_custom_lr_scheduler:
        plt.axvline(x=trainer.custom_lr_warmup_steps, color='r', linestyle='--', alpha=0.5, label='Warmup End')
        plt.axvline(x=trainer.custom_lr_warmup_steps + trainer.custom_lr_decay_steps, color='g', linestyle='--', alpha=0.5, label='Decay End')
        plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"LR schedule plot saved to {save_path}")
    
    plt.show()
    
    # Print some statistics
    print(f"\n=== LR Schedule Statistics ===")
    print(f"Initial LR: {lrs[0]:.2e}")
    print(f"Max LR: {max(lrs):.2e} at step {np.argmax(lrs)}")
    print(f"Final LR: {lrs[-1]:.2e}")
    print(f"Total steps: {num_steps}")
    
    return steps, lrs
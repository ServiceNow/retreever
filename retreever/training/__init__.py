"""Training infrastructure."""

from retreever.training.trainer import RetrievalTrainer, EvaluateCallback, get_trainer
from retreever.training.depth_schedulers import (
    LinearDepthScheduler,
    ExponentialDepthScheduler,
    RandomDepthScheduler,
    RandomHeavyTailedDepthScheduler,
    RandomUniformDepthScheduler,
    KNOWN_SCHEDULERS,
)

__all__ = [
    # Trainer
    "RetrievalTrainer",
    "EvaluateCallback",
    "get_trainer",
    # Depth schedulers
    "LinearDepthScheduler",
    "ExponentialDepthScheduler",
    "RandomDepthScheduler",
    "RandomHeavyTailedDepthScheduler",
    "RandomUniformDepthScheduler",
    "KNOWN_SCHEDULERS",
]

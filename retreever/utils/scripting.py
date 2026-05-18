from typing import NamedTuple
import argparse
import os
import subprocess
import random
import numpy as np
import torch

"""Utilities intended for 'entrypoint' scripts.

IN the scope of this file:
- Helpers for parsing command line arguments.
- Set/get/handle the environment/secrets/logs/etc. for torch/deepspeed/wandb/etc.

OUT of this file's scope:
- Paths/constants/etc.
- Handling "actual work" for data/models/training/evaluation/etc.
"""


# ******** Command line arguments ********


def parse_bool_flag(s: str) -> bool:
    """Interpret a string (from a command line arguments) as a boolean value"""
    _FALSY_STRINGS = {"off", "false", "no", "0"}
    _TRUTHY_STRINGS = {"on", "true", "yes", "1"}
    if s.lower() in _FALSY_STRINGS:
        return False
    elif s.lower() in _TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag")


# ******** torch.distributed, deepspeed etc. ********


class LocalRankAndWorldSize(NamedTuple):
    local_rank: int
    world_size: int


def get_local_rank_and_world_size() -> LocalRankAndWorldSize:
    """Source of truth for rank and world size.

    Our convention is that the environment variables LOCAL_RANK and WORLD_SIZE
    are the underlying source of truth, which should be properly set
    automatically by torchrun/deepspeed/etc.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    assert 0 <= local_rank < world_size
    if torch.distributed.is_initialized():
        assert torch.distributed.get_rank() == local_rank
        assert torch.distributed.get_world_size() == world_size
    return LocalRankAndWorldSize(local_rank=local_rank, world_size=world_size)


def print_rank_0(logger, message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            logger(message)
    else:
        logger(message)


# ******** Compute environment ********


def set_random_seed(seed: int = 42) -> None:
    """Set random seeds."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ******** Git ********


def is_git_clean() -> str:
    return len(subprocess.check_output(["git", "status", "--porcelain=v1", "2>/dev/null"])) == 0


def get_git_sha(assert_git_clean: bool = True) -> str:
    if assert_git_clean:
        assert is_git_clean(), "Unclean git! Add and/or commit your changes and try again."
    raw = subprocess.check_output(["git", "rev-parse", "HEAD"])
    try:
        sha = raw.decode()
        assert sha[-1] == "\n"
        sha = sha[:-1]
        assert len(sha) == 40
        assert all(hex in "0123456789abcdef" for hex in sha)
    except AssertionError:
        raise RuntimeError(f"Call to git returned: {raw}")
    return sha

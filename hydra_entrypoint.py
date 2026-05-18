import argparse

from hydra import compose, initialize

from retreever.utils.scripting import get_local_rank_and_world_size
from scripts.train import train as retreever_main
from scripts.train import CONFIG_NAME as retreever_cfg


# @hydra.main(version_base=None, config_path="scripts/config/", config_name="train.yaml")
def main():
    """Root-directory entrypoint script for the hydra-configured training run.

    Our scripts are modules meant to be called from the repo's root folder using python's `-m` argument, e.g.,

    ```
    python3 -m scripts.train [rest of the arguments here]
    ```

    However, torchrun and deepspeed do not support this `-m` feature: they need an actual python script as input.
    The present script fills that purpose. To launch with deepspeed:

    ```
    deepspeed --num_gpus 1 hydra_entrypoint.py --deepspeed=scripts/config/deepspeed.json [rest of the arguments here]
    ```
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deepspeed",
        type=str,
        help="Config for deepspeed.",
        default="scripts/config/deepspeed.json",
    )
    parser.add_argument(
        "--local_rank",
        # deepspeed automatically adds a local_rank argument to the call. We ignore it and use the environment variable instead in the script.
        type=int,
        help="Ignore this argument: the only source of truth is the `LOCAL_RANK` environment variable. If this argument is present, we assert that it matches the environment variable, then we ignore it. The only reason for this argument to be here is to be 'swallowed' from the command line arguments, preventing it to be passed to the train main.",
    )
    parsed, unparsed = parser.parse_known_args()

    if parsed.local_rank is not None:
        # The source of truth is the environment variable LOCAL_RANK.
        local_rank, _ = get_local_rank_and_world_size()
        assert parsed.local_rank == local_rank

    unparsed.append(f"+deepspeed={parsed.deepspeed}")

    initialize(version_base=None, config_path="scripts/config/")
    cfg = compose(config_name=retreever_cfg, overrides=unparsed)

    cfg.deepspeed = parsed.deepspeed

    retreever_main(cfg)


if __name__ == "__main__":
    main()

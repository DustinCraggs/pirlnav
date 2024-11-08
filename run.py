#!/usr/bin/env python3
import argparse
import os
import random
from datetime import datetime

import numpy as np
import numba
import quaternion
import torch
import habitat

from habitat import logger
from habitat.config import Config
from habitat_baselines.common.baseline_registry import baseline_registry

from pirlnav.config import get_config
from pirlnav.gen_representation_dataset import (
    RepresentationGenerator,
    generate_episode_split_index,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval", "gen", "gen_sub_split"],
        required=True,
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Set the random seed for the experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()
    run_exp(**vars(args))


def execute_exp(config: Config, run_type: str, seed=None) -> None:
    r"""This function runs the specified config with the specified runtype
    Args:
    config: Habitat.config
    runtype: str {train or eval}
    """
    if seed is None:
        # set a random seed (from detectron2)
        seed = (
            os.getpid()
            + int(datetime.now().strftime("%S%f"))
            + int.from_bytes(os.urandom(2), "big")
        )
        logger.info("Using a generated random seed {}".format(seed))
    else:
        logger.info("Using seed {}".format(seed))

    config.defrost()
    config.RUN_TYPE = run_type
    config.TASK_CONFIG.SEED = seed
    config.freeze()
    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)
    if config.FORCE_TORCH_SINGLE_THREADED and torch.cuda.is_available():
        torch.set_num_threads(1)

    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)

    if run_type == "train":
        trainer.train()
    elif run_type == "eval":
        trainer.eval()
    elif run_type == "gen":
        generator = RepresentationGenerator(config)
        generator.generate()
    elif run_type == "gen_sub_split":
        generate_episode_split_index(config)


def run_exp(exp_config: str, run_type: str, seed=None, opts=None) -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.

    Returns:
        None.
    """
    config = get_config(exp_config, opts)
    execute_exp(config, run_type, seed)


if __name__ == "__main__":
    main()

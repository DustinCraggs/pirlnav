#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import json
import os
import random
from typing import Any, List, Type, Union

import habitat
from habitat import Config, Env, RLEnv, VectorEnv, logger, make_dataset
import numpy as np


def make_env_fn(
    config: Config,
    env_class: Union[Type[Env], Type[RLEnv]],
    episode_index=None,
) -> Union[Env, RLEnv]:
    r"""Same as habitat.utils.env_utils.make_env_fn, but with options to sort episodes
    by scene and goal, and to sample every EPISODE_STRIDE-th episode.
    """
    if "TASK_CONFIG" in config:
        config = config.TASK_CONFIG
    dataset = make_dataset(config.DATASET.TYPE, config=config.DATASET)

    if episode_index is not None:
        dataset.episodes = filter_episodes(dataset.episodes, episode_index)
        print(
            f"Creating env with {len(dataset.episodes)} episodes "
            f"({get_num_unique(dataset.episodes)} unique scene-object pairs)"
        )

    env = env_class(config=config, dataset=dataset)
    env.seed(config.SEED)
    return env


def get_num_unique(episodes):
    return len(set((ep.scene_id, ep.object_category) for ep in episodes))


def generate_dataset_split_json(config: Config, output_path, stride) -> None:
    r"""Generate a dataset split json file for the given config and save it to the
    output path. This is useful for generating a dataset split json file for a
    dataset that is not natively supported by Habitat Lab.

    :param config: Config object that contains the dataset information.
    :param output_path: Path to save the dataset split json file.
    """
    dataset_config = config.TASK_CONFIG.DATASET
    dataset = make_dataset(dataset_config.TYPE, config=dataset_config)
    scenes = dataset_config.CONTENT_SCENES
    if "*" in dataset_config.CONTENT_SCENES:
        scenes = dataset.get_scenes_to_load(dataset_config)

    episodes = dataset.episodes
    print(f"Total number of episodes before: {len(episodes)}")
    num_unique_pairs_before = get_num_unique(episodes)
    episodes.sort(key=lambda ep: (ep.scene_id, ep.object_category))
    episodes = episodes[::stride]
    num_unique_pairs_after = get_num_unique(episodes)

    print("Unique scene-object pairs:")
    print(f"\tBefore: {num_unique_pairs_before}")
    print(f"\tAfter: {num_unique_pairs_after}")

    # TODO: Shuffle object categories? Shuffle all then sort by scene again. Contiguous
    # scenes will improve performance.
    episode_ids = [
        {
            "scene_id": ep.scene_id,
            "episode_id": ep.episode_id,
            "object_category": ep.object_category,
        }
        for ep in episodes
    ]

    print(f"Total number of episodes: {len(episode_ids)}")
    print(f"Number of scenes: {len(scenes)}")

    with open(output_path, "w") as f:
        json.dump(episode_ids, f)


def filter_episodes(episodes, episode_index):
    keys = ["scene_id", "episode_id", "object_category"]
    episode_index = set([tuple(ep[k] for k in keys) for ep in episode_index])
    return [
        ep
        for ep in episodes
        if (ep.scene_id, ep.episode_id, ep.object_category) in episode_index
    ]


def construct_envs(
    config: Config,
    env_class: Union[Type[Env], Type[RLEnv]],
    workers_ignore_signals: bool = False,
    shuffle_scenes: bool = True,
    episode_index=None,
) -> VectorEnv:
    r"""Create VectorEnv object with specified config and env class type.
    To allow better performance, dataset are split into small ones for
    each individual env, grouped by scenes.

    :param config: configs that contain num_environments as well as information
    :param necessary to create individual environments.
    :param env_class: class type of the envs to be created.
    :param workers_ignore_signals: Passed to :ref:`habitat.VectorEnv`'s constructor

    :return: VectorEnv object created according to specification.
    """

    num_environments = config.NUM_ENVIRONMENTS
    configs = []
    env_classes = [env_class for _ in range(num_environments)]

    dataset_config = config.TASK_CONFIG.DATASET
    
    dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE)
    scenes = dataset_config.CONTENT_SCENES
    if "*" in dataset_config.CONTENT_SCENES:
        scenes = dataset.get_scenes_to_load(dataset_config)

    if num_environments < 1:
        raise RuntimeError("NUM_ENVIRONMENTS must be strictly positive")

    if len(scenes) == 0:
        raise RuntimeError(
            "No scenes to load, multiple process logic relies on being able to split "
            "scenes uniquely between processes"
        )

    if shuffle_scenes:
        random.shuffle(scenes)

    scene_splits: List[List[str]] = [[] for _ in range(num_environments)]
    if len(scenes) < num_environments:
        logger.warn(
            f"There are less scenes ({len(scenes)}) than environments ({num_environments})"
            "Each environment will use all the scenes instead of using a subset. "
        )
        for scene in scenes:
            for split in scene_splits:
                split.append(scene)
    else:
        for idx, scene in enumerate(scenes):
            scene_splits[idx % len(scene_splits)].append(scene)
        assert sum(map(len, scene_splits)) == len(scenes)

    for i in range(num_environments):
        proc_config = config.clone()
        proc_config.defrost()

        task_config = proc_config.TASK_CONFIG
        task_config.SEED = task_config.SEED + i
        if len(scenes) > 0:
            task_config.DATASET.CONTENT_SCENES = scene_splits[i]

        task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = config.SIMULATOR_GPU_ID

        task_config.SIMULATOR.AGENT_0.SENSORS = config.SENSORS

        proc_config.freeze()
        configs.append(proc_config)

    vector_env_cls: Type[Any]
    if os.environ.get("HABITAT_ENV_DEBUG", 0):
        logger.warn(
            "Using the debug Vector environment interface. Expect slower performance."
        )
        vector_env_cls = habitat.ThreadedVectorEnv
    else:
        vector_env_cls = habitat.VectorEnv

    ep_indexes = itertools.repeat(episode_index)
    envs = vector_env_cls(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(zip(configs, env_classes, ep_indexes)),
        workers_ignore_signals=workers_ignore_signals,
    )
    return envs

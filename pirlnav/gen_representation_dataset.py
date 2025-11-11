import gzip
import json
import logging
import os
import pickle
import re
import time
import traceback
import types
import cv2
import einops
import numpy as np

# import ray
import torch
import torchvision
import zarr
import sys
import tqdm
import requests
import io
import habitat_sim
import networkx as nx

from enum import Enum
from concurrent.futures import ProcessPoolExecutor

from habitat_sim.utils.common import quat_to_magnum
from habitat.sims.habitat_simulator.actions import HabitatSimV0ActionSpaceConfiguration
from pirlnav.environment import SimpleRLEnv
from pirlnav.utils.env_utils import construct_envs, generate_dataset_split_json
from habitat.core.vector_env import CURRENT_EPISODE_NAME
from habitat.core.registry import registry

# from vc_models.models.vit import model_utils, vit
# from transformers import CLIPVisionModel, AutoProcessor
from scipy.spatial.transform import Rotation
from PIL import Image


# TODO: Temporary hack as these are not accessible as installable packages:
# sys.path.append("/storage/dc/sg/sg_habitat")
sys.path.append("/home/dc/sg_new/sg_habitat")
from libs.mapper.map_incremental import IncrementalMapper, SimIncrementalMapper
from libs.mapper.parallel_mapper import (
    SimIncrementalMapper as RemoteSimIncrementalMapper,
)
from scripts.exploration.models import load_checkpoint
from scripts.exploration.graph_dataset import get_cost_matrix, get_objectnav_cost_matrix
from scripts.exploration.descriptor_generator import ClipDescriptorGenerator
from libs.planner_global.plan_topo import predict_path_lengths

# from libs.experiments import model_loader
# from libs.matcher import lightglue as matcher_lg
# from scripts.exploration.create_training_maps import contract_da_edges


# TODO: Change generator to return dict of tensors?
# TODO: DataClass to hold the data and metadata for each episode step
# TODO: Consistent json ep index output and filtering of already-completed episodes
OBJECTNAV_GOALS = ["chair", "bed", "toilet", "plant", "tv_monitor", "sofa"]
# OBJECTNAV_GOAL_REMAPPING = {
#     "chair": "chair",
#     "bed": "bed",
#     "toilet": "toilet",
#     "plant": "plant",
#     "tv": "tv_monitor",
#     "monitor": "tv_monitor",
#     "sofa": "sofa",
#     "couch": "sofa",
# }


def generate_episode_split_index(config):
    output_path = config["TASK_CONFIG"]["SUB_SPLIT_GENERATOR"]["INDEX_PATH"]
    stride = config["TASK_CONFIG"]["SUB_SPLIT_GENERATOR"]["STRIDE"]
    start_idx = config["TASK_CONFIG"]["SUB_SPLIT_GENERATOR"]["START_IDX"]
    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # generate_dataset_split_json(config, output_path, stride, object_classes=["toilet"])
    generate_dataset_split_json(config, output_path, stride, start_idx)


def get_data_generators(config, num_envs):
    data_generator_registry = {
        "non_visual": NonVisualObservationsGenerator,
        "raw_image": RawImageGenerator,
        "ground_truth_costmap": GroundTruthCostmapImageGenerator,
        "ground_truth_perception_graph": GroundTruthPerceptionGraphGenerator,
        "predicted_costmap": PredictedCostmapImageGenerator,
        "costmap_visualisation": CostmapVisualisationGenerator,
        "clip": ClipGenerator,
        "vc_1": Vc1Generator,
        "cogvlm2": CogVlmGenerator,
        "agent_state": AgentStateGenerator,
    }

    generator_config = config["TASK_CONFIG"]["REPRESENTATION_GENERATOR"]

    return [
        data_generator_registry[name](
            num_envs=num_envs, **args if args is not None else {}
        )
        for name, args in generator_config["data_generators"].items()
    ]


def get_data_storage(config, episodes):
    storage_registry = {
        "zarr": ZarrDataStorage,
        "nomad": NomadDataStorage,
        "graph": GraphDataStorage,
        "video": VideoDataStorage,
    }

    storage_config = config["TASK_CONFIG"]["REPRESENTATION_GENERATOR"]["data_storage"]
    storage_kwargs = storage_config.get("kwargs", {})

    output_path = storage_config["output_path"]
    return storage_registry[storage_config["name"]](
        output_path, episodes, **storage_kwargs
    )


class CustomEnv(SimpleRLEnv):
    """env._env._sim.semantic_scene is not pickleable. We only need to get a mapping
    from IDs (in the semantic image observation) to class labels, however, so this
    custom env patches that in. Implementation is based on ImageExtractor.
    """

    def get_semantic_instance_id_to_name_map(self):
        semantic_scene = self._env._sim.semantic_scene

        instance_id_to_name = {}
        for obj in semantic_scene.objects:
            if obj and obj.category:
                obj_id = int(obj.id.split("_")[-1])
                instance_id_to_name[obj_id] = (obj.category.name(), obj.aabb.center)

        return instance_id_to_name

    def get_object_centers(self):
        semantic_scene = self._env._sim.semantic_scene
        instance_id_to_ob_center = {}
        for instance in semantic_scene.objects:
            instance_id = int(instance.id.split("_")[-1])
            instance_id_to_ob_center[instance_id] = instance.obb.center

        return instance_id_to_ob_center


@registry.register_action_space_configuration(name="v1_no_op_look")
class NoOpLookActionSpaceConfiguration(HabitatSimV0ActionSpaceConfiguration):
    """Makes look actions no-ops."""

    def get(self):
        config = super().get()
        new_config = {
            4: habitat_sim.ActionSpec(
                "look_up",
                habitat_sim.ActuationSpec(amount=0),
            ),
            5: habitat_sim.ActionSpec(
                "look_down",
                habitat_sim.ActuationSpec(amount=0),
            ),
        }

        config.update(new_config)

        return config


class VideoDataStorage:
    def __init__(self, output_path, episodes, fps=10, **kwargs):
        self._output_path = output_path
        self._fps = fps
        self._completed_eps = []

    def save_episode(self, data, scene_id, episode_id, object_category):
        scene_id = scene_id.split("/")[-2]
        ep_id = f"{scene_id}_{episode_id}_{object_category}"
        output_path = f"{self._output_path}/{ep_id}.mp4"

        os.makedirs(self._output_path, exist_ok=True)

        # Inpaint key text on images:
        for d in data:
            for k, img in d.items():
                cv2.putText(
                    np.ascontiguousarray(img, dtype=np.uint8),
                    k,
                    (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

        data = {k: np.stack([d[k] for d in data]) for k in data[0]}

        # Stitch all data into a grid:
        # Assumes all images are the same size:
        images = np.stack([v for v in data.values()])
        images = einops.rearrange(images, "(b1 b2) t h w c -> t (b1 h) (b2 w) c", b1=2)

        height, width = images[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(output_path, fourcc, self._fps, (width, height))

        for img in images:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            video_writer.write(img_bgr)

        video_writer.release()

        self._completed_eps.append(ep_id)

    def close(self):
        pass


class GraphDataStorage:

    def __init__(self, output_path, episodes, num_workers=2, **kwargs):
        self._output_path = output_path
        # self._should_save_caches = should_save_caches
        # self._cache_save_interval_eps = cache_save_interval_eps

        self._completed_eps = []

        # self._pool = ProcessPoolExecutor(num_workers)
        # self._futures = []

    def save_episode(self, data, scene_id, episode_id, object_category):
        # Only save the final graphs and caches:
        data = data[-1]

        scene_id = scene_id.split("/")[-2]
        ep_id = f"{scene_id}_{episode_id}_{object_category}"
        output_path = f"{self._output_path}/{ep_id}"

        os.makedirs(output_path, exist_ok=True)

        self._write_graphs(data, output_path)
        # future = self._pool.submit(self._write_graphs, data, output_path)
        # self._futures.append(future)

        self._completed_eps.append(ep_id)

        # Save the episode index in case of early termination:
        ep_index_path = f"{self._output_path}/ep_index.txt"
        with open(ep_index_path, "a") as f:
            for ep in self._completed_eps:
                f.write(f"{ep}\n")

        # try:
        #     result = future.result()
        # except Exception as e:
        #     print(f"Task raised an exception: {e}")

    @staticmethod
    def _write_graphs(data, output_path):
        graph_keys = [k for k in data.keys() if "graph" in k]
        for k in graph_keys:
            save_path = f"{output_path}/{k}.pkl.gz"
            with gzip.open(save_path, "wb") as f:
                pickle.dump(data[k], f)

        # num_eps = len(self._completed_eps)
        # if self._should_save_caches and num_eps % self._cache_save_interval_eps:
        #     self._save_caches(data)

    # def _save_caches(self, data):
    #     # The cache is shared across all eps:
    #     cache_keys = [k for k in data.keys() if "cache" in k]
    #     for k in cache_keys:
    #         save_path = f"{self._output_path}/{k}.pkl"
    #         with open(save_path, "wb") as f:
    #             pickle.dump(data[k], f)

    def close(self):
        pass
        # for f in self._futures:
        #     f.result()
        # self._pool.shutdown()
        # if self._should_save_caches:
        #     # Save the final cache:
        #     self._save_caches(self._last_data[-1])


class ZarrDataStorage:

    def __init__(self, output_path, episodes, batch_chunk_size=1000, **kwargs):
        self._output_path = output_path
        self._batch_chunk_size = batch_chunk_size

        self._zarr_file = None
        self._completed_eps = []

        # The episode index maps the scene_id, episode_id, and object category to the
        # row in the zarr file:
        lengths = [ep["length"] for ep in episodes]
        cumulative_lengths = np.cumsum([0, *lengths])

        self._total_length = cumulative_lengths[-1]
        self._episode_index = {
            (ep["scene_id"], ep["episode_id"], ep["object_category"]): (
                row,
                ep["length"],
            )
            for ep, row in zip(episodes, cumulative_lengths)
        }

        print(f"Num eps {len(episodes)}")
        print(f"Total length {self._total_length}")

    def _init_zarr_file(self, data):
        self._zarr_file = zarr.open(self._output_path, mode="w")
        self._data_group = self._zarr_file.create_group("data")
        # In older versions of zarr, "meta*" names are reserved, so prefix with "_":
        self._meta_group = self._zarr_file.create_group("_meta")

        for key, data_array in data.items():
            chunks = [None] * len(data_array.shape)
            chunks[0] = self._batch_chunk_size

            self._data_group[key] = zarr.create(
                shape=(self._total_length, *data_array.shape[1:]),
                dtype=data_array.dtype,
                chunks=chunks,
            )

    def save_episode(self, data, scene_id, episode_id, object_category):
        row, expected_length = self._episode_index[
            (scene_id, episode_id, object_category)
        ]

        data = {k: np.stack([d[k] for d in data]) for k in data[0]}

        data_length = data[next(iter(data.keys()))].shape[0]
        assert (
            data_length == expected_length
        ), f"Episode length mismatch. Expected: {expected_length}, got: {data_length}"

        if self._zarr_file is None:
            # Create the zarr file and groups and save the data:
            self._init_zarr_file(data)

        # Append the data to the existing zarr file:
        for key, data_array in data.items():
            self._data_group[key][row : row + expected_length] = data_array

        # Update the completed episode index:
        self._completed_eps.append(
            {
                "scene_id": scene_id,
                "episode_id": episode_id,
                "object_category": object_category,
                "row": int(row),
                "length": int(data_length),
            }
        )

        # Save the episode index in case of early termination:
        # TODO: Move to superclass:
        ep_index_path = f"{self._output_path}/ep_index.json"
        json.dump(self._completed_eps, open(ep_index_path, "w"))

    def close(self):
        pass


class NomadDataStorage:

    def __init__(
        self, output_path, episodes, num_workers=10, save_waypoint_plot=False, **kwargs
    ):
        self._output_path = output_path
        self._completed_eps = []
        self._save_waypoint_plot = save_waypoint_plot

        self._pool = ProcessPoolExecutor(num_workers)
        self._futures = []

    def save_episode(self, data, scene_id, episode_id, object_category):
        """
        This data storage looks for specific data keys and saves them in the format
        expected by the NoMaD training code.
        """
        scene_id = scene_id.split("/")[-2]
        ep_id = f"{scene_id}_{episode_id}_{object_category}"
        save_dir = f"{self._output_path}/{ep_id}"

        data = {k: np.stack([d[k] for d in data]) for k in data[0]}

        os.makedirs(save_dir, exist_ok=True)

        positions = data["position"]
        yaws = data["yaw"]

        # Save the data:
        f = self._pool.submit(
            self._write_traj_data,
            save_dir,
            np.array(data["rgb"]),
            np.array(data["agent_state"]),
            positions,
            yaws,
        )
        self._futures.append(f)

        # Update traj_names.txt. Write entire file every time in case there happens
        # to already be a file with the same name:
        self._completed_eps.append(f"{ep_id}")
        ep_index_path = f"{self._output_path}/traj_names.txt"
        with open(ep_index_path, "w") as f:
            for ep in self._completed_eps:
                f.write(f"{ep}\n")

        if self._save_waypoint_plot:
            self.plot_traj(save_dir, positions, yaws)

    @staticmethod
    def _write_traj_data(save_dir, rgb_data, agent_states, positions, yaws):
        try:
            os.makedirs(f"{save_dir}/images/", exist_ok=True)

            images = [Image.fromarray(arr) for arr in rgb_data]
            for i, img in enumerate(images):
                img.save(f"{save_dir}/images/{i}.png")

            np.save(f"{save_dir}/agent_states.npy", agent_states)

            traj_data = {
                "position": positions,
                "yaw": yaws,
            }
            pickle.dump(traj_data, open(f"{save_dir}/traj_data.pkl", "wb"))
        except Exception as e:
            print(f"Error saving episode to {save_dir}: {e}")
            with open(f"{save_dir}/error.txt", "w") as f:
                f.write(str(e))
                # Save stack trace:
                f.write("\n")
                f.write(traceback.format_exc())

    @staticmethod
    def plot_traj(save_dir, positions, yaws):
        import matplotlib.pyplot as plt

        fig = plt.gcf()
        fig.set_size_inches(20, 20)

        plt.plot(positions[:, 0], positions[:, 1], marker=".")
        # Plot yaws as arrows:
        for i, (pos, yaw) in enumerate(zip(positions[:], yaws[:])):
            plt.arrow(
                pos[0],
                pos[1],
                0.1 * np.cos(yaw),
                0.1 * np.sin(yaw),
                width=0.005,
                color="red",
            )

        # Plot initial and final position in green:
        plt.plot(positions[0, 0], positions[0, 1], marker="o", color="green")
        plt.plot(positions[-1, 0], positions[-1, 1], marker="o", color="orange")

        # Square axes:
        plt.gca().set_aspect("equal", adjustable="box")
        # plt.show()
        plt.savefig(f"{save_dir}/waypoints.png", dpi=300)
        plt.close()

    def close(self):
        for f in self._futures:
            f.result()
        self._pool.shutdown()


class RepresentationGenerator:

    def __init__(self, config):
        self._envs, episodes = self._init_envs(config)
        self._num_envs = config["NUM_ENVIRONMENTS"]

        self._remaining_ep_set = set(
            (ep["scene_id"], ep["episode_id"], ep["object_category"]) for ep in episodes
        )

        self._data_generators = get_data_generators(config, self._num_envs)
        self._data_storage = get_data_storage(config, episodes=episodes)

        rep_gen_config = config["TASK_CONFIG"]["REPRESENTATION_GENERATOR"]

        self._store_last_data_only = False
        if rep_gen_config["data_storage"]["name"] == "graph":
            self._store_last_data_only = True

        self._skip_non_movement_actions = config["TASK_CONFIG"][
            "REPRESENTATION_GENERATOR"
        ]["skip_non_movement_actions"]

        self._skip_look_actions = rep_gen_config["skip_look_actions"]

        self._generate_skip_index = rep_gen_config["generate_skip_index"]

        if self._generate_skip_index:
            self._non_movement_ep_index = episodes.copy()
            ep_index_path = config["TASK_CONFIG"]["DATASET"]["SUB_SPLIT_INDEX_PATH"]
            self._non_movement_ep_index_path = (
                f"{ep_index_path[:-5]}_no_look_actions.json"
            )

    def _init_envs(self, config):
        config.defrost()
        config.TASK_CONFIG.TASK.SENSORS.extend(
            ["DEMONSTRATION_SENSOR", "INFLECTION_WEIGHT_SENSOR"]
        )
        config.freeze()

        sub_split_index_path = config["TASK_CONFIG"]["DATASET"]["SUB_SPLIT_INDEX_PATH"]
        with open(sub_split_index_path, "r") as f:
            sub_split_index = json.load(f)

        filter_existing_path = config["TASK_CONFIG"]["DATASET"]["FILTER_EXISTING_PATH"]
        if filter_existing_path is not None:
            with open(filter_existing_path, "r") as f:
                lines = f.readlines()
            # The completed episode tracker uses a non-standard format:
            matcher = re.compile(r"(.*)_(\d+)_(\w+)")
            filter_eps = set(matcher.match(line).groups() for line in lines)

            sub_split_index = [
                ep
                for ep in sub_split_index
                if (
                    ep["scene_id"].split("/")[-2],
                    ep["episode_id"],
                    ep["object_category"],
                )
                not in filter_eps
            ]

        ep_keys = ["scene_id", "episode_id", "object_category", "length"]
        episodes = [{k: ep[k] for k in ep_keys} for ep in sub_split_index]

        # env_cls = get_env_class(config["ENV_NAME"])
        envs = construct_envs(
            config,
            CustomEnv,
            workers_ignore_signals=False,
            shuffle_scenes=False,
            episode_index=sub_split_index,
        )

        return envs, episodes

    def generate(self):
        # TODO: With a stride of 10, one scene-goal pair is missing
        total_num_eps = sum(self._envs.count_episodes())
        print(f"Number of episodes: {total_num_eps}")

        # Track the data for each episode separately:
        rollout_data = [[] for _ in range(self._num_envs)]

        actions = [None] * self._num_envs
        observations = self._envs.reset()
        rewards = [0.0] * self._num_envs
        # Done is true on the first step of each episode (non-standard convention):
        dones = [True] * self._num_envs
        should_skips = [False] * self._num_envs
        movement_step_counts = [0] * self._num_envs

        movement_ep_lengths = {}

        # Track the episode info to deduplicate episodes from cycling environments:
        current_episodes = self._envs.current_episodes()

        step_data = self._generate_step(
            current_episodes, actions, observations, rewards, dones, None, should_skips
        )
        for ep, data in zip(rollout_data, step_data):
            ep.append(data)

        c = 0

        pbar = tqdm.tqdm(total=total_num_eps)
        while self._remaining_ep_set:
            # "next_actions" contains the actions from the BC dataset:
            actions = [o["next_actions"] for o in observations]

            prev_obs = observations
            # TODO: Manually pick look up and look down actions to ensure they're
            # no-ops:
            outputs = self._envs.step(actions)
            observations, rewards, dones, infos = zip(*outputs)

            should_skips = [
                self._should_skip(actions[i], prev_obs[i], dones[i], observations[i])
                for i in range(self._num_envs)
            ]

            # If done, save the episode (the current obs is for the next ep):
            for i, ep in enumerate(rollout_data):
                ep_metadata = current_episodes[i]
                done = dones[i]

                # TODO: Need to change this line if removing all non-movement actions:
                movement_step_counts[i] += not self._is_look_action(actions[i])

                if done:
                    # Check if this is a duplicate episode (i.e. an environment that
                    # cycled through all its eps):
                    ep_key = (
                        ep_metadata.scene_id,
                        ep_metadata.episode_id,
                        ep_metadata.object_category,
                    )
                    if ep_key in self._remaining_ep_set:
                        self._remaining_ep_set.remove(ep_key)
                        # Each element of ep is a dict. Concat each value into a
                        # separate array:
                        self._data_storage.save_episode(ep, *ep_key)

                        # Update non-movement ep index:
                        movement_ep_lengths[ep_key] = movement_step_counts[i]
                        movement_step_counts[i] = 0
                        pbar.update(1)
                        c += 1
                        if c > 200:
                            exit()
                    else:
                        print(f"Duplicate episode {ep_key} (skipping)")

                    ep.clear()
                    # This env is now on a new ep, so update its metadata:
                    current_episodes[i] = self._envs.call_at(i, CURRENT_EPISODE_NAME)

            step_data = self._generate_step(
                current_episodes,
                actions,
                observations,
                rewards,
                dones,
                infos,
                should_skips,
            )

            for ep, data, should_skip in zip(rollout_data, step_data, should_skips):
                if not should_skip:
                    ep.append(data)

                    if self._store_last_data_only:
                        # Keep only the last step's data:
                        while len(ep) > 1:
                            ep.pop(0)

        if self._generate_skip_index:
            # Save the ep index with lengths adjusted to only include non-movement
            # actions:
            for ep in self._non_movement_ep_index:
                ep["length"] = int(
                    movement_ep_lengths[
                        (ep["scene_id"], ep["episode_id"], ep["object_category"])
                    ]
                )
            print(f"Num non-movement eps: {len(self._non_movement_ep_index)}")
            with open(self._non_movement_ep_index_path, "w") as f:
                json.dump(self._non_movement_ep_index, f)

        pbar.close()
        self._data_storage.close()

    def _generate_step(
        self, ep_metadata, actions, observations, rewards, dones, infos, skipped_last
    ):
        data = {}
        for data_generator in self._data_generators:
            output = data_generator.generate(
                ep_metadata,
                actions,
                observations,
                rewards,
                dones,
                infos,
                self._envs,
                skipped_last,
            )
            data.update(output)

        ep_data = [
            {k: v[env_idx] for k, v in data.items()}
            for env_idx in range(self._num_envs)
        ]
        return ep_data

    def _get_final_data(self, env_idx):
        final_data = {}
        for data_generator in self._data_generators:
            if hasattr(data_generator, "get_final_data"):
                output = data_generator.get_final_data(env_idx)
                final_data.update(output)
        return final_data

    def _should_skip(self, prev_action, prev_obs, prev_done, obs):
        if self._skip_look_actions:
            return self._is_look_action(prev_action)
        return False

    def _is_non_movement_action(self, prev_action, prev_obs, prev_done, obs):
        # TODO: This is no longer used. Testing navigability using collisions seems
        # fundamental to the ObjectNav HD demo behaviours.
        # Never skip the first step:
        if prev_done:
            return False

        # Skip look actions:
        if self._is_look_action(prev_action):
            return True

        # Skip if the agent didn't move and move forward was the action:
        prev_action = HabitatSimActions(prev_action)
        move_distance = np.linalg.norm(prev_obs["gps"] - obs["gps"])
        return prev_action == HabitatSimActions.MOVE_FORWARD and move_distance < 0.1

    def _is_look_action(self, prev_action):
        # Skip look actions:
        prev_action = HabitatSimActions(prev_action)
        return prev_action in [HabitatSimActions.LOOK_DOWN, HabitatSimActions.LOOK_UP]


class RawImageGenerator:

    def __init__(self, resize_and_crop_to=None, **kwargs):
        self.data_names = ["rgb"]

        self._transforms = []
        if resize_and_crop_to:
            self._transforms.append(torchvision.transforms.Resize(resize_and_crop_to))
            self._transforms.append(
                torchvision.transforms.CenterCrop(resize_and_crop_to)
            )

    def generate(
        self,
        ep_metadata,
        actions,
        observations,
        rewards,
        dones,
        infos,
        envs,
        skipped_last,
    ):
        images = [o["rgb"] for o in observations]

        if self._transforms:
            images = self._apply_transforms(images)

        return {"rgb": images}

    def _apply_transforms(self, images):
        images = [torch.as_tensor(img).permute(2, 0, 1) for img in images]
        for transform in self._transforms:
            images = [transform(img) for img in images]
        images = [img.permute(1, 2, 0) for img in images]
        return images


class GroundTruthCostmapImageGenerator(RawImageGenerator):

    def __init__(self, num_envs, resize_and_crop_to=None, **kwargs):
        super().__init__(resize_and_crop_to=resize_and_crop_to, **kwargs)
        self.data_names = ["goal_costmap", "gt_costmap"]

        self._scene_instance_map = {}
        self._shortest_path_maps = {}

        self._running_max_costs = [0.0] * num_envs

    def generate(
        self,
        ep_metadata,
        actions,
        observations,
        rewards,
        dones,
        infos,
        envs,
        skipped_last,
    ):
        for i, done in enumerate(dones):
            if done:
                # Reset the running max costs for this environment:
                self._running_max_costs[i] = 0.0

        goal_costmaps = []
        gt_costmaps = []

        for i, (ep, obs) in enumerate(zip(ep_metadata, observations)):
            goal_costmap, gt_costmap = self._get_costmaps(ep, obs, envs, i)
            goal_costmaps.append(goal_costmap)
            gt_costmaps.append(gt_costmap)

        if self._transforms:
            # TODO: Use nearest interpolation?
            goal_costmaps = self._apply_transforms(goal_costmaps)
            gt_costmaps = self._apply_transforms(gt_costmaps)

        return {"goal_costmap": goal_costmaps, "gt_costmap": gt_costmaps}

    def _get_costmaps(self, episode, obs, envs, env_idx):
        scene = episode.scene_id.split("/")[-1]
        goal = episode.object_category

        if scene not in self._scene_instance_map:
            self._scene_instance_map[scene] = build_semantic_instance_to_id_map(
                envs, env_idx
            )

        instance_to_label_pos = self._scene_instance_map[scene]
        sem_img_height, sem_img_width = obs["semantic"].shape[:2]

        goal_mask_img = np.zeros((sem_img_height, sem_img_width, 1), dtype=bool)
        for instance_id, (label, dist) in instance_to_label_pos.items():
            if label == goal:
                mask = obs["semantic"] == instance_id
                goal_mask_img[mask] = True

        if (scene, goal) not in self._shortest_path_maps:
            self._shortest_path_maps[(scene, goal)] = self._get_shortest_path_map(
                instance_to_label_pos, goal, envs, env_idx
            )

        gt_costmap = self._get_geodesic_distance_costmap(
            obs["semantic"], self._shortest_path_maps[(scene, goal)], env_idx
        )

        return goal_mask_img, gt_costmap

    def _get_shortest_path_map(
        self, instance_to_label_pos, goal, envs, env_idx, use_log_costs=True
    ):
        goal_positions = [
            pos for label, pos in instance_to_label_pos.values() if goal in label
        ]
        if not goal_positions:
            with open("missing_goal.txt", "a") as f:
                f.write(
                    f"No goal positions found for {goal} in scene {envs.current_episodes()[env_idx].scene_id}\n"
                )
                f.write(
                    f"Labels: {list(label for label, cost in instance_to_label_pos.values())}"
                )
            return {instance: 1.0 for instance in instance_to_label_pos}

        instance_to_cost = {
            instance_id: (
                envs.call_sim_at(
                    env_idx,
                    "geodesic_distance",
                    {"position_a": pos, "position_b": goal_positions},
                )
                if goal not in label
                else 0.0
            )
            for instance_id, (label, pos) in instance_to_label_pos.items()
        }

        if use_log_costs:
            # Use log costs to avoid large values:
            instance_to_cost = {
                instance_id: np.log(cost + 1) if np.isfinite(cost) else np.inf
                for instance_id, cost in instance_to_cost.items()
            }

        return instance_to_cost

    def _get_geodesic_distance_costmap(self, semantic_img, instance_to_cost, env_idx):
        normalised_instance_to_cost = self._get_normalised_costs(
            semantic_img, instance_to_cost, env_idx
        )

        costmap_img = np.ones_like(semantic_img, dtype=np.float32)
        # Map each instance_id in the semantic image to its cost (looping is faster
        # than the vectorised remapping):
        for k in np.unique(semantic_img):
            costmap_img[semantic_img == k] = normalised_instance_to_cost.get(k, 1.0)

        return (costmap_img * 255).astype(np.uint8)

    def _get_normalised_costs(
        self, semantic_img, instance_to_cost, env_idx, max_valid_cost=0.8
    ):
        current_instances = np.unique(semantic_img)
        current_costs = [
            instance_to_cost.get(instance, np.inf) for instance in current_instances
        ]

        if not np.isfinite(current_costs).any():
            # If all costs are invalid, return a map with all costs set to 1.0:
            return {instance: 1.0 for instance in current_instances}

        max_frame_cost = max(cost for cost in current_costs if np.isfinite(cost))

        # Update the maximum cost encountered so far for this episode. This is to match
        # normalisation of the cost predictor:
        self._running_max_costs[env_idx] = max(
            self._running_max_costs[env_idx], max_frame_cost
        )

        # Normalise the costs to [0, max_valid_cost], with 1.0 for invalid costs:
        max_cost = self._running_max_costs[env_idx]

        instance_to_cost = {
            instance_id: (
                cost / max_cost * max_valid_cost if np.isfinite(cost) else 1.0
            )
            for instance_id, cost in instance_to_cost.items()
        }

        return instance_to_cost


class GroundTruthPerceptionGraphGenerator:

    def __init__(
        self,
        num_envs,
        max_descriptor_update_image_frac,
        use_remote_mappers=False,
        **kwargs,
    ):
        self.data_names = ["gt_perception_graph"]
        self._use_remote_mappers = use_remote_mappers

        descriptor_generator = ClipDescriptorGenerator()
        self._make_mapper_fn = lambda: SimIncrementalMapper(
            max_descriptor_update_image_frac=max_descriptor_update_image_frac,
            descriptor_generator=descriptor_generator,
        )

        self._mappers = [None for _ in range(num_envs)]
        self._graphs = [None for _ in range(num_envs)]

        path = f"/storage/dc/sg/sg_habitat:{os.environ.get('PYTHONPATH', '')}"

        if self._use_remote_mappers:
            import ray

            ray.init(
                logging_level=logging.WARNING,
                runtime_env={"env_vars": {"PYTHONPATH": path}},
                # dashboard_host="0.0.0.0",
            )

            num_clip_workers = 2
            # RemoteDescriptorGenerator = ray.remote(num_cpus=0.1, num_gpus=0.1)(
            #     ClipDescriptorGenerator
            # )
            RemoteDescriptorGenerator = ray.serve.deployment(
                num_replicas=num_clip_workers,
                max_ongoing_requests=100,
                ray_actor_options={"num_cpus": 0.1, "num_gpus": 0.1},
            )(ClipDescriptorGenerator)

            descriptor_generator = RemoteDescriptorGenerator.bind()
            handle = ray.serve.run(descriptor_generator)

            # handle = ray.serve.get_app_handle()

            # descriptor_generator_worker = ray.util.ActorPool(
            #     [RemoteDescriptorGenerator.remote() for _ in range(num_clip_workers)]
            # )
            # descriptor_actors = [
            #     RemoteDescriptorGenerator.remote() for _ in range(num_clip_workers)
            # ]
            self._mappers = [
                RemoteSimIncrementalMapper.remote(
                    descriptor_generator_worker=[handle],
                    # descriptor_generator_worker=[
                    #     descriptor_actors[i % num_clip_workers]
                    # ],
                    # descriptor_generator_worker=descriptor_actors,
                    max_descriptor_update_image_frac=max_descriptor_update_image_frac,
                )
                for i in range(num_envs)
            ]

        self._env_idx_to_scene = {}
        self._scene_object_centers = {}
        self._scene_instance_id_to_label_dist_cache = {}

        # self._label_remap_func = load_object_label_remapping_func()

    def generate(
        self,
        ep_metadata,
        actions,
        observations,
        rewards,
        dones,
        infos,
        envs,
        skipped_last,
    ):
        t0_gen = time.perf_counter()

        # t0_scene_update = time.perf_counter()
        # Update scene instances:
        for i in range(len(observations)):
            scene = ep_metadata[i].scene_id.split("/")[-1]
            self._env_idx_to_scene[i] = scene
            if scene not in self._scene_object_centers:
                self._scene_object_centers[scene] = envs.call_at(
                    i, "get_object_centers"
                )
                self._scene_instance_id_to_label_dist_cache[scene] = (
                    build_semantic_instance_to_id_map(envs, i)
                )
        # print(f"\tScene Update Time: {time.perf_counter() - t0_scene_update:.2f}s")

        for i, done in enumerate(dones):
            if done:
                # Done is true on the first step of a new episode.
                if self._use_remote_mappers:
                    scene = self._env_idx_to_scene[i]
                    object_centers = self._scene_object_centers[scene]
                    self._mappers[i].reset.remote(object_centers)
                    self._graphs[i] = nx.Graph()
                else:
                    self._mappers[i] = self._make_mapper_fn()
                    self._graphs[i] = nx.Graph()

        t0_submit = time.perf_counter()
        # Update graphs
        graph_futures = []

        for i, (graph, mapper, obs) in enumerate(
            zip(self._graphs, self._mappers, observations)
        ):
            graph_future = self._get_updated_graph(
                graph,
                mapper,
                obs["rgb"],
                obs["semantic"],
                obs["depth"],
                self._scene_object_centers[scene],
            )

            graph_futures.append(graph_future)
        print(f"\tSubmission Time: {time.perf_counter() - t0_submit:.2f}s")

        t0_get_graphs = time.perf_counter()
        if self._use_remote_mappers:
            import ray

            self._graphs = [ray.get(f) for f in graph_futures]
        else:
            self._graphs = graph_futures
        print(f"\tRetrieval Time: {time.perf_counter() - t0_get_graphs:.2f}s")

        t0_labeling = time.perf_counter()
        # for graph in self._graphs:
        #     self._add_instance_labels(
        #         graph, self._scene_instance_id_to_label_dist_cache[scene]
        #     )
        print(f"\tLabeling Time: {time.perf_counter() - t0_labeling:.2f}s")
        print(f"Generation Time: {time.perf_counter() - t0_gen:.2f}s")
        return {"gt_perception_graph": self._graphs}

    def get_final_data(self, env_idx):
        graph = self._mappers[env_idx].graph.remote()
        graph = ray.get(graph)
        scene = self._env_idx_to_scene[env_idx]
        self._add_instance_labels(
            graph, self._scene_instance_id_to_label_dist_cache[scene]
        )
        return {"gt_perception_graph": graph}

    def _get_updated_graph(
        self,
        graph,
        mapper,
        rgb,
        semantic,
        depth,
        object_centers,
    ):
        semantic = semantic[..., 0]

        if self._use_remote_mappers:
            return mapper.update.remote(rgb, semantic, None)
            # return mapper.update.remote(graph, object_centers, rgb, semantic, None)
        else:
            return mapper.update(graph, object_centers, rgb, semantic, None)

    def _add_instance_labels(self, graph, instance_id_to_label_dist):
        new_nodes = [
            (n, attr) for n, attr in graph.nodes(data=True) if "label" not in attr
        ]

        labels = [
            instance_id_to_label_dist.get(attr["instance_id"], ["unknown"])[0]
            for n, attr in new_nodes
        ]

        # TODO: Add raw labels?
        for (n, attr), label in zip(new_nodes, labels):
            attr["label"] = label
            if label in OBJECTNAV_GOALS:
                attr["goal_label"] = label


class PredictedCostmapImageGenerator:

    def __init__(
        self,
        num_envs,
        model_path,
        edge_weight_name,
        resize_and_crop_to=None,
        costed_segment_max=0.8,
        max_seq_len=300,
        **kwargs,
    ):
        self._edge_weight_name = edge_weight_name
        self._cost_scale = costed_segment_max
        self._max_seq_len = max_seq_len

        # TODO: Support different graph generators
        self._graph_generator = GroundTruthPerceptionGraphGenerator(num_envs, **kwargs)
        self._descriptor_generator = ClipDescriptorGenerator()
        self._cost_predictor, train_config = load_checkpoint(model_path)
        self._cost_predictor.eval()

        self._goal_attributes = train_config.dataset.goal_attributes
        self._node_attributes = train_config.dataset.node_attributes

        self.data_names = ["predicted_costmap"]

        # TODO: Create cachedict class?
        self._goal_descriptor_cache = {}
        self._visual_descriptor_cache = [{} for _ in range(num_envs)]

        self._transforms = []
        if resize_and_crop_to:
            self._transforms.append(torchvision.transforms.Resize(resize_and_crop_to))
            self._transforms.append(
                torchvision.transforms.CenterCrop(resize_and_crop_to)
            )

    def generate(
        self,
        ep_metadata,
        actions,
        observations,
        rewards,
        dones,
        infos,
        envs,
        skipped_last,
        return_tensors=False,
    ):
        for i, done in enumerate(dones):
            if done:
                # Clear the visual descriptor cache for this environment:
                self._visual_descriptor_cache[i] = {}

        graph_data = self._graph_generator.generate(
            ep_metadata,
            actions,
            observations,
            rewards,
            dones,
            infos,
            envs,
            skipped_last,
        )
        graphs = graph_data["gt_perception_graph"]

        costmaps = []

        for i in range(len(graphs)):
            goal = ep_metadata[i].object_category
            mapper = self._graph_generator._mappers[i]
            costmap = self._get_costmap(mapper, graphs[i], goal)

            if self._transforms:
                costmap = self._apply_transforms(costmap)

            costmaps.append(costmap)

        return {"predicted_costmap": costmaps}

    def _get_costmap(self, mapper, graph, goal):
        if goal not in self._goal_descriptor_cache:
            self._goal_descriptor_cache[goal] = (
                self._descriptor_generator.get_text_descriptors([goal])[0]
            )

        goal_descriptor = torch.tensor(self._goal_descriptor_cache[goal]).unsqueeze(0)

        costs = predict_path_lengths(
            self._cost_predictor,
            graph,
            goal_descriptor,
            ["visual_descriptor"],
            edge_weight_name=self._edge_weight_name,
            max_seq_len=None,
            # max_seq_len=self._max_seq_len,
        )

        # TODO: Need a proper inference function that uses the model's config
        # to determine how to process costs:
        # Predictions are logp1, reverse:
        # costs = np.exp(costs) - 1.0

        # Ordering of graph.nodes should be consistent with the cost sequence:
        node_id_to_cost = {n: costs[n] for n in graph.nodes()}

        costmap = mapper.get_latest_frame_costmap(node_id_to_cost, self._cost_scale)
        costmap = torch.tensor(costmap)

        return costmap

    def _apply_transforms(self, costmap):
        costmap = torch.as_tensor(costmap).permute(2, 0, 1)
        for transform in self._transforms:
            costmap = transform(costmap)
        costmap = costmap.permute(1, 2, 0)
        return costmap


class CostmapVisualisationGenerator:

    def __init__(
        self,
        num_envs,
        model_path,
        edge_weight_name,
        costed_segment_max=0.8,
        max_seq_len=300,
        resize_and_crop_to=None,
        use_loaded_gt_graphs=True,
        **kwargs,
    ):
        self.data_names = ["costmap_visualisation"]

        self._costmap_generator = PredictedCostmapImageGenerator(
            num_envs,
            model_path,
            edge_weight_name,
            resize_and_crop_to=resize_and_crop_to,
            costed_segment_max=costed_segment_max,
            max_seq_len=max_seq_len,
            **kwargs,
        )

        if use_loaded_gt_graphs:
            self._gt_costmap_generator = LoadedGraphCostmapGenerator(
                edge_weight_name=edge_weight_name, **kwargs
            )
            self._gt_graph_name = "loaded_graph_costmap"
        else:
            self._gt_costmap_generator = GroundTruthCostmapImageGenerator(num_envs)
            self._gt_graph_name = "gt_costmap"

        self._transforms = []
        if resize_and_crop_to:
            self._transforms.append(torchvision.transforms.Resize(resize_and_crop_to))
            self._transforms.append(
                torchvision.transforms.CenterCrop(resize_and_crop_to)
            )

    def generate(self, ep_metadata, actions, observations, *args):
        costmap_data = self._costmap_generator.generate(
            ep_metadata, actions, observations, *args
        )
        costmaps = costmap_data["predicted_costmap"]

        gt_costmap_data = self._gt_costmap_generator.generate(
            ep_metadata, actions, observations, *args
        )
        gt_costmaps = gt_costmap_data[self._gt_graph_name]

        if self._transforms:
            costmaps = self._apply_transforms(costmaps)
            gt_costmaps = self._apply_transforms(gt_costmaps)
            rgb = self._apply_transforms([o["rgb"] for o in observations])

        costmap_deltas = [
            np.abs(cm.astype(float) - gt_cm.astype(float))
            for cm, gt_cm in zip(costmaps, gt_costmaps)
        ]

        def process_costmap(cm):
            cm = cm.squeeze(-1).astype(np.uint8)
            cm = cv2.applyColorMap(cm, cv2.COLORMAP_SUMMER)
            # cm = cv2.applyColorMap(cm, cv2.COLORMAP_WINTER)
            return cm

        costmaps = [process_costmap(cm) for cm in costmaps]
        gt_costmaps = [process_costmap(cm) for cm in gt_costmaps]
        costmap_deltas = [process_costmap(cm) for cm in costmap_deltas]

        return {
            "rgb": rgb,
            "costmap_delta": costmap_deltas,
            "costmap": costmaps,
            "gt_costmap": gt_costmaps,
        }

    def _apply_transforms(self, images):
        images = [torch.as_tensor(img).permute(2, 0, 1) for img in images]
        for transform in self._transforms:
            images = [transform(img) for img in images]
        images = [img.permute(1, 2, 0).numpy() for img in images]
        return images


class LoadedGraphCostmapGenerator:

    def __init__(
        self,
        graph_base_path,
        graph_name,
        edge_weight_name,
        label_attribute,
        use_log_costs=True,
        **kwargs,
    ):
        self.data_names = ["loaded_graph_costmap"]

        self._graph_base_path = graph_base_path
        self._graph_name = graph_name
        self._edge_weight_name = edge_weight_name
        self._label_attribute = label_attribute
        self._use_log_costs = use_log_costs

        self._graphs = {}
        self._instance_id_to_cost_dicts = {}
        self._running_max_costs = {}

    def generate(
        self,
        ep_metadata,
        actions,
        observations,
        rewards,
        dones,
        infos,
        envs,
        skipped_last,
    ):
        for i, done in enumerate(dones):
            if done:
                ep = ep_metadata[i]
                scene_id = ep.scene_id.split("/")[-2]
                ep_number = ep.episode_id
                goal = ep.object_category

                ep_id = f"{scene_id}_{ep_number}_{goal}"
                graph_file = f"{self._graph_base_path}/{ep_id}/{self._graph_name}"
                print(f"Loading graph from {graph_file}")
                with gzip.open(graph_file, "rb") as graph_file:
                    self._graphs[i] = pickle.load(graph_file)

                self._instance_id_to_cost_dicts[i] = self._get_instance_id_to_cost_dict(
                    self._graphs[i], goal
                )
                self._running_max_costs[i] = 0.0

        costmaps = []
        for i, obs in enumerate(observations):
            instance_id_to_cost = self._instance_id_to_cost_dicts[i]

            instance_id_to_cost, self._running_max_costs[i] = normalise_costs(
                obs["semantic"],
                instance_id_to_cost,
                self._running_max_costs[i],
            )

            costmap = build_costmap(
                obs["semantic"], instance_id_to_cost, max_valid_cost=0.8
            )
            costmaps.append(costmap)

        return {"loaded_graph_costmap": costmaps}

    def _get_instance_id_to_cost_dict(self, graph, goal):
        cost_matrix = get_cost_matrix(graph, weight=self._edge_weight_name)

        objectnav_cost_matrix = get_objectnav_cost_matrix(
            graph, cost_matrix, [goal], self._label_attribute
        )

        if self._use_log_costs:
            objectnav_cost_matrix = np.log(objectnav_cost_matrix + 1.0)

        return {
            d["instance_id"]: objectnav_cost_matrix[0, i]
            for i, d in graph.nodes(data=True)
        }


def build_costmap(semantic_img, instance_id_to_cost, max_valid_cost=0.8):
    costmap = -np.ones_like(semantic_img, dtype=np.float32)

    for k in np.unique(semantic_img):
        costmap[semantic_img == k] = instance_id_to_cost.get(k, -1.0)

    costmap /= costmap.max()
    costmap *= max_valid_cost

    # Set un-costed segments to 1.0
    costmap[costmap < 0] = 1.0
    return (costmap * 255).astype(np.uint8)


def normalise_costs(semantic_img, instance_to_cost, max_cost_so_far):
    max_current_frame_cost = max(
        instance_to_cost.get(k, -1.0) for k in np.unique(semantic_img)
    )
    max_cost_so_far = max(max_cost_so_far, max_current_frame_cost)

    instance_to_cost = {
        instance_id: cost / max_cost_so_far
        for instance_id, cost in instance_to_cost.items()
    }

    return instance_to_cost, max_cost_so_far


# def _get_normalised_costs(
#     self, semantic_img, instance_to_cost, env_idx, max_valid_cost=0.8
# ):
#     current_instances = np.unique(semantic_img)
#     current_costs = [
#         instance_to_cost.get(instance, np.inf) for instance in current_instances
#     ]

#     if not np.isfinite(current_costs).any():
#         # If all costs are invalid, return a map with all costs set to 1.0:
#         return {instance: 1.0 for instance in current_instances}

#     max_frame_cost = max(cost for cost in current_costs if np.isfinite(cost))

#     # Update the maximum cost encountered so far for this episode. This is to match
#     # normalisation of the cost predictor:
#     self._running_max_costs[env_idx] = max(
#         self._running_max_costs[env_idx], max_frame_cost
#     )

#     # Normalise the costs to [0, max_valid_cost], with 1.0 for invalid costs:
#     max_cost = self._running_max_costs[env_idx]

#     instance_to_cost = {
#         instance_id: (cost / max_cost * max_valid_cost if np.isfinite(cost) else 1.0)
#         for instance_id, cost in instance_to_cost.items()
#     }

#     return instance_to_cost


class ClipGenerator:

    def __init__(
        self,
        batch_size=64,
        device="cuda",
        model_path=None,
        use_float16=True,
        **kwargs,
    ):
        self._batch_size = batch_size
        self._device = device
        self._dtype = torch.float16 if use_float16 else torch.float32

        model_path = model_path or "openai/clip-vit-base-patch32"
        self._model = CLIPVisionModel.from_pretrained(model_path)
        self._processor = AutoProcessor.from_pretrained(model_path)

        self._model.to("cuda")

        self.data_names = [
            "clip_embedding",
            "last_two_hidden_layers",
        ]

    @torch.no_grad()
    def generate(
        self,
        ep_metadata,
        actions,
        observations,
        rewards,
        dones,
        infos,
        envs,
        skipped_last,
        return_tensors=False,
    ):
        images = [Image.fromarray(o["rgb"]) for o in observations]

        clip_embeddings = []
        last_two_hidden_layers = []

        for batch in batched(images, self._batch_size):
            inputs = self._processor(images=batch, return_tensors="pt", padding=True)
            inputs.to(self._device)
            outputs = self._model(**inputs, output_hidden_states=True)

            # Add each batch element as a separate tensor:
            clip_embeddings.extend(outputs[1])
            last_two_hidden_layers.extend(torch.stack(outputs[2][-2:], dim=1))

        if return_tensors:
            return (
                torch.stack(clip_embeddings).to(self._dtype),
                torch.stack(last_two_hidden_layers).to(self._dtype),
            )

        to_numpy = lambda t: t.detach().cpu().to(self._dtype).numpy()
        data = clip_embeddings, last_two_hidden_layers
        return {k: list(map(to_numpy, v)) for k, v in zip(self.data_names, data)}


class Vc1Generator:

    def __init__(
        self,
        batch_size=64,
        device="cuda",
        use_float16=True,
        model_path=None,
        **kwargs,
    ):
        self._batch_size = batch_size
        self._device = device
        self._dtype = torch.float16 if use_float16 else torch.float32

        model_path = model_path or model_utils.VC1_LARGE_NAME
        self._model, _, self._model_transforms, _ = model_utils.load_model(model_path)

        def dual_handle_outcome(self, x):
            x = self.norm(x)
            cls_tokens = x[:, 0]
            embeddings = vit.reshape_embedding(x[:, 1:])
            return cls_tokens, embeddings

        # Monkey patch the model to return CLS and embeddings:
        self._model.handle_outcome = types.MethodType(dual_handle_outcome, self._model)
        self._model.to(self._device)

        self._img_transform = torchvision.transforms.ToTensor()

        self.data_names = [
            "cls",
            "last_hidden_layer",
            "last_hidden_layer_pooled",
        ]

    @torch.no_grad()
    def generate(
        self,
        ep_metadata,
        actions,
        observations,
        rewards,
        dones,
        infos,
        envs,
        skipped_last,
        return_tensors=False,
    ):
        images = [Image.fromarray(o["rgb"]) for o in observations]

        images = [self._img_transform(img) for img in images]
        images = torch.stack(images).to(self._device)

        cls_tokens = []
        embeddings = []
        pooled_embeddings = []

        pool_fn = torch.nn.AvgPool2d(4, padding=1, count_include_pad=False)

        for batch in batched(images, self._batch_size):
            # TODO: Why does eai-vc readme say "The img loaded should be Bx3x250x250"?
            # It appears to be resized by the first transform anyway.
            # Output will be of size Bx3x224x224
            batch = self._model_transforms(batch)
            cls, last_hidden_layer = self._model(batch)

            # Add sequence dim to cls to match token sequences:
            cls_tokens.append(cls.unsqueeze(1).unsqueeze(1))

            reshape = lambda t: einops.rearrange(
                t, "batch tok seq1 seq2 -> batch seq1 seq2 tok"
            )
            embeddings.append(reshape(last_hidden_layer))
            pooled_embeddings.append(reshape(pool_fn(last_hidden_layer)))

        if return_tensors:
            return (
                torch.cat(cls_tokens).to(self._dtype),
                torch.cat(embeddings).to(self._dtype),
                torch.cat(pooled_embeddings).to(self._dtype),
            )

        to_numpy = lambda t: t.detach().cpu().to(self._dtype).numpy()
        data = cls_tokens, embeddings, pooled_embeddings
        return {k: map(to_numpy, v) for k, v in zip(self.data_names, data)}


class CogVlmGenerator:

    def __init__(
        self,
        batch_size=2,
        device="cuda",
        use_float16=True,
        model_path=None,
        prompt=None,
        num_hidden_layers=2,
        visual_pooling_kernel_size=12,
        **kwargs,
    ):
        self._batch_size = batch_size
        self._device = device
        self._dtype = torch.float16 if use_float16 else torch.float32
        self._prompt = prompt

        # Load the model:
        response = requests.post(
            "http://localhost:5000/load_model",
            json={
                "model_path": model_path,
                "device": device,
                "output_hidden_states": True,
                "output_last_num_hidden_layers": num_hidden_layers,
                "visual_pooling_kernel_size": visual_pooling_kernel_size,
            },
        )
        print(f"Load model response: {response.json()}")

        self.data_names = [
            "last_two_hidden_layers_pooled",
            "masks",
        ]

    @staticmethod
    def make_files(images, prompt):
        files = []
        for i, image in enumerate(images):
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            buffer.seek(0)
            files.append(("images", (f"image_{i}.png", buffer, "image/png")))
            files.append(("prompts", (None, prompt)))
        return files

    @staticmethod
    def base64_to_numpy(base64_array):
        with io.BytesIO(base64_array.encode("latin-1")) as buffer:
            # with io.BytesIO(base64.decodebytes(base64_array).encode("ascii")) as buffer:
            return np.load(buffer)

    @torch.no_grad()
    def generate(
        self,
        ep_metadata,
        actions,
        observations,
        rewards,
        dones,
        infos,
        envs,
        skipped_last,
        return_tensors=False,
    ):
        images = [Image.fromarray(o["rgb"]) for o in observations]

        last_two_hidden_layers_pooled = []
        masks = []

        for batch in batched(images, self._batch_size):
            files = self.make_files(batch, self._prompt)
            response = requests.post("http://localhost:5000/query_batch", files=files)
            response_json = response.json()

            hidden_states = self.base64_to_numpy(response_json["hidden_states"])
            hidden_state_masks = self.base64_to_numpy(
                response_json["hidden_state_masks"]
            )

            last_two_hidden_layers_pooled.append(hidden_states)
            masks.append(hidden_state_masks)

        if return_tensors:
            return (
                torch.cat(last_two_hidden_layers_pooled).to(self._dtype),
                torch.cat(masks).to(self._dtype),
            )

        # TODO: Pad to max generation length
        last_two_hidden_layers_pooled = np.concatenate(last_two_hidden_layers_pooled)
        masks = np.concatenate(masks)

        to_numpy = lambda t: t.detach().cpu().to(self._dtype).numpy()
        data = last_two_hidden_layers_pooled, masks
        return {k: map(to_numpy, v) for k, v in zip(self.data_names, data)}


class NonVisualObservationsGenerator:

    def __init__(self, **kwargs):
        self._obs_keys = [
            "objectgoal",
            "compass",
            "gps",
            "next_actions",
            "inflection_weight",
        ]
        self.data_names = [
            *self._obs_keys,
            "reward",
            "done",
        ]

    def generate(
        self,
        ep_metadata,
        actions,
        observations,
        rewards,
        dones,
        infos,
        envs,
        skipped_last,
    ):
        data = {k: [obs[k] for obs in observations] for k in self._obs_keys}
        data["reward"] = rewards
        data["done"] = dones
        return data


class HabitatSimActions(Enum):
    STOP = 0
    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    LOOK_UP = 4
    LOOK_DOWN = 5


class AgentStateGenerator:

    def __init__(self, remove_non_movement_actions=True, **kwargs):
        """
        remove_non_movement_actions: If True, remove non-movement actions from the
        dataset (except stop action) by returning None.
        """
        self.data_names = ["agent_state", "position", "yaw"]

        self.remove_non_movement_actions = remove_non_movement_actions
        # Track the previous output positions. These are different to the sim agent
        # states, as rotations are only converted to small movements for the output
        # positions:
        self.prev_output_positions = None

    def generate(
        self,
        ep_metadata,
        prev_actions,
        observations,
        rewards,
        dones,
        infos,
        envs,
        skipped_last,
    ):
        num_envs = len(observations)
        agent_states = [envs.call_sim_at(i, "get_agent_state") for i in range(num_envs)]

        positions, yaws = zip(*[state_to_traj_sample(state) for state in agent_states])
        positions = [np.array(pos) for pos in positions]
        yaws = list(yaws)

        if self.prev_output_positions is None:
            self.prev_output_positions = positions

        prev_actions = [
            HabitatSimActions(a) if a is not None else None for a in prev_actions
        ]

        for i in range(num_envs):
            positions[i] = convert_rotation_to_small_movement(
                dones[i],
                prev_actions[i],
                self.prev_output_positions[i],
                positions[i],
                yaws[i],
            )

        # self.prev_output_positions = list(positions)
        self.prev_output_positions = [
            positions[i] if not skipped_last[i] else self.prev_output_positions[i]
            for i in range(num_envs)
        ]

        return {
            "agent_state": agent_states,
            "position": positions,
            "yaw": yaws,
        }


def batched(iterable, batch_size):
    n = len(iterable)
    return (iterable[i : min(i + batch_size, n)] for i in range(0, n, batch_size))


def data_dict_to_batched_tensors(data_dict):
    return {k: torch.stack(v) for k, v in data_dict.items()}


def convert_rotation_to_small_movement(
    done, prev_action, prev_output_pos, pos, yaw, movement_size=0.025
):
    # The "output_pos" is different to the true position in the sim, because we don't
    # actually apply the small movements in the sim. Need to move relative to the output
    # pos to "point" in the right direction.
    # TODO: We could apply the small movements in the sim, but then we would have to
    # assess the impact of collisions
    is_rotation = prev_action in [
        HabitatSimActions.TURN_LEFT,
        HabitatSimActions.TURN_RIGHT,
    ]

    # Done is true on the first step of an episode. In this case, prev_action is for
    # the previous episode, so no need to apply rotation in this case:
    if done or not is_rotation:
        return pos

    movement = movement_size * np.array([np.cos(yaw), np.sin(yaw)])
    return prev_output_pos + movement


def build_semantic_instance_to_id_map(envs, env_idx):
    instance_id_to_label_dist = envs.call_at(
        env_idx, "get_semantic_instance_id_to_name_map"
    )

    # The target object categories sometimes have additional text (e.g. "armchair"),
    # or multiple names for the same thing (e.g. couch, sofa) so we remap them to a
    # canonical label:
    # TODO: How was this done for the original ObjectNav dataset? There are some
    # false positives here (e.g. "tv cabinet"):
    remap_label_func = load_object_label_remapping_func()

    for instance_id, (orig_label, cost) in instance_id_to_label_dist.items():
        canonical_label = remap_label_func(orig_label)
        instance_id_to_label_dist[instance_id] = (canonical_label, cost)

    return instance_id_to_label_dist


def load_object_label_remapping_func():
    # Poorly documented. Original mappings come from this file, but it's not in the
    # version of habitat-sim used by pirlnav. Copy it to the root of this project:
    # habitat-sim/data/hm3d_semantics/hm3dsem_category_mappings.tsv

    import pandas as pd

    df = pd.read_csv("hm3dsem_category_mappings.tsv", sep="\t")

    mapping = {
        raw: canonical for raw, canonical in zip(df["raw_category"], df["mpcat40"])
    }

    return lambda label: mapping[label.strip().lower()]


# def convert_rotation_to_small_movement(
#     action, current_pos, prev_pos, yaw_after, movement_size=0.05
# ):
#     action = HabitatSimActions(action)
#     is_rotation = action in [HabitatSimActions.TURN_LEFT, HabitatSimActions.TURN_RIGHT]

#     if not is_rotation:
#         return current_pos

#     movement = movement_size * np.array([np.cos(yaw_after), np.sin(yaw_after)])
#     return prev_pos + movement


# def pos_to_2d(pos):
#     """
#     Agent state coordinate system: x-right, y-up, z-backward

#     NoMaD expects: x-forward, y-right
#     """
#     # Convert -z to x, x to y:
#     return np.array([-pos[2], pos[0]])


# Quaternion utils from sg_habitat:
def quat_mn2np(q):
    return np.concatenate([np.array(q.vector), [q.scalar]])


def quat_hab_to_euler(q):
    # expects habitat q, returns euler angles in radians
    return Rotation.from_quat(quat_mn2np(quat_to_magnum(q))).as_euler("xyz")
    # return getEulerFromQuaternion(quat_mn2np(quat_to_magnum(q)))


def hs_quat_to_array(q):
    mn_mat = quat_to_magnum(q).to_matrix()
    return np.array(mn_mat)


from spatialmath import SE3
from spatialmath.base import trnorm


def SE3_from4x4(pose):
    # check is False, do see https://github.com/bdaiinstitute/spatialmath-python/issues/28
    # return SE3.Rt(pose[:3,:3], pose[:3,-1],check=False)
    # if pose is instance list of R (3x3),t(3), conver tto pose
    if isinstance(pose, list) and len(pose) == 2:
        pose4x4 = np.eye(4)
        R, t = pose
        pose4x4[:3, :3] = R
        pose4x4[:3, -1] = t.flatten()
        pose = pose4x4
    return SE3(trnorm(np.array(pose)))


def state_to_traj_sample(state):
    # states are T_wb (world to base), convert that to camera
    T_bc = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    )  # camera to base
    R = hs_quat_to_array(state.rotation)
    t = state.position

    pose = SE3_from4x4([R, t]) @ SE3(T_bc)
    yaw = np.arctan2(pose.R[0, 2], pose.R[2, 2])
    # yaws = np.array([np.arctan2(Ri[0,2], Ri[2,2]) for Ri in Rs])
    position = t[[2, 0]]
    return position, yaw


# def quat_to_yaw(quat):
#     """
#     Agent state coordinate system: x-right, y-up, z-backward

#     NoMaD expects: x-forward, y-right
#     """
#     hab_z_negative_yaw = quat_hab_to_euler(quat)[1]
#     # Flip across NoMaD x-axis:
#     return (np.pi - np.abs(hab_z_negative_yaw)) * -np.sign(hab_z_negative_yaw)
#     # R_init = np.array(quat_to_magnum(quat).to_matrix())
#     # return -np.arctan2(R_init[0,2], R_init[2,2])

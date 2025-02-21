from enum import Enum
import json
import os
import pickle
import time
import types
import einops
import numpy as np
import torch
import torchvision
import zarr
import tqdm
import requests
import io

from PIL import Image
from concurrent.futures import ProcessPoolExecutor

from habitat_sim.utils.common import quat_to_magnum
from pirlnav.utils.env_utils import (
    construct_envs,
    generate_dataset_split_json,
    get_episodes,
)
from habitat.core.environments import get_env_class
from habitat.core.vector_env import CURRENT_EPISODE_NAME

from vc_models.models.vit import model_utils, vit
from scipy.spatial.transform import Rotation
from transformers import CLIPVisionModel, AutoProcessor

# TODO: Change generator to return dict of tensors


def generate_episode_split_index(config):
    output_path = config["TASK_CONFIG"]["SUB_SPLIT_GENERATOR"]["INDEX_PATH"]
    stride = config["TASK_CONFIG"]["SUB_SPLIT_GENERATOR"]["STRIDE"]
    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # generate_dataset_split_json(config, output_path, stride, object_classes=["toilet"])
    generate_dataset_split_json(config, output_path, stride)


def get_data_generators(config):
    data_generator_registry = {
        "non_visual": NonVisualObservationsGenerator,
        "raw_image": RawImageGenerator,
        "clip": ClipGenerator,
        "vc_1": Vc1Generator,
        "cogvlm2": CogVlmGenerator,
        "agent_state": AgentStateGenerator,
    }

    generator_config = config["TASK_CONFIG"]["REPRESENTATION_GENERATOR"]

    return [
        data_generator_registry[name](**args if args is not None else {})
        for name, args in generator_config["data_generators"].items()
    ]


def get_data_storage(config):
    storage_registry = {
        "zarr": ZarrDataStorage,
        "nomad": NomadDataStorage,
    }

    storage_config = config["TASK_CONFIG"]["REPRESENTATION_GENERATOR"]["data_storage"]
    storage_kwargs = storage_config.get("kwargs", {})

    output_path = storage_config["output_path"]
    return storage_registry[storage_config["name"]](output_path, **storage_kwargs)


class ZarrDataStorage:

    def __init__(self, output_path, batch_chunk_size=1000, **kwargs):
        self._output_path = output_path
        self._batch_chunk_size = batch_chunk_size

        self._zarr_file = None
        self._completed_eps = []

    def _init_zarr_file(self, data):
        self._zarr_file = zarr.open(self._output_path, mode="w")
        self._data_group = self._zarr_file.create_group("data")
        # In older versions of zarr, "meta*" names are reserved, so prefix with "_":
        self._meta_group = self._zarr_file.create_group("_meta")

        for key, data_array in data.items():
            chunks = [None] * len(data_array.shape)
            chunks[0] = self._batch_chunk_size
            self._data_group[key] = zarr.array(data_array, chunks=chunks)

    def save_episode(self, data, scene_id, episode_id, object_category):
        if self._zarr_file is None:
            # First episode starts at row 0:
            row = 0
            # Create the zarr file and groups and save the data:
            self._init_zarr_file(data)
        else:
            row = self._data_group[next(data.keys())].shape[0]
            # Append the data to the existing zarr file:
            for key, data_array in data.items():
                self._data_group[key].append(data_array)

        # Update the completed episode index:
        self._completed_eps.append(
            {
                "scene_id": scene_id,
                "episode_id": episode_id,
                "object_category": object_category,
                "row": row,
            }
        )

        # Save the episode index in case of early termination:
        ep_index_path = f"{self._output_zarr_path}/ep_index.json"
        json.dump(self._completed_eps, open(ep_index_path, "w"))

    def close(self):
        self._zarr_file.close()


class NomadDataStorage:

    def __init__(self, output_path, num_workers=20, **kwargs):
        self._output_path = output_path
        self._completed_eps = []

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

        os.makedirs(save_dir, exist_ok=True)

        # Save the data:
        f = self._pool.submit(
            self._write_traj_data,
            save_dir,
            np.array(data["rgb"]),
            np.array(data["agent_state"]),
            np.array(data["position"]),
            np.array(data["yaw"]),
        )
        self._futures.append(f)

        # Update traj_names.txt. Write entire file every time in case there happens
        # to already be a file with the same name:
        self._completed_eps.append(f"{ep_id}")
        ep_index_path = f"{self._output_path}/traj_names.txt"
        with open(ep_index_path, "w") as f:
            for ep in self._completed_eps:
                f.write(f"{ep}\n")

    @staticmethod
    def _write_traj_data(save_dir, rgb_data, agent_states, positions, yaws):
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

    def plot_traj(positions, yaws):
        import matplotlib.pyplot as plt

        plt.plot(positions[:, 0], positions[:, 1], marker=".")
        # Plot yaws as arrows:
        for i, (pos, yaw) in enumerate(zip(positions[:], yaws[:])):
            plt.arrow(
                pos[0],
                pos[1],
                0.1 * np.cos(yaw),
                0.1 * np.sin(yaw),
                width=0.01,
                color="red",
            )
        # Square axes:
        plt.gca().set_aspect("equal", adjustable="box")
        plt.show()

    def close(self):
        self._pool.shutdown(wait=True)


class RepresentationGenerator:

    def __init__(self, config):
        print(
            "WARNING: If using a different NUM_ENVIRONMENTS when generating different "
            "datasets, the order of the episodes will be different."
        )
        self._envs, self._remaining_episodes, self._remaining_ep_set = self._init_envs(
            config
        )
        self._num_envs = config["NUM_ENVIRONMENTS"]

        self._data_generators = get_data_generators(config)
        self._data_storage = get_data_storage(config)

    def _init_envs(self, config):
        config.defrost()
        config.TASK_CONFIG.TASK.SENSORS.extend(
            ["DEMONSTRATION_SENSOR", "INFLECTION_WEIGHT_SENSOR"]
        )
        config.freeze()

        sub_split_index_path = config["TASK_CONFIG"]["DATASET"]["SUB_SPLIT_INDEX_PATH"]
        with open(sub_split_index_path, "r") as f:
            sub_split_index = json.load(f)

        episode_ids = set(
            [
                (ep["scene_id"], ep["episode_id"], ep["object_category"])
                for ep in sub_split_index
            ]
        )

        envs = construct_envs(
            config,
            get_env_class(config.ENV_NAME),
            workers_ignore_signals=False,
            shuffle_scenes=False,
            episode_index=sub_split_index,
        )

        episodes = get_episodes(config, sub_split_index)

        return envs, episodes, episode_ids

    def generate(self):
        # TODO: With a stride of 10, one scene-goal pair is missing
        total_num_eps = sum(self._envs.count_episodes())
        print(f"Number of episodes: {total_num_eps}")

        # Track the data for each episode separately:
        rollout_data = [[] for _ in range(self._num_envs)]

        for i in range(self._num_envs):
            next_ep = self._remaining_episodes.pop()
            self._envs.call_at(i, "set_current_episode", {"episode": next_ep})

        actions = [None] * self._num_envs
        observations = self._envs.reset()
        rewards = [0.0] * self._num_envs
        # Done is true on the first step of each episode (non-standard convention):
        dones = [True] * self._num_envs

        # Track the episode info to deduplicate episodes from cycling environments:
        previous_episodes = self._envs.current_episodes()
        print(
            f"{previous_episodes[0].scene_id}, {previous_episodes[0].episode_id}, {previous_episodes[0].object_category}"
        )
        step_data = self._generate_step(actions, observations, rewards, dones, None)
        for ep, data in zip(rollout_data, step_data):
            ep.append(data)

        pbar = tqdm.tqdm(total=total_num_eps)
        while self._remaining_ep_set:
            # "next_actions" contains the actions from the BC dataset:
            actions = [o["next_actions"] for o in observations]

            outputs = self._envs.step(actions)
            observations, rewards, dones, infos = zip(*outputs)

            # If done, save the episode (the current obs is for the next ep):
            for i, ep in enumerate(rollout_data):
                ep_metadata = previous_episodes[i]

                if dones[i]:
                    t0 = time.time()
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
                        data = {k: np.stack([d[k] for d in ep]) for k in ep[0]}
                        self._data_storage.save_episode(data, *ep_key)
                        pbar.update(1)

                        if len(self._remaining_ep_set) > 0:
                            # Reset the env to the next episode:
                            next_ep = self._remaining_episodes.pop()
                            # Set the next episode manually, then reset (TODO: Is this
                            # slow?):
                            t_set = time.time()
                            self._envs.call_at(
                                i, "set_current_episode", {"episode": next_ep}
                            )
                            observations = list(observations)
                            observations[i] = self._envs.call_at(i, "reset")
                            previous_episodes[i] = next_ep
                            print(f"Set time: {time.time() - t_set}")
                    else:
                        print(f"Duplicate episode {ep_key} (skipping)")
                    ep.clear()
                    t1 = time.time()
                    # This env is now on a new ep, so update its metadata:
                    previous_episodes[i] = self._envs.call_at(i, CURRENT_EPISODE_NAME)
                    print(f"get ep time {time.time() - t1}")

                    print(f"Reset time: {time.time() - t0}")

            step_data = self._generate_step(
                actions, observations, rewards, dones, infos
            )
            for ep, data in zip(rollout_data, step_data):
                ep.append(data)

        pbar.close()
        self._data_storage.close()

    def _generate_step(self, actions, observations, rewards, dones, infos):
        data = {}
        for data_generator in self._data_generators:
            output = data_generator.generate(
                actions, observations, rewards, dones, infos, self._envs
            )
            data.update(output)
        ep_data = [
            {k: v[env_idx] for k, v in data.items()}
            for env_idx in range(self._num_envs)
        ]
        return ep_data


class RawImageGenerator:

    def __init__(self):
        self.data_names = ["rgb"]

    @torch.no_grad()
    def generate(self, actions, observations, rewards, dones, infos, envs):
        return {"rgb": [o["rgb"] for o in observations]}


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
        self, actions, observations, rewards, dones, infos, envs, return_tensors=False
    ):
        images = [Image.fromarray(o["rgb"]) for o in observations]

        clip_embeddings = []
        last_two_hidden_layers = []

        for batch in batched(images, self._batch_size):
            inputs = self._processor(images=batch, return_tensors="pt", padding=True)
            inputs.to(self._device)
            outputs = self._model(**inputs, output_hidden_states=True)

            clip_embeddings.append(outputs[1])
            last_two_hidden_layers.append(torch.stack(outputs[2][-2:], dim=1))

        if return_tensors:
            return (
                torch.cat(clip_embeddings).to(self._dtype),
                torch.cat(last_two_hidden_layers).to(self._dtype),
            )

        to_numpy = lambda t: t.detach().cpu().to(self._dtype).numpy()
        data = clip_embeddings, last_two_hidden_layers
        return {k: map(to_numpy, v) for k, v in zip(self.data_names, data)}


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
        self, actions, observations, rewards, dones, infos, envs, return_tensors=False
    ):
        # images = [o["rgb"].swapaxes(0, -1).astype(np.float32) for o in observations]
        # print([img.shape for img in images])
        images = [Image.fromarray(o["rgb"]) for o in observations]

        images = [self._img_transform(img) for img in images]
        images = torch.stack(images).to(self._device)
        # print(f"images shape: {images.shape}")

        # images_old = torch.stack([torch.tensor(o["rgb"]) for o in observations])
        # print(f"images_old shape 1: {images_old.shape}")
        # images_old = einops.rearrange(images_old, "b h w c -> b c h w")
        # images_old = images_old.to(self._device, dtype=torch.float32)
        # print(f"images_old shape 2: {images_old.shape}")

        # torch.save(images, "temp/images.pt")
        # torch.save(images_old, "temp/images_old.pt")

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
        self, actions, observations, rewards, dones, infos, envs, return_tensors=False
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

    def generate(self, actions, observations, rewards, dones, infos, envs):
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

    def __init__(self, **kwargs):
        self.data_names = ["agent_state", "position", "yaw"]

        self.prev_positions = None

    def generate(self, actions, observations, rewards, dones, infos, envs):
        num_envs = len(observations)
        agent_states = [envs.call_sim_at(i, "get_agent_state") for i in range(num_envs)]

        positions, yaws = zip(*[state_to_traj_sample(state) for state in agent_states])

        # positions = [pos_to_2d(state.position) for state in agent_states]
        # yaws = [quat_to_yaw(state.rotation) for state in agent_states]

        if self.prev_positions is not None:
            positions = [
                convert_rotation_to_small_movement(
                    action, current_pos, prev_pos, yaw_after
                )
                for action, current_pos, prev_pos, yaw_after in zip(
                    actions, positions, self.prev_positions, yaws
                )
            ]
        self.prev_positions = positions

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
    action, current_pos, prev_pos, yaw_after, movement_size=0.05
):
    action = HabitatSimActions(action)
    is_rotation = action in [HabitatSimActions.TURN_LEFT, HabitatSimActions.TURN_RIGHT]

    if not is_rotation:
        return current_pos

    movement = movement_size * np.array([np.cos(yaw_after), np.sin(yaw_after)])
    return prev_pos + movement


def pos_to_2d(pos):
    """
    Agent state coordinate system: x-right, y-up, z-backward

    NoMaD expects: x-forward, y-right
    """
    # Convert -z to x, x to y:
    return np.array([-pos[2], pos[0]])


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


def quat_to_yaw(quat):
    """
    Agent state coordinate system: x-right, y-up, z-backward

    NoMaD expects: x-forward, y-right
    """
    hab_z_negative_yaw = quat_hab_to_euler(quat)[1]
    # Flip across NoMaD x-axis:
    return (np.pi - np.abs(hab_z_negative_yaw)) * -np.sign(hab_z_negative_yaw)
    # R_init = np.array(quat_to_magnum(quat).to_matrix())
    # return -np.arctan2(R_init[0,2], R_init[2,2])

import json
import os
import numpy as np
import torch
import zarr
import tqdm

from PIL import Image

from pirlnav.utils.env_utils import construct_envs, generate_dataset_split_json
from habitat.core.environments import get_env_class

from transformers import CLIPVisionModel, AutoProcessor

# TODO: Change generator to return dict of tensors


def generate_episode_split_index(config):
    output_path = config["TASK_CONFIG"]["SUB_SPLIT_GENERATOR"]["INDEX_PATH"]
    stride = config["TASK_CONFIG"]["SUB_SPLIT_GENERATOR"]["STRIDE"]
    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    generate_dataset_split_json(config, output_path, stride)


class RepresentationGenerator:

    def __init__(self, config):
        print(
            "WARNING: If using a different NUM_ENVIRONMENTS when generating different "
            "datasets, the order of the episodes will be different."
        )
        self._init_envs(config)
        generator_config = config["TASK_CONFIG"]["REPRESENTATION_GENERATOR"]

        self._num_envs = config["NUM_ENVIRONMENTS"]
        self._output_zarr_path = generator_config["output_zarr_path"]
        self._ep_index_path = f"{self._output_zarr_path}/ep_index.json"
        self._batch_chunk_size = generator_config["batch_chunk_size"]

        self._zarr_file = None
        data_generator_name = generator_config["data_generator"]["name"]

        if data_generator_name == "non_visual":
            self._data_generator = NonVisualObservationsGenerator()
        elif data_generator_name == "clip":
            generator_kwargs = generator_config["data_generator"].get("clip_kwargs", {})
            self._data_generator = ClipGenerator(**generator_kwargs)

        self._c = 0

    def _init_envs(self, config):
        config.defrost()
        config.TASK_CONFIG.TASK.SENSORS.extend(
            ["DEMONSTRATION_SENSOR", "INFLECTION_WEIGHT_SENSOR"]
        )
        config.freeze()

        sub_split_index_path = config["TASK_CONFIG"]["DATASET"]["SUB_SPLIT_INDEX_PATH"]
        with open(sub_split_index_path, "r") as f:
            sub_split_index = json.load(f)

        self._remaining_ep_set = set(
            set([tuple(ep.values()) for ep in sub_split_index])
        )
        self._completed_eps = []

        self.envs = construct_envs(
            config,
            get_env_class(config.ENV_NAME),
            workers_ignore_signals=False,
            shuffle_scenes=False,
            episode_index=sub_split_index,
        )

    def _init_zarr_file(self, data_names, data):
        self._zarr_file = zarr.open(self._output_zarr_path, mode="w")
        self._data_group = self._zarr_file.create_group("data")
        # In older versions of zarr, "meta*" names are reserved, so prefix with "_":
        self._meta_group = self._zarr_file.create_group("_meta")

        for name, data_array in zip(data_names, data):
            chunks = [None] * len(data_array.shape)
            chunks[0] = self._batch_chunk_size
            self._data_group[name] = zarr.array(data_array, chunks=chunks)

    def _save_data(self, data, ep_id):
        # Check if this is a duplicate episode (i.e. an environment that cycled through
        # all its eps):
        if ep_id not in self._remaining_ep_set:
            print(f"Duplicate episode {ep_id} (skipping)")
            return
        print(f"Completed episode {ep_id}")
        self._remaining_ep_set.remove(ep_id)

        data_names = self._data_generator.data_names
        if self._zarr_file is None:
            # First episode starts at row 0:
            row = 0
            # Create the zarr file and groups and save the data:
            self._init_zarr_file(data_names, data)
        else:
            row = self._data_group[data_names[0]].shape[0]
            # Append the data to the existing zarr file:
            for name, data_array in zip(data_names, data):
                print(f"{name}: {data_array.shape}")
                self._data_group[name].append(data_array)
        # Update the completed episode index:
        self._completed_eps.append(
            {
                "scene_id": ep_id[0],
                "episode_id": ep_id[1],
                "object_category": ep_id[2],
                "row": row,
            }
        )
        # Save the episode index in case of early termination:
        json.dump(self._completed_eps, open(self._ep_index_path, "w"))

    def generate(self):
        # TODO: With a stride of 10, one scene-goal pair is missing
        total_num_eps = sum(self.envs.count_episodes())
        print(f"Number of episodes: {total_num_eps}")

        # Track the data for each episode separately:
        rollout_data = [[] for _ in range(self._num_envs)]

        observations = self.envs.reset()
        rewards = [0.0] * self._num_envs
        # Done is true on the first step of each episode (weird convention):
        dones = [True] * self._num_envs

        step_data = self._data_generator.generate(observations, rewards, dones, None)
        for ep, data in zip(rollout_data, step_data):
            ep.append(data)

        ep_count = 0
        while self._remaining_ep_set:
            # "next_actions" contains the actions from the BC dataset:
            actions = [o["next_actions"] for o in observations]
            previous_episodes = self.envs.current_episodes()
            outputs = self.envs.step(actions)
            observations, rewards, dones, infos = zip(*outputs)

            # Save the images:
            # images = [Image.fromarray(o["rgb"]) for o in observations]
            # for i, img in enumerate(images):
            #     current_ep = self.envs.current_episodes()[i]
            #     scene, ep = current_ep.scene_id, current_ep.episode_id
            #     scene = scene.replace("/", "_")
            #     os.makedirs(f"imgs/one_pc/scene_{scene}/ep_{ep}", exist_ok=True)
            #     print(f"SAVING imgs/one_pc/scene_{scene}/ep_{ep}/step_{self._c}_done_{dones[i]}.png")
            #     img.save(f"imgs/one_pc/scene_{scene}/ep_{ep}/step_{self._c}_done_{dones[i]}.png")
            # self._c += 1

            step_data = self._data_generator.generate(
                observations, rewards, dones, infos
            )

            # If done, save the episode (the current obs is for the next ep):
            for ep, ep_metadata, done in zip(rollout_data, previous_episodes, dones):
                if done:
                    # Each element of ep is a list of different data types. Concat each
                    # data type into a separate array:
                    data = [np.stack([d[i] for d in ep]) for i in range(len(ep[0]))]
                    ep_id = (
                        ep_metadata.scene_id,
                        ep_metadata.episode_id,
                        ep_metadata.object_category,
                    )
                    self._save_data(data, ep_id)
                    ep.clear()

            for ep, data in zip(rollout_data, step_data):
                ep.append(data)

            ep_count += sum(dones)
            if sum(dones) > 0:
                print(f"Episodes completed: {ep_count}")


class ClipGenerator:

    def __init__(
        self,
        batch_size=64,
        device="cuda",
        model_path=None,
        use_float16="float32",
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
    def generate(self, observations, rewards, dones, infos, return_tensors=False):
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
            return torch.cat(clip_embeddings), torch.cat(last_two_hidden_layers)

        clip_embeddings = (
            torch.cat(clip_embeddings).detach().cpu().to(self._dtype).numpy()
        )
        last_two_hidden_layers = (
            torch.cat(last_two_hidden_layers).detach().cpu().to(self._dtype).numpy()
        )
        # Return all data as a sequence of tuples for each environment:
        return zip(clip_embeddings, last_two_hidden_layers)


class NonVisualObservationsGenerator:

    def __init__(self):
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

    def generate(self, observations, rewards, dones, infos):
        obs_data = [[obs[key] for key in self._obs_keys] for obs in observations]
        return [
            [*obs, reward, done] for obs, reward, done in zip(obs_data, rewards, dones)
        ]


def batched(iterable, n):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]

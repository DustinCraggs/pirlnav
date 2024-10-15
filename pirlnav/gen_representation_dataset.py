import os
import numpy as np
import torch
import zarr
import tqdm

from PIL import Image

from habitat.utils.env_utils import construct_envs
from habitat.core.environments import get_env_class

from transformers import CLIPVisionModel, AutoProcessor

# TODO: Change generator to return dict of tensors


class RepresentationGenerator:

    def __init__(self, config):
        self._init_envs(config)
        generator_config = config["TASK_CONFIG"]["REPRESENTATION_GENERATOR"]

        self._num_envs = config["NUM_ENVIRONMENTS"]
        self._output_zarr_path = generator_config["output_zarr_path"]
        self._batch_chunk_size = generator_config["batch_chunk_size"]

        self._zarr_file = None
        data_generator_name = generator_config["data_generator_name"]
        if data_generator_name == "clip":
            self._data_generator = ClipGenerator()
        elif data_generator_name == "non_visual":
            self._data_generator = NonVisualObservationsGenerator()

        self._c = 0


    def _init_envs(self, config):
        config.defrost()
        config.TASK_CONFIG.TASK.SENSORS.extend(
            ["DEMONSTRATION_SENSOR", "INFLECTION_WEIGHT_SENSOR"]
        )
        config.freeze()

        self.envs = construct_envs(
            config,
            get_env_class(config.ENV_NAME),
            workers_ignore_signals=False,
            shuffle_scenes=False,
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

    def _save_data(self, data):
        data_names = self._data_generator.data_names
        if self._zarr_file is None:
            # Create the zarr file and groups and save the data:
            self._init_zarr_file(data_names, data)
        else:
            # Append the data to the existing zarr file:
            for name, data_array in zip(data_names, data):
                print(f"{name}: {data_array.shape}")
                self._data_group[name].append(data_array)

    def generate(self):
        # TODO: Discard repeated episodes for environments that cycle at the end
        # TODO: Ensure that every goal is represented with a stride of 10 (i.e., are
        # there scene-goal pairs that are have <10 examples in the full dataset?)
        total_num_eps = sum(self.envs.count_episodes())
        print(f"Number of episodes: {total_num_eps}")

        # Track the data for each episode separately:
        rollout_data = [[] for _ in range(self._num_envs)]

        observations = self.envs.reset()
        rewards = [0.0] * self._num_envs
        dones = [False] * self._num_envs

        step_data = self._data_generator.generate(observations, rewards, dones, None)
        for ep, data in zip(rollout_data, step_data):
            ep.append(data)

        ep_count = 0
        while ep_count < total_num_eps:
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
            #     os.makedirs(f"img/one_pc/scene_{scene}/ep_{ep}", exist_ok=True)
            #     img.save(f"img/one_pc/scene_{scene}/ep_{ep}/step_{self._c}.png")
            # self._c += 1

            step_data = self._data_generator.generate(
                observations, rewards, dones, infos
            )

            for episode, done in zip(previous_episodes, dones):
                if done:
                    print("Done")
                    print(f"Episode {episode.episode_id}")
                    print(f"Scene: {episode.scene_id}")

            # If done, save the episode (the current obs is for the next ep):
            for ep, done in zip(rollout_data, dones):
                if done:
                    # Each element of ep is a list of different data types. Concat each
                    # data type into a separate array:
                    data = [np.stack([d[i] for d in ep]) for i in range(len(ep[0]))]
                    self._save_data(data)
                    ep.clear()

            for ep, data in zip(rollout_data, step_data):
                ep.append(data)

            ep_count += sum(dones)
            if sum(dones) > 0:
                print(f"Episodes completed: {ep_count}")


class ClipGenerator:

    def __init__(self, batch_size=64, device="cuda"):
        self._batch_size = batch_size
        self._device = device

        self._model = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-base-patch32",
        )
        self._processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

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

        clip_embeddings = torch.cat(clip_embeddings).detach().cpu().numpy()
        last_two_hidden_layers = (
            torch.cat(last_two_hidden_layers).detach().cpu().numpy()
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

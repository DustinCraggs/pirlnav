import glob
import gzip
import json
import os
from collections import defaultdict
import time
from typing import DefaultDict, Dict, List, Optional, Union

import numpy as np
import torch
from habitat.utils import profiling_wrapper
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import (
    append_text_to_image,
    draw_collision,
    images_to_video,
)
from numpy import ndarray
from torch import Tensor

from pirlnav.policy.models.resnet_gn import ResNet


class SimpleProfiler:
    def __init__(self):
        self._entry_times = {}
        self._total_times = defaultdict(float)
        self._counts = defaultdict(int)

    def reset(self):
        self._entry_times.clear()
        self._total_times.clear()
        self._counts.clear()

    def enter(self, key):
        self._entry_times[key] = time.time()

    def exit(self, key):
        self._total_times[key] += time.time() - self._entry_times[key]
        # Only increment count on exit so that stats will be more correct if accessed
        # before all keys have exited:
        self._counts[key] += 1

    def get_stats(self):
        return {k: v / self._counts[k] for k, v in self._total_times.items()}


def load_encoder(encoder, path):
    assert os.path.exists(path)
    if isinstance(encoder.backbone, ResNet):
        state_dict = torch.load(path, map_location="cpu")["teacher"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        # Print shapes recursively
        # print("\n".join([f"{k}: {v.shape}" for k, v in encoder.state_dict().items()]))
        # print("\n".join([f"{k}: {v.shape}" for k, v in state_dict.items()]))

        # expand backbone.conv1.weight if input channels do not match
        input_channels = encoder.backbone.conv1.weight.shape[1]
        if input_channels > 3:
            print("Expanding backbone.conv1.weight to match {input_channels=}")
            conv1_weight = state_dict["backbone.conv1.weight"]
            input_channels = encoder.backbone.conv1.weight.shape[1]
            num_new_channels = input_channels - 3
            # Arbitrarily use the first channel to initialize the new channels:
            new_channels = conv1_weight[:, :1, :, :].repeat(1, num_new_channels, 1, 1)
            conv1_weight = torch.cat([conv1_weight, new_channels], dim=1)
            state_dict["backbone.conv1.weight"] = conv1_weight

        return encoder.load_state_dict(state_dict=state_dict, strict=False)
    else:
        raise ValueError("unknown encoder backbone")


def observations_to_image(observation: Dict, info: Dict) -> np.ndarray:
    r"""Generate image of single frame from observation and info
    returned from a single environment step().

    Args:
        observation: observation returned from an environment step().
        info: info returned from an environment step().

    Returns:
        generated image of a single frame.
    """
    render_obs_images: List[np.ndarray] = []
    for sensor_name in observation:
        if "rgb" in sensor_name:
            rgb = observation[sensor_name]
            if not isinstance(rgb, np.ndarray):
                rgb = rgb.cpu().numpy()

            render_obs_images.append(rgb)
        elif "depth" in sensor_name:
            depth_map = observation[sensor_name].squeeze() * 255.0
            if not isinstance(depth_map, np.ndarray):
                depth_map = depth_map.cpu().numpy()

            depth_map = depth_map.astype(np.uint8)
            depth_map = np.stack([depth_map for _ in range(3)], axis=2)
            render_obs_images.append(depth_map)

    # add image goal if observation has image_goal info
    if "imagegoal" in observation or "imagegoalrotation" in observation:
        if "imagegoal" in observation:
            rgb = observation["imagegoal"]
        else:
            rgb = observation["imagegoalrotation"]
        if not isinstance(rgb, np.ndarray):
            rgb = rgb.cpu().numpy()

        render_obs_images.append(rgb)

    assert len(render_obs_images) > 0, "Expected at least one visual sensor enabled."

    # shapes_are_equal = len(set(x.shape for x in render_obs_images)) == 1
    # if not shapes_are_equal:
    #     render_frame = tile_images(render_obs_images)
    # else:
    #     render_frame = np.concatenate(render_obs_images, axis=1)

    render_frame = np.concatenate(render_obs_images, axis=1)
    # draw collision
    if "collisions" in info and info["collisions"]["is_collision"]:
        render_frame = draw_collision(render_frame)

    if "top_down_map" in info:
        top_down_map = maps.colorize_draw_agent_and_fit_to_height(
            info["top_down_map"], render_frame.shape[0]
        )
        render_frame = np.concatenate((render_frame, top_down_map), axis=1)
    return render_frame


def generate_video(
    video_option: List[str],
    video_dir: Optional[str],
    images: List[np.ndarray],
    episode_id: Union[int, str],
    checkpoint_idx: int,
    metrics: Dict[str, float],
    fps: int = 10,
    verbose: bool = True,
) -> None:
    r"""Generate video according to specified information.

    Args:
        video_option: string list of "tensorboard" or "disk" or both.
        video_dir: path to target video directory.
        images: list of images to be converted to video.
        episode_id: episode id for video naming.
        checkpoint_idx: checkpoint index for video naming.
        metric_name: name of the performance metric, e.g. "spl".
        metric_value: value of metric.
        tb_writer: tensorboard writer object for uploading video.
        fps: fps for generated video.
    Returns:
        None
    """
    if len(images) < 1:
        return

    metric_strs = []
    for k, v in metrics.items():
        metric_strs.append(f"{k}={v:.2f}")

    video_name = f"episode={episode_id}-ckpt={checkpoint_idx}-" + "-".join(metric_strs)
    if "disk" in video_option:
        assert video_dir is not None
        images_to_video(images, video_dir, video_name, verbose=verbose)


def add_info_to_image(frame, info):
    string = "d2g: {} | a2g: {} |\nsimple reward: {} |\nsuccess: {} | angle success: {}".format(
        round(info["distance_to_goal"], 3),
        round(info["angle_to_goal"], 3),
        round(info["simple_reward"], 3),
        round(info["success"], 3),
        round(info["angle_success"], 3),
    )
    frame = append_text_to_image(frame, string)
    return frame


def write_json(data, path):
    with open(path, "w") as file:
        file.write(json.dumps(data))


def load_dataset(path):
    with gzip.open(path, "rb") as file:
        data = json.loads(file.read(), encoding="utf-8")
    return data


def load_json_dataset(path):
    file = open(path, "r")
    data = json.loads(file.read())
    return data


def _to_tensor(v: Union[Tensor, ndarray]) -> torch.Tensor:
    if torch.is_tensor(v):
        return v
    elif isinstance(v, np.ndarray):
        if v.dtype == np.uint32:
            return torch.from_numpy(v.astype(int))
        else:
            return torch.from_numpy(v)
    else:
        return torch.tensor(v, dtype=torch.float)


@torch.no_grad()
@profiling_wrapper.RangeContext("batch_obs")
def batch_obs(
    observations: List[Dict],
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    r"""Transpose a batch of observation dicts to a dict of batched
    observations.

    Args:
        observations:  list of dicts of observations.
        device: The torch.device to put the resulting tensors on.
            Will not move the tensors if None

    Returns:
        transposed dict of torch.Tensor of observations.
    """
    batch: DefaultDict[str, List] = defaultdict(list)

    for obs in observations:
        for sensor in obs:
            batch[sensor].append(_to_tensor(obs[sensor]))

    batch_t: Dict[str, torch.Tensor] = {}

    for sensor in batch:
        batch_t[sensor] = torch.stack(batch[sensor], dim=0).to(device=device)

    return batch_t


def linear_warmup(
    epoch: int, start_update: int, max_updates: int, start_lr: int, end_lr: int
) -> float:
    r"""Returns a multiplicative factor for linear value decay

    Args:
        epoch: current epoch number
        total_num_updates: total number of

    Returns:
        multiplicative factor that decreases param value linearly
    """
    # logger.info("policy: {}, {}, {}, {}, {}".format(epoch, start_update, max_updates, start_lr, end_lr))
    if epoch < start_update:
        return 1.0

    if epoch > max_updates:
        return end_lr

    if max_updates == start_update:
        return end_lr

    pct_step = (epoch - start_update) / (max_updates - start_update)
    step_lr = (end_lr - start_lr) * pct_step + start_lr
    if step_lr > end_lr:
        step_lr = end_lr
    # logger.info("{}, {}, {}, {}, {}, {}".format(epoch, start_update, max_updates, start_lr, end_lr, step_lr))
    return step_lr


def critic_linear_decay(
    epoch: int, start_update: int, max_updates: int, start_lr: int, end_lr: int
) -> float:
    r"""Returns a multiplicative factor for linear value decay

    Args:
        epoch: current epoch number
        total_num_updates: total number of

    Returns:
        multiplicative factor that decreases param value linearly
    """
    # logger.info("critic lr: {}, {}, {}, {}, {}".format(epoch, start_update, max_updates, start_lr, end_lr))
    if epoch <= start_update:
        return 1

    if epoch >= max_updates:
        return end_lr

    if max_updates == start_update:
        return end_lr

    pct_step = (epoch - start_update) / (max_updates - start_update)
    step_lr = start_lr - (start_lr - end_lr) * pct_step
    if step_lr < end_lr:
        step_lr = end_lr
    # logger.info("{}, {}, {}, {}, {}, {}".format(epoch, start_update, max_updates, start_lr, end_lr, step_lr))
    return step_lr

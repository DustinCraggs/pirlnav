#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import os
import random
import time
from habitat.core.spaces import ActionSpace, EmptySpace
import zarr

from collections import defaultdict, deque
from typing import Any, Dict, List

import numpy as np
import torch
import tqdm
from gym import spaces
from habitat import Config, logger
from habitat.core.registry import registry
from habitat.core.environments import get_env_class
from habitat.utils import profiling_wrapper
from habitat.utils.render_wrapper import overlay_frame
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter, get_writer
from habitat_baselines.rl.ddppo.ddp_utils import (
    EXIT,
    add_signal_handlers,
    init_distrib_slurm,
    is_slurm_batch_job,
    load_resume_state,
    rank0_only,
    requeue_job,
    save_resume_state,
)
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer
from habitat_baselines.utils.common import (
    ObservationBatchingCache,
    action_array_to_dict,
    batch_obs,
    generate_video,
    get_num_actions,
    is_continuous_action_space,
    linear_decay,
)
from torch import nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from pirlnav.algos.agent import DDPILAgent, ILAgent
from pirlnav.common.rollout_storage import RolloutStorage
from pirlnav.gen_representation_dataset import ClipGenerator
from pirlnav.pvr_dataset import get_pvr_dataset
from pirlnav.utils.env_utils import construct_envs


# TODO:
# - Reduce precision of saved datasets
# - Policy eval
# - Each worker is training on the same data. Need to use a random offset for first
# dataloader.
# - Check if dataset is shuffled
# - Use multiple streams from the dataset
# - Manually inspect obs to ensure they are set
# TODO Next:
# - Ensure profiling is working and logged to wandb
#   - Need to log dataloader time, inference time, training iteration time
# - Use decoder for last bridge transformer layer
# - Generate a 10% CLIP dataset
# - Separate the visual and PVR policies; process nv obs differently for
#   pvr policy
# TODO reexamine:
# - Data generation
#   - Is there roughly uniform distribution of scene-goal pairs? If not, should sample
#     instead of using stride.
#   - Is the dataset order shuffled and deterministic?
# Done:
# - When running standard IL trainer, disable dataset sorting


@baseline_registry.register_trainer(name="pvr-pirlnav-il")
class PVRILEnvDDPTrainer(PPOTrainer):
    def __init__(self, config=None):
        super().__init__(config)
        self.action_space = self._make_action_space()

    def _make_action_space(self) -> spaces.Discrete:
        # TODO: There seems to be no easy way to get the action space without
        # constructing envs.
        sim_config = self.config.TASK_CONFIG.SIMULATOR
        action_space_name = sim_config.ACTION_SPACE_CONFIG
        action_config = registry.get_action_space_configuration(action_space_name)(
            sim_config
        ).get()
        action_dict = {v.name: EmptySpace() for v in action_config.values()}
        return ActionSpace(action_dict)

    def _make_observation_space(self, pvr_shapes, num_goals) -> spaces.Dict:
        # Make the (non-visual) task sensors to grab their observation spaces:
        # TODO: Initialising the sensors was too awkward, since they seem to all require
        # different args, so defining the space manually.
        obs_space = {}

        obs_space["objectgoal"] = spaces.Box(
            low=0,
            high=num_goals,
            shape=(1,),
            dtype=np.int64,
        )

        obs_space["gps"] = spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(2,),
            dtype=np.float32,
        )

        obs_space["inflection_weight"] = spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(1,),
            dtype=np.float32,
        )

        obs_space["compass"] = spaces.Box(
            low=-np.pi,
            high=np.pi,
            shape=(1,),
            dtype=np.float32,
        )

        # The first dimension of the shapes is the batch size:
        pvr_spaces = {
            k: spaces.Box(low=-np.inf, high=np.inf, shape=v[1:])
            for k, v in pvr_shapes.items()
        }
        obs_space = {**obs_space, **pvr_spaces}

        return spaces.Dict(obs_space)

    def _init_demonstration_dataset(self):
        pvr_config = self.config.TASK_CONFIG.PVR
        pvr_dataset = get_pvr_dataset(
            pvr_config.pvr_data_path,
            pvr_config.non_visual_obs_data_path,
            pvr_keys=pvr_config.pvr_keys,
            nv_keys=pvr_config.non_visual_keys,
            in_memory=False,
        )
        # TODO: This is assuming a single "environment". Thus, we need only num_steps
        # to fill the RolloutStorage. In future, we will create a separate stream from
        # the dataset for each "environment".
        # TODO: We need to shuffle the dataset in advance (and store on disk).
        batch_size = self.config.IL.BehaviorCloning.num_steps
        self._pvr_dataloader = DataLoader(pvr_dataset, batch_size=batch_size)
        self._pvr_dataloader_iter = iter(self._pvr_dataloader)

        example_batch = next(iter(self._pvr_dataloader))
        pvr_shapes = {k: example_batch[k].shape for k in pvr_config.pvr_keys}

        nv_dataset = zarr.open(pvr_config.non_visual_obs_data_path, mode="r")
        num_goals = max(nv_dataset["data"]["objectgoal"]) + 1
        self._obs_space = self._make_observation_space(pvr_shapes, num_goals)
        self._pvr_token_dim = pvr_shapes[pvr_config.pvr_keys[0]][-1]

    def _collect_demonstration_batch(self) -> int:
        """
        This replaces _collect_environment_result, as the environments are not required
        when using the PVR demonstration dataset.

        TODO: Currently requires single-buffer RolloutStorage and num_envs=1.
        """
        try:
            batch = next(self._pvr_dataloader_iter)
        except StopIteration:
            self._pvr_dataloader_iter = iter(self._pvr_dataloader)
            batch = next(self._pvr_dataloader_iter)

        # TODO: The observations from the dataset are already batched, but would using
        # the cache still meaningfully impact performance?
        obs_keys = self.config.TASK_CONFIG.PVR.obs_keys
        observations = {k: v.to(self.device) for k, v in batch.items() if k in obs_keys}
        actions = batch["next_actions"].to(self.device)
        rewards = batch["reward"].to(self.device)
        dones = batch["done"].to(self.device)

        # TODO: Need to feed actions and observations to the RolloutStorage even though
        # they're already batched (will replace RolloutStorage with a dataset next):
        num_steps = min(self.config.IL.BehaviorCloning.num_steps, len(actions))

        for step in range(num_steps):
            # Providing last action here (standard IL trainer does this on the next
            # iteration):
            this_step_obs = {k: v[step] for k, v in observations.items()}

            self.rollouts.insert(
                next_observations=this_step_obs,
                actions=actions[step],
                rewards=rewards[step],
                next_masks=~dones[step],
                buffer_index=0,
            )
            self.rollouts.advance_rollout(0)

        # Return number of steps collected
        return num_steps

    def _setup_actor_critic_agent(self, il_cfg: Config) -> None:
        r"""Sets up actor critic and agent for IL.

        Args:
            il_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        observation_space = self._obs_space
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )
        self.obs_space = observation_space

        policy = baseline_registry.get_policy(self.config.IL.POLICY.name)
        self.actor_critic = policy.from_config(
            self.config,
            observation_space,
            self.action_space,
            True,
            self._pvr_token_dim,
        )
        self.actor_critic.to(self.device)

        if not self.config.RL.DDPPO.train_encoder:
            self._static_encoder = True
            for param in self.actor_critic.net.visual_encoder.parameters():
                param.requires_grad_(False)

        agent_cls = ILAgent if not self._is_distributed else DDPILAgent
        self.agent = agent_cls(
            actor_critic=self.actor_critic,
            num_envs=1,
            num_mini_batch=il_cfg.num_mini_batch,
            lr=il_cfg.lr,
            encoder_lr=il_cfg.encoder_lr,
            eps=il_cfg.eps,
            max_grad_norm=il_cfg.max_grad_norm,
            wd=il_cfg.wd,
            entropy_coef=il_cfg.entropy_coef,
        )

    def _init_envs(self, config=None, shuffle_scenes: bool = True) -> None:
        if config is None:
            config = self.config

        self.envs = construct_envs(
            config,
            get_env_class(config.ENV_NAME),
            workers_ignore_signals=is_slurm_batch_job(),
            shuffle_scenes=shuffle_scenes,
        )

    def _init_train(self):
        resume_state = load_resume_state(self.config)
        if resume_state is not None:
            self.config: Config = resume_state["config"]

        if self.config.RL.DDPPO.force_distributed:
            self._is_distributed = True

        if is_slurm_batch_job():
            add_signal_handlers()

        # Add replay sensors
        self.config.defrost()
        self.config.TASK_CONFIG.TASK.SENSORS.extend(
            ["DEMONSTRATION_SENSOR", "INFLECTION_WEIGHT_SENSOR"]
        )
        self.config.freeze()

        if self._is_distributed:
            local_rank, tcp_store = init_distrib_slurm(
                self.config.RL.DDPPO.distrib_backend
            )
            if rank0_only():
                logger.info(
                    "Initialized DD-PPO with {} workers".format(
                        torch.distributed.get_world_size()
                    )
                )

            self.config.defrost()
            self.config.TORCH_GPU_ID = local_rank
            self.config.SIMULATOR_GPU_ID = local_rank
            # Multiply by the number of simulators to make sure they also get unique seeds
            self.config.TASK_CONFIG.SEED += (
                torch.distributed.get_rank() * self.config.NUM_ENVIRONMENTS
            )
            self.config.freeze()

            random.seed(self.config.TASK_CONFIG.SEED)
            np.random.seed(self.config.TASK_CONFIG.SEED)
            torch.manual_seed(self.config.TASK_CONFIG.SEED)
            self.num_rollouts_done_store = torch.distributed.PrefixStore(
                "rollout_tracker", tcp_store
            )
            self.num_rollouts_done_store.set("num_done", "0")

        if rank0_only() and self.config.VERBOSE:
            logger.info(f"config: {self.config}")

        profiling_wrapper.configure(
            capture_start_step=self.config.PROFILING.CAPTURE_START_STEP,
            num_steps_to_capture=self.config.PROFILING.NUM_STEPS_TO_CAPTURE,
        )

        self._init_demonstration_dataset()

        action_space = self.action_space
        self.policy_action_space = action_space

        if is_continuous_action_space(action_space):
            # Assume ALL actions are NOT discrete
            action_shape = (get_num_actions(action_space),)
            discrete_actions = False
        else:
            # For discrete pointnav
            action_shape = None
            discrete_actions = True

        il_cfg = self.config.IL.BehaviorCloning
        policy_cfg = self.config.POLICY
        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.config.TORCH_GPU_ID)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        if rank0_only() and not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)

        self._setup_actor_critic_agent(il_cfg)
        if self._is_distributed:
            self.agent.init_distributed(find_unused_params=True)  # type: ignore

        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )

        obs_space = self.obs_space
        if self._static_encoder:
            self._encoder = self.actor_critic.net.visual_encoder
            obs_space = spaces.Dict(
                {
                    "visual_features": spaces.Box(
                        low=np.finfo(np.float32).min,
                        high=np.finfo(np.float32).max,
                        shape=self._encoder.output_shape,
                        dtype=np.float32,
                    ),
                    **obs_space.spaces,
                }
            )

        self._nbuffers = 2 if il_cfg.use_double_buffered_sampler else 1

        self.rollouts = RolloutStorage(
            il_cfg.num_steps,
            self.config.NUM_ENVIRONMENTS,
            obs_space,
            self.policy_action_space,
            policy_cfg.STATE_ENCODER.hidden_size,
            num_recurrent_layers=self.actor_critic.net.num_recurrent_layers,
            is_double_buffered=il_cfg.use_double_buffered_sampler,
            action_shape=action_shape,
            discrete_actions=discrete_actions,
        )
        self.rollouts.to(self.device)

        num_envs = self.config.NUM_ENVIRONMENTS
        self.current_episode_reward = torch.zeros(num_envs, 1)
        self.running_episode_stats = dict(
            count=torch.zeros(num_envs, 1),
            reward=torch.zeros(num_envs, 1),
        )
        self.window_episode_stats = defaultdict(
            lambda: deque(maxlen=il_cfg.reward_window_size)
        )

        self.env_time = 0.0
        self.pth_time = 0.0
        self.t_start = time.time()

    def _compute_actions_and_step_envs(self, buffer_index: int = 0):
        num_envs = self.envs.num_envs
        env_slice = slice(
            int(buffer_index * num_envs / self._nbuffers),
            int((buffer_index + 1) * num_envs / self._nbuffers),
        )

        t_sample_action = time.time()

        # fetch actions from replay buffer
        step_batch = self.rollouts.buffers[
            self.rollouts.current_rollout_step_idxs[buffer_index],
            env_slice,
        ]
        next_actions = step_batch["observations"]["next_actions"]
        actions = next_actions.long().unsqueeze(-1)

        # NB: Move actions to CPU.  If CUDA tensors are
        # sent in to env.step(), that will create CUDA contexts
        # in the subprocesses.
        # For backwards compatibility, we also call .item() to convert to
        # an int
        actions = actions.to(device="cpu")
        self.pth_time += time.time() - t_sample_action

        profiling_wrapper.range_pop()  # compute actions

        t_step_env = time.time()

        for index_env, act in zip(
            range(env_slice.start, env_slice.stop), actions.unbind(0)
        ):
            if act.shape[0] > 1:
                step_action = action_array_to_dict(self.policy_action_space, act)
            else:
                step_action = act.item()
            self.envs.async_step_at(index_env, step_action)

        self.env_time += time.time() - t_step_env

        self.rollouts.insert(
            actions=actions,
            buffer_index=buffer_index,
        )

    @profiling_wrapper.RangeContext("_update_agent")
    def _update_agent(self):
        t_update_model = time.time()

        self.agent.train()

        (
            action_loss,
            rnn_hidden_states,
            dist_entropy,
            _,
        ) = self.agent.update(self.rollouts)

        self.rollouts.after_update(rnn_hidden_states)
        self.pth_time += time.time() - t_update_model

        return (
            action_loss,
            dist_entropy,
        )

    @profiling_wrapper.RangeContext("train")
    def train(self) -> None:
        r"""Main method for training DD/PPO.

        Returns:
            None
        """

        self._init_train()

        count_checkpoints = 0
        prev_time = 0

        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: 1 - self.percent_done(),
        )

        resume_state = load_resume_state(self.config)
        if resume_state is not None:
            self.agent.load_state_dict(resume_state["state_dict"])
            self.agent.optimizer.load_state_dict(resume_state["optim_state"])
            lr_scheduler.load_state_dict(resume_state["lr_sched_state"])

            requeue_stats = resume_state["requeue_stats"]
            self.env_time = requeue_stats["env_time"]
            self.pth_time = requeue_stats["pth_time"]
            self.num_steps_done = requeue_stats["num_steps_done"]
            self.num_updates_done = requeue_stats["num_updates_done"]
            self._last_checkpoint_percent = requeue_stats["_last_checkpoint_percent"]
            count_checkpoints = requeue_stats["count_checkpoints"]
            prev_time = requeue_stats["prev_time"]

            self.running_episode_stats = requeue_stats["running_episode_stats"]
            self.window_episode_stats.update(requeue_stats["window_episode_stats"])

        ppo_cfg = self.config.RL.PPO
        il_cfg = self.config.IL.BehaviorCloning

        with (
            get_writer(self.config, flush_secs=self.flush_secs)
            if rank0_only()
            else contextlib.suppress()
        ) as writer:
            while not self.is_done():
                profiling_wrapper.on_start_step()
                profiling_wrapper.range_push("train update")

                if il_cfg.use_linear_clip_decay:
                    self.agent.clip_param = il_cfg.clip_param * (
                        1 - self.percent_done()
                    )

                if rank0_only() and self._should_save_resume_state():
                    requeue_stats = dict(
                        env_time=self.env_time,
                        pth_time=self.pth_time,
                        count_checkpoints=count_checkpoints,
                        num_steps_done=self.num_steps_done,
                        num_updates_done=self.num_updates_done,
                        _last_checkpoint_percent=self._last_checkpoint_percent,
                        prev_time=(time.time() - self.t_start) + prev_time,
                        running_episode_stats=self.running_episode_stats,
                        window_episode_stats=dict(self.window_episode_stats),
                    )

                    save_resume_state(
                        dict(
                            state_dict=self.agent.state_dict(),
                            optim_state=self.agent.optimizer.state_dict(),
                            lr_sched_state=lr_scheduler.state_dict(),
                            config=self.config,
                            requeue_stats=requeue_stats,
                        ),
                        self.config,
                    )

                if EXIT.is_set():
                    profiling_wrapper.range_pop()  # train update
                    self.envs.close()
                    requeue_job()
                    return

                self.agent.eval()

                profiling_wrapper.range_push("rollouts loop")
                count_steps_delta = self._collect_demonstration_batch()
                profiling_wrapper.range_pop()  # rollouts loop

                if self._is_distributed:
                    self.num_rollouts_done_store.add("num_done", 1)

                (
                    action_loss,
                    dist_entropy,
                ) = self._update_agent()

                if il_cfg.use_linear_lr_decay:
                    lr_scheduler.step()  # type: ignore

                self.num_updates_done += 1
                losses = self._coalesce_post_step(
                    dict(
                        action_loss=action_loss,
                        entropy=dist_entropy,
                    ),
                    count_steps_delta,
                )

                self._training_log(writer, losses, prev_time)

                # checkpoint model
                if rank0_only() and self.should_checkpoint():
                    self.save_checkpoint(
                        f"ckpt.{count_checkpoints}.pth",
                        dict(
                            step=self.num_steps_done,
                            wall_time=(time.time() - self.t_start) + prev_time,
                        ),
                    )
                    count_checkpoints += 1

                profiling_wrapper.range_pop()  # train update

    @rank0_only
    def _training_log(self, writer, losses: Dict[str, float], prev_time: int = 0):
        deltas = {
            k: ((v[-1] - v[0]).sum().item() if len(v) > 1 else v[0].sum().item())
            for k, v in self.window_episode_stats.items()
        }
        deltas["count"] = max(deltas["count"], 1.0)

        writer.add_scalar(
            "reward",
            deltas["reward"] / deltas["count"],
            self.num_steps_done,
        )

        # Check to see if there are any metrics
        # that haven't been logged yet
        metrics = {
            k: v / deltas["count"]
            for k, v in deltas.items()
            if k not in {"reward", "count"}
        }

        for k, v in metrics.items():
            writer.add_scalar(f"metrics/{k}", v, self.num_steps_done)
        for k, v in losses.items():
            writer.add_scalar(f"losses/{k}", v, self.num_steps_done)

        fps = self.num_steps_done / ((time.time() - self.t_start) + prev_time)
        writer.add_scalar("metrics/fps", fps, self.num_steps_done)

        # log stats
        if self.num_updates_done % self.config.LOG_INTERVAL == 0:
            logger.info(
                "update: {}\tfps: {:.3f}\t".format(
                    self.num_updates_done,
                    fps,
                )
            )

            logger.info(
                "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                "frames: {}".format(
                    self.num_updates_done,
                    self.env_time,
                    self.pth_time,
                    self.num_steps_done,
                )
            )

            logger.info(
                "Average window size: {}  {}  {}".format(
                    len(self.window_episode_stats["count"]),
                    "  ".join(
                        "{}: {:.3f}".format(k, v / deltas["count"])
                        for k, v in deltas.items()
                        if k != "count"
                    ),
                    "  ".join("{}: {:.3f}".format(k, v) for k, v in losses.items()),
                )
            )

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        print(f"-------------- {checkpoint_path} --------------")
        # TODO: Don't need to init the whold dataset here just to get the metadata:
        self._init_demonstration_dataset()

        if self._is_distributed:
            raise RuntimeError("Evaluation does not support distributed mode")

        # Map location CPU is almost always better than mapping to a CUDA device.
        if self.config.EVAL.SHOULD_LOAD_CKPT:
            ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")
        else:
            ckpt_dict = {}

        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.TASK_CONFIG.DATASET.TYPE = "ObjectNav-v1"
        # Set to v2 when using the training dataset:
        # config.TASK_CONFIG.DATASET.TYPE = "ObjectNav-v2"
        config.freeze()

        if len(self.config.VIDEO_OPTION) > 0 and self.config.VIDEO_RENDER_TOP_DOWN:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()

        if config.VERBOSE:
            logger.info(f"env config: {config}")

        self._init_envs(config, shuffle_scenes=False)

        action_space = self.action_space
        if self.using_velocity_ctrl:
            # For navigation using a continuous action space for a task that
            # may be asking for discrete actions
            self.policy_action_space = action_space["VELOCITY_CONTROL"]
            action_shape = (2,)
            discrete_actions = False
        else:
            self.policy_action_space = action_space
            if is_continuous_action_space(action_space):
                # Assume NONE of the actions are discrete
                action_shape = (get_num_actions(action_space),)
                discrete_actions = False
            else:
                # For discrete pointnav
                action_shape = (1,)
                discrete_actions = True

        il_cfg = config.IL.BehaviorCloning
        policy_cfg = config.POLICY
        self._setup_actor_critic_agent(il_cfg)

        if self.agent.actor_critic.should_load_agent_state:
            self.agent.load_state_dict(
                {
                    k.replace("model.", "actor_critic."): v
                    for k, v in ckpt_dict["state_dict"].items()
                }
            )
        self.actor_critic = self.agent.actor_critic

        observations = self.envs.reset()
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        current_episode_reward = torch.zeros(self.envs.num_envs, 1, device="cpu")

        test_recurrent_hidden_states = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            self.actor_critic.net.num_recurrent_layers,
            policy_cfg.STATE_ENCODER.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            *action_shape,
            device=self.device,
            dtype=torch.long if discrete_actions else torch.float,
        )
        not_done_masks = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            1,
            device=self.device,
            dtype=torch.bool,
        )
        stats_episodes: Dict[Any, Any] = (
            {}
        )  # dict of dicts that stores stats per episode

        rgb_frames = [
            [] for _ in range(self.config.NUM_ENVIRONMENTS)
        ]  # type: List[List[np.ndarray]]
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        number_of_eval_episodes = self.config.TEST_EPISODE_COUNT
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            if total_num_eps < number_of_eval_episodes:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps

        # Make representation generator:
        generator_config = config.TASK_CONFIG.REPRESENTATION_GENERATOR
        data_generator_name = generator_config.data_generator_name

        if data_generator_name == "clip":
            data_generator = ClipGenerator()

        rewards, dones, infos = None, None, None

        pbar = tqdm.tqdm(total=number_of_eval_episodes)
        logger.info("Sampling actions deterministically...")
        self.actor_critic.eval()
        while len(stats_episodes) < number_of_eval_episodes and self.envs.num_envs > 0:
            current_episodes = self.envs.current_episodes()
            # Add PVR to batch:
            pvrs = data_generator.generate(
                observations,
                rewards,
                dones,
                infos,
                return_tensors=True,
            )
            pvrs = dict(zip(data_generator.data_names, pvrs))

            for k in config.TASK_CONFIG.PVR.pvr_keys:
                batch[k] = pvrs[k]

            # for k, v in batch.items():
            #     print(k, v)

            with torch.no_grad():
                (
                    actions,
                    test_recurrent_hidden_states,
                ) = self.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=True,
                )

                prev_actions.copy_(actions)  # type: ignore
            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            # For backwards compatibility, we also call .item() to convert to
            # an int
            if actions[0].shape[0] > 1:
                step_data = [
                    action_array_to_dict(self.policy_action_space, a)
                    for a in actions.to(device="cpu")
                ]
            else:
                step_data = [a.item() for a in actions.to(device="cpu")]

            outputs = self.envs.step(step_data)

            observations, rewards_l, dones, infos = [list(x) for x in zip(*outputs)]
            batch = batch_obs(  # type: ignore
                observations,
                device=self.device,
                cache=self._obs_batching_cache,
            )
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            )

            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device="cpu"
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                if (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                # episode ended
                if not not_done_masks[i].item():

                    # print("------------------------------------------------------------------")
                    # print("------------------------------------------------------------------")
                    # print("------------------------------------------------------------------")
                    # print("------------------------------------------------------------------")
                    # print("------------------------------------------------------------------")
                    # print("------------------------------------------------------------------")
                    # print(current_episodes)

                    pbar.update()
                    episode_stats = {"reward": current_episode_reward[i].item()}
                    episode_stats.update(self._extract_scalars_from_info(infos[i]))
                    current_episode_reward[i] = 0
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats

                    if len(self.config.VIDEO_OPTION) > 0:
                        ep_id = f"{current_episodes[i].scene_id.replace('/','_')}_{current_episodes[i].episode_id}"
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR,
                            images=rgb_frames[i],
                            episode_id=ep_id,
                            # episode_id=current_episodes[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metrics=self._extract_scalars_from_info(infos[i]),
                            fps=self.config.VIDEO_FPS,
                            tb_writer=writer,
                            keys_to_include_in_name=self.config.EVAL_KEYS_TO_INCLUDE_IN_NAME,
                        )

                        rgb_frames[i] = []

                # episode continues
                elif len(self.config.VIDEO_OPTION) > 0:
                    # TODO move normalization / channel changing out of the policy and undo it here
                    frame = observations_to_image(
                        {k: v[i] for k, v in batch.items()}, infos[i]
                    )
                    if self.config.VIDEO_RENDER_ALL_INFO:
                        frame = overlay_frame(frame, infos[i])

                    rgb_frames[i].append(frame)

            not_done_masks = not_done_masks.to(device=self.device)
            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        num_episodes = len(stats_episodes)
        aggregated_stats = {}
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum(v[stat_key] for v in stats_episodes.values()) / num_episodes
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        writer.add_scalar(
            "eval_reward/average_reward", aggregated_stats["reward"], step_id
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        for k, v in metrics.items():
            writer.add_scalar(f"eval_metrics/{k}", v, step_id)

        self.envs.close()

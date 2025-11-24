import os
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from typing import Optional
from collections import defaultdict
from gym import Space
from habitat import Config, logger
from habitat.tasks.nav.nav import EpisodicCompassSensor, EpisodicGPSSensor
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.models.rnn_state_encoder import build_rnn_state_encoder
from habitat_baselines.rl.ppo import Net

from pirlnav.policy.policy import ILPolicy
from pirlnav.policy.pvr_encoder import PvrEncoder
from pirlnav.policy.transforms import get_transform
from pirlnav.policy.visual_encoder import VisualEncoder
from pirlnav.utils.utils import load_encoder


class ObjectNavILMAENet(Net):
    r"""A baseline sequence to sequence network that concatenates instruction,
    RGB, and depth encodings before decoding an action distribution with an RNN.
    Modules:
        Instruction encoder
        Depth encoder
        RGB encoder
        RNN state encoder
    """

    def __init__(
        self,
        observation_space: Space,
        policy_config: Config,
        num_actions: int,
        run_type: str,
        hidden_size: int,
        rnn_type: str,
        num_recurrent_layers: int,
        use_pvr_encoder: bool = False,
        use_fixed_size_embedding: bool = False,
        pvr_token_dim: Optional[int] = None,
        pvr_obs_keys: Optional[list] = None,
        disable_visual_inputs: bool = False,
    ):
        super().__init__()
        self.policy_config = policy_config
        rnn_input_size = 0

        rgb_config = policy_config.RGB_ENCODER
        name = "resize"
        if rgb_config.use_augmentations and run_type == "train":
            name = rgb_config.augmentations_name
        if rgb_config.use_augmentations_test_time and run_type == "eval":
            name = rgb_config.augmentations_name
        self.visual_transform = get_transform(name, size=rgb_config.image_size)
        self.visual_transform.randomize_environments = (
            rgb_config.randomize_augmentations_over_envs
        )

        self.use_pvr_encoder = use_pvr_encoder
        self.use_fixed_size_embedding = use_fixed_size_embedding
        self.pvr_obs_keys = pvr_obs_keys

        self.disable_visual_inputs = disable_visual_inputs

        if EpisodicGPSSensor.cls_uuid in observation_space.spaces:
            input_gps_dim = observation_space.spaces[EpisodicGPSSensor.cls_uuid].shape[
                0
            ]
            self.gps_embedding = nn.Linear(input_gps_dim, 32)
            rnn_input_size += 32
            logger.info("\n\nSetting up GPS sensor")

        if EpisodicCompassSensor.cls_uuid in observation_space.spaces:
            assert (
                observation_space.spaces[EpisodicCompassSensor.cls_uuid].shape[0] == 1
            ), "Expected compass with 2D rotation."
            input_compass_dim = 2  # cos and sin of the angle
            self.compass_embedding_dim = 32
            self.compass_embedding = nn.Linear(
                input_compass_dim, self.compass_embedding_dim
            )
            rnn_input_size += 32
            logger.info("\n\nSetting up Compass sensor")

        if ObjectGoalSensor.cls_uuid in observation_space.spaces:
            self._n_object_categories = (
                int(observation_space.spaces[ObjectGoalSensor.cls_uuid].high[0]) + 1
            )
            logger.info("Object categories: {}".format(self._n_object_categories))
            self.obj_categories_embedding = nn.Embedding(self._n_object_categories, 32)
            rnn_input_size += 32
            logger.info("\n\nSetting up Object Goal sensor")

        if policy_config.SEQ2SEQ.use_prev_action:
            self.prev_action_embedding = nn.Embedding(num_actions + 1, 32)
            rnn_input_size += self.prev_action_embedding.embedding_dim

        if self.disable_visual_inputs:
            self.visual_encoder = None
        elif self.use_fixed_size_embedding:
            rnn_input_size += pvr_token_dim
        elif use_pvr_encoder:
            self.non_visual_embedding = nn.Linear(rnn_input_size, pvr_token_dim)
            self.pvr_encoder = PvrEncoder(
                pvr_token_dim,
                policy_config.PVR_ENCODER.num_heads,
                policy_config.PVR_ENCODER.num_layers,
                policy_config.PVR_ENCODER.dropout,
            )
            # policy_config.RGB_ENCODER.hidden_size
            rnn_input_size += pvr_token_dim
        else:
            self._costmap_names = rgb_config.get("costmap_names", [])

            if self._costmap_names:

                def _costmap_transform(img):
                    img = img.permute(0, 3, 1, 2)
                    img = TF.resize(img, rgb_config.image_size)
                    img = TF.center_crop(img, output_size=rgb_config.image_size)
                    return img.permute(0, 2, 3, 1)

                self._costmap_resize = _costmap_transform

            self.visual_encoder = VisualEncoder(
                image_size=rgb_config.image_size,
                backbone=rgb_config.backbone,
                input_channels=rgb_config.input_channels,
                resnet_baseplanes=rgb_config.resnet_baseplanes,
                resnet_ngroups=rgb_config.resnet_baseplanes // 2,
                avgpooled_image=rgb_config.avgpooled_image,
                drop_path_rate=rgb_config.drop_path_rate,
            )

            self.visual_fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(
                    self.visual_encoder.output_size,
                    policy_config.RGB_ENCODER.hidden_size,
                ),
                nn.ReLU(True),
            )

            rnn_input_size += policy_config.RGB_ENCODER.hidden_size
            logger.info("RGB encoder is {}".format(policy_config.RGB_ENCODER.backbone))

            # load pretrained weights
            if rgb_config.pretrained_encoder is not None:
                msg = load_encoder(self.visual_encoder, rgb_config.pretrained_encoder)
                logger.info(
                    "Using weights from {}: {}".format(
                        rgb_config.pretrained_encoder, msg
                    )
                )

            # freeze backbone
            if rgb_config.freeze_backbone:
                for p in self.visual_encoder.backbone.parameters():
                    p.requires_grad = False

        self.rnn_input_size = rnn_input_size

        logger.info(
            "State enc: rnn_input_size {} - hidden_size {} - rnn_type {} - num_recurrent_layers {}".format(
                rnn_input_size, hidden_size, rnn_type, num_recurrent_layers
            )
        )

        self.state_encoder = build_rnn_state_encoder(
            rnn_input_size,
            hidden_size=hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )
        self._hidden_size = hidden_size
        self.train()

        self.use_final_obs_resid_mlp = policy_config.SEQ2SEQ.use_final_obs_resid_mlp
        if self.use_final_obs_resid_mlp:
            # self.final_obs_resid_mlp = nn.Sequential(
            #     nn.Linear(hidden_size + rnn_input_size, hidden_size),
            #     nn.ReLU(True),
            #     nn.Linear(hidden_size, hidden_size),
            # )
            num_final_mlp_layers = 3
            input_size = hidden_size + rnn_input_size
            final_mlp_layers = []
            for _ in range(num_final_mlp_layers - 1):
                final_mlp_layers.append(
                    nn.Sequential(
                        nn.Linear(input_size, hidden_size),
                        nn.LayerNorm(hidden_size),
                        nn.ReLU(True),
                    )
                )
                input_size = hidden_size

            final_mlp_layers.append(nn.Linear(hidden_size, hidden_size))
            self.final_obs_resid_mlp = nn.Sequential(*final_mlp_layers)

        self.batch_store = defaultdict(list)

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind and self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        r"""
        instruction_embedding: [batch_size x INSTRUCTION_ENCODER.output_size]
        depth_embedding: [batch_size x DEPTH_ENCODER.output_size]
        rgb_embedding: [batch_size x RGB_ENCODER.output_size]
        """
        N = rnn_hidden_states.size(1)
        x = []

        if EpisodicGPSSensor.cls_uuid in observations:
            obs_gps = observations[EpisodicGPSSensor.cls_uuid]

            if len(obs_gps.size()) == 3:
                obs_gps = obs_gps.contiguous().view(-1, obs_gps.size(2))
            x.append(self.gps_embedding(obs_gps))

        if EpisodicCompassSensor.cls_uuid in observations:
            obs_compass = observations["compass"]

            if len(obs_compass.size()) == 3:
                obs_compass = obs_compass.contiguous().view(-1, obs_compass.size(2))
            compass_observations = torch.stack(
                [
                    torch.cos(obs_compass),
                    torch.sin(obs_compass),
                ],
                -1,
            )
            compass_embedding = self.compass_embedding(
                compass_observations.float().squeeze(dim=1)
            )
            x.append(compass_embedding)

        if ObjectGoalSensor.cls_uuid in observations:
            object_goal = observations[ObjectGoalSensor.cls_uuid].long()
            if len(object_goal.size()) == 3:
                object_goal = object_goal.contiguous().view(-1, object_goal.size(2))
            x.append(self.obj_categories_embedding(object_goal).squeeze(dim=1))

        if self.policy_config.SEQ2SEQ.use_prev_action:
            prev_actions_embedding = self.prev_action_embedding(
                ((prev_actions.float() + 1) * masks).long().view(-1)
            )
            x.append(prev_actions_embedding)

        if self.disable_visual_inputs:
            pass
        elif self.use_fixed_size_embedding:
            for k in self.pvr_obs_keys:
                # Remove extra dimensions that exist to match sequential PVRs:
                x.append(observations[k].squeeze(1).squeeze(1))
        elif self.use_pvr_encoder:
            nv_obs = torch.cat(x, dim=1)
            nv_tokens = self.non_visual_embedding(nv_obs).unsqueeze(1)
            # nv_tokens = None
            # TODO: For multi-PVR, use single encoder and project to same
            # dimensionality
            pvr_tokens = torch.cat([observations[k] for k in self.pvr_obs_keys])

            pvr_embedding = self.pvr_encoder(pvr_tokens, nv_tokens)
            x.append(pvr_embedding)
        elif self.visual_encoder is not None:
            rgb_obs = observations["rgb"]

            if self._costmap_names:
                # Pre-resize the RGB observation to match costmap size:
                rgb_obs = self._costmap_resize(rgb_obs)

            # Channel-wise stack rgb and costmaps. This needs to occur before visual
            # transforms in order to apply the same augmentations:
            for costmap_name in self._costmap_names:
                orig_shape = observations[costmap_name].shape
                costmap = self._costmap_resize(observations[costmap_name])

                # if costmap_name == "goal_costmap":
                #     # Convert boolean goal_costmap to float (quick hack as the
                #     # goal_costmap is only for testing currently):
                #     costmap = costmap * 255.0
                if rgb_obs.shape[:3] != costmap.shape[:3]:
                    print("SHAPE MISMATCH")
                    print(f"{rgb_obs.shape=} {costmap.shape=}")

                # Channel stack:
                rgb_obs = torch.cat([rgb_obs, costmap], dim=-1)

            # observations["rgb"] = rgb_obs
            # if len(rgb_obs.size()) == 5:
            #     observations["rgb"] = rgb_obs.contiguous().view(
            #         -1, rgb_obs.size(2), rgb_obs.size(3), rgb_obs.size(4)
            #     )

            # visual encoder
            # rgb = observations["rgb"]
            rgb = rgb_obs
            rgb = self.visual_transform(rgb, N)
            rgb = self.visual_encoder(rgb)
            rgb = self.visual_fc(rgb)
            x.append(rgb)

        x = torch.cat(x, dim=1)
        all_obs = x

        # TODO: Shouldn't there be a projection here to use same dim for RNN every time?

        x, rnn_hidden_states = self.state_encoder(
            x, rnn_hidden_states.contiguous(), masks
        )

        if self.use_final_obs_resid_mlp:
            x_obs = torch.cat((x, all_obs), dim=1)
            x = self.final_obs_resid_mlp(x_obs)

        return x, rnn_hidden_states


@baseline_registry.register_policy
class ObjectNavILMAEPolicy(ILPolicy):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        policy_config: Config,
        run_type: str,
        hidden_size: int,
        rnn_type: str,
        num_recurrent_layers: int,
        use_pvr_encoder: bool = False,
        use_fixed_size_embedding: bool = False,
        pvr_token_dim: Optional[int] = None,
        pvr_obs_keys: Optional[list] = None,
    ):
        super().__init__(
            ObjectNavILMAENet(
                observation_space=observation_space,
                policy_config=policy_config,
                num_actions=action_space.n,
                run_type=run_type,
                hidden_size=hidden_size,
                rnn_type=rnn_type,
                num_recurrent_layers=num_recurrent_layers,
                use_pvr_encoder=use_pvr_encoder,
                use_fixed_size_embedding=use_fixed_size_embedding,
                pvr_token_dim=pvr_token_dim,
                pvr_obs_keys=pvr_obs_keys,
            ),
            action_space.n,
            no_critic=policy_config.CRITIC.no_critic,
            mlp_critic=policy_config.CRITIC.mlp_critic,
            critic_hidden_dim=policy_config.CRITIC.hidden_dim,
        )

    @classmethod
    def from_config(
        cls,
        config: Config,
        observation_space,
        action_space,
        pvr_token_dim=None,
    ):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            policy_config=config.POLICY,
            run_type=config.RUN_TYPE,
            hidden_size=config.POLICY.STATE_ENCODER.hidden_size,
            rnn_type=config.POLICY.STATE_ENCODER.rnn_type,
            num_recurrent_layers=config.POLICY.STATE_ENCODER.num_recurrent_layers,
            use_pvr_encoder=config.TASK_CONFIG.PVR.use_pvr_encoder,
            use_fixed_size_embedding=config.TASK_CONFIG.PVR.use_fixed_size_embedding,
            pvr_token_dim=pvr_token_dim,
            pvr_obs_keys=config.TASK_CONFIG.PVR.pvr_keys,
        )

    @property
    def num_recurrent_layers(self):
        return self.net.num_recurrent_layers

    def freeze_visual_encoders(self):
        for param in self.net.visual_encoder.parameters():
            param.requires_grad_(False)

    def unfreeze_visual_encoders(self):
        for param in self.net.visual_encoder.parameters():
            param.requires_grad_(True)

    def freeze_state_encoder(self):
        for param in self.net.state_encoder.parameters():
            param.requires_grad_(False)

    def unfreeze_state_encoder(self):
        for param in self.net.state_encoder.parameters():
            param.requires_grad_(True)

    def freeze_actor(self):
        for param in self.action_distribution.parameters():
            param.requires_grad_(False)

    def unfreeze_actor(self):
        for param in self.action_distribution.parameters():
            param.requires_grad_(True)

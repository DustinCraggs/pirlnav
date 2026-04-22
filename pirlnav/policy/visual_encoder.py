import numpy as np
import torch
from habitat_baselines.rl.ddppo.policy.running_mean_and_var import RunningMeanAndVar
from torch import nn as nn
from torch.nn import functional as F

from pirlnav.policy.models import resnet_gn as resnet


class VisualEncoder(nn.Module):
    def __init__(
        self,
        image_size: int,
        backbone: str,
        input_channels: int = 3,
        costmap_channels: int = 0,
        resnet_baseplanes: int = 32,
        resnet_ngroups: int = 32,
        normalize_visual_inputs: bool = True,
        avgpooled_image: bool = False,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.avgpooled_image = avgpooled_image
        self.costmap_channels = costmap_channels

        if normalize_visual_inputs:
            self.running_mean_and_var: nn.Module = RunningMeanAndVar(
                input_channels
            )
        else:
            self.running_mean_and_var = nn.Sequential()

        if "resnet" in backbone:
            make_backbone = getattr(resnet, backbone)
            # Init backbone with standard RGB input channels to allow checkpoint loading
            self.backbone = make_backbone(
                input_channels, resnet_baseplanes, resnet_ngroups
            )

            spatial_size = image_size
            if self.avgpooled_image:
                spatial_size = image_size // 2

            final_spatial = int(spatial_size * self.backbone.final_spatial_compress)
            after_compression_flat_size = 2048
            num_compression_channels = int(
                round(after_compression_flat_size / (final_spatial**2))
            )
            self.compression = nn.Sequential(
                nn.Conv2d(
                    self.backbone.final_channels,
                    num_compression_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(1, num_compression_channels),
                nn.ReLU(True),
            )

            output_shape = (
                num_compression_channels,
                final_spatial,
                final_spatial,
            )
            self.output_size = np.prod(output_shape)
        else:
            raise ValueError("unknown backbone {}".format(backbone))

    def set_up_costmap_stem(self):
        """
        Expand the first convolutional layer to accept additional (costmap) channels.
        This should be called after loading any pre-trained weights. New channels
        are zero-initialized, and the RGB weights are retained.
        """
        if self.costmap_channels == 0:
            return

        old_conv1 = self.backbone.conv1
        new_conv1 = nn.Conv2d(
            in_channels=old_conv1.in_channels + self.costmap_channels,
            out_channels=old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=old_conv1.bias is not None,
        )

        # Copy the pre-trained RGB weights:
        new_conv1.weight.data[:, : old_conv1.in_channels, :, :] = old_conv1.weight.data

        # Zero-initialise the new costmap channels:
        nn.init.zeros_(new_conv1.weight.data[:, old_conv1.in_channels :, :, :])

        if old_conv1.bias is not None:
            new_conv1.bias.data = old_conv1.bias.data

        # Replace the layer:
        self.backbone.conv1 = new_conv1

        # Ensure it requires gradients (important if backbone gets frozen):
        self.backbone.conv1.weight.requires_grad = True

        if hasattr(self, 'running_mean_and_var'):
            old_rmv = self.running_mean_and_var
            new_rmv = RunningMeanAndVar(old_conv1.in_channels + self.costmap_channels)
            new_rmv.to(old_rmv._mean.device)

            # Copy pre-trained ImageNet/PIRLNav stats for the first 3 channels
            new_rmv._mean.data[:, :old_conv1.in_channels, :, :] = old_rmv._mean.data
            new_rmv._var.data[:, :old_conv1.in_channels, :, :] = old_rmv._var.data
            new_rmv._count.data = old_rmv._count.data

            # The 30 new costmap channels will safely default to mean=0, var=1
            self.running_mean_and_var = new_rmv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.avgpooled_image:
            x = F.avg_pool2d(x, 2)
        x = self.running_mean_and_var(x)
        x = self.backbone(x)
        x = self.compression(x)
        return x

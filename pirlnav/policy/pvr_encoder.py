import numpy as np
import torch
import einops

from torch import nn

# TODO:
# - Use a single decoder layer as the final layer to avoid processing obs queries


class PvrEncoder(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        num_layers,
        dropout=0.1,
        layer_norm_eps=1e-5,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=norm,
        )

        self.act_token = nn.Embedding(1, embed_dim)
        # self.act_token = nn.Parameter(torch.randn(1, embed_dim))

    def forward(self, pvr_tokens, non_visual_tokens=None):
        r"""
        Args:
            pvr_tokens (torch.Tensor): Shape (batch, pvr_seq_len, embed_dim)
            non_visual_tokens (torch.Tensor): Shape (batch, nv_seq_len, embed_dim)

        Returns:
            torch.Tensor: Embedded tokens of shape
                (batch, pvr_seq_len + nv_seq_len + 1, embed_dim)
        """
        # print(f"pvr_tokens initial {pvr_tokens.shape}")
        # Concatenate PVR tokens. This could be across layers of the PVR transformer,
        # or across the image patch dimensions:
        pvr_tokens = einops.rearrange(
            pvr_tokens, "b layer seq tok -> b (layer seq) tok"
        )
        act_token = self.act_token(
            torch.zeros((pvr_tokens.size(0), 1), dtype=int).to(pvr_tokens.device)
        )
        # act_token = self.act_token.expand(pvr_tokens.size(0), -1, -1)
        # print(f"pvr_tokens {pvr_tokens.shape}")
        # print(f"act_token {act_token.shape}")

        if non_visual_tokens is None:
            seq = torch.cat([pvr_tokens, act_token], dim=1)
        else:
            # print(f"non_visual_tokens {non_visual_tokens.shape}")
            seq = torch.cat([pvr_tokens, non_visual_tokens, act_token], dim=1)

        # Only return last token, as this is the ACT token:
        # print(f"seq {seq.shape}")
        output = self.encoder(seq)
        # print(f"output shape {output.shape}")
        # print(f"output {output}")
        # print(f"output final {output[:, -1, :]}")
        # print(f"output final shape {output[:, -1, :].shape}")
        return output[:, -1, :]
        # return self.encoder(seq)[:, -1, :]

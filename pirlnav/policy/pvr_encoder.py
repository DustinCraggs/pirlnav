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
        )
        norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=norm,
        )

        self.act_token = nn.Parameter(torch.randn(1, embed_dim))

    def forward(self, pvr_tokens, non_visual_tokens):
        r"""
        Args:
            pvr_tokens (torch.Tensor): Shape (batch, pvr_seq_len, embed_dim)
            non_visual_tokens (torch.Tensor): Shape (batch, nv_seq_len, embed_dim)

        Returns:
            torch.Tensor: Embedded tokens of shape
                (batch, pvr_seq_len + nv_seq_len + 1, embed_dim)
        """
        pvr_tokens = einops.rearrange(
            pvr_tokens, "b layer seq tok -> b (layer seq) tok"
        )
        act_token = self.act_token.expand(pvr_tokens.size(0), -1, -1)

        seq = torch.cat([pvr_tokens, non_visual_tokens, act_token], dim=1)
        # Only return last token, as this is the ACT token:
        return self.encoder(seq)[:, -1, :]

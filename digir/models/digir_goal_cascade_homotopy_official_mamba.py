"""
DIGIR variant: Homotopy Multi-Path Unrolling with official Mamba blocks.

Pipeline:
    history + KG -> dual-granularity encoder -> official Mamba Goal Decoder
    -> homotopy multi-path unrolling -> official Mamba over each 1D path
    -> official Mamba over future steps -> multi-modal trajectories.

This file does NOT overwrite existing model scripts. It keeps the same
homotopy/path-unrolling architecture as
digir_goal_cascade_homotopy_spatial_temporal_ssm.py, but replaces the custom
SelectiveSSMBlock with the official mamba-ssm Mamba layer.
"""
from __future__ import annotations

import torch.nn as nn

try:
    from mamba_ssm import Mamba
except Exception as exc:  # pragma: no cover - depends on server CUDA extension
    Mamba = None
    _MAMBA_IMPORT_ERROR = exc
else:
    _MAMBA_IMPORT_ERROR = None

from models.digir_goal_cascade_homotopy_spatial_temporal_ssm import (
    DIGIR as HomotopySpatialTemporalSSM,
)


class OfficialMambaBlock(nn.Module):
    """
    Residual wrapper around the official mamba-ssm Mamba layer.

    The official Mamba module maps (B, L, D) -> (B, L, D). This wrapper adds the
    normalization, residual connection, dropout, and FFN that the previous custom
    SelectiveSSMBlock already provided, so it can be swapped in safely.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if Mamba is None:
            raise ImportError(
                "Official Mamba block requires `mamba-ssm`. Install it on the server with:\n"
                "  pip install causal-conv1d>=1.4.0 --no-build-isolation\n"
                "  pip install mamba-ssm[causal-conv1d] --no-build-isolation"
            ) from _MAMBA_IMPORT_ERROR

        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=int(d_model),
            d_state=int(d_state),
            d_conv=int(d_conv),
            expand=int(expand),
        )
        self.dropout = nn.Dropout(float(dropout))
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x):
        """
        Args:
            x: (B, L, D)
        Returns:
            (B, L, D)
        """
        x = x + self.dropout(self.mamba(self.norm(x)))
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x


class DIGIR(HomotopySpatialTemporalSSM):
    """
    Homotopy-guided spatial-temporal decoder using official Mamba blocks.

    Only the sequence modeling blocks are changed:
      - goal_ssm: official Mamba over goal queries
      - spatial_ssm: official Mamba over unrolled corridor nodes
      - route_ssm: official Mamba over future time steps

    All data loading, losses, evaluation, DDP, and homotopy path construction are
    inherited unchanged from the existing training/model stack.
    """

    def __init__(self, config):
        super().__init__(config)

        d = int(self.d_model)
        dropout = float(config.get("dropout", 0.1))
        expand = int(config.get("official_mamba_expand", config.get("mamba_expansion", 2)))
        d_state = int(config.get("official_mamba_d_state", 16))
        d_conv = int(config.get("official_mamba_d_conv", 4))

        goal_layers = int(config.get("mamba_goal_layers", 2))
        spatial_layers = int(config.get("spatial_ssm_layers", 2))
        temporal_layers = int(config.get("mamba_route_layers", 4))

        def make_block():
            return OfficialMambaBlock(
                d_model=d,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )

        # Replace the inherited custom SelectiveSSMBlock stacks with official Mamba.
        self.goal_ssm = nn.ModuleList([make_block() for _ in range(goal_layers)])
        self.spatial_ssm = nn.ModuleList([make_block() for _ in range(spatial_layers)])
        self.route_ssm = nn.ModuleList([make_block() for _ in range(temporal_layers)])

        self.official_mamba_config = {
            "d_state": d_state,
            "d_conv": d_conv,
            "expand": expand,
            "goal_layers": goal_layers,
            "spatial_layers": spatial_layers,
            "temporal_layers": temporal_layers,
        }

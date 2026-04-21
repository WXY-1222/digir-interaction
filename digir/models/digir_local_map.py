"""
DIGIR variant with local sparse map cross-attention.

This file intentionally does NOT modify the original `models/digir.py`.
It subclasses the original DIGIR model and only changes scene encoding:
each vehicle attends to a local subset of map/KG nodes.
"""
import torch
import torch.nn as nn

from models.digir import DIGIR as BaseDIGIR
from models.dual_granularity.cross_attention import CrossAttention


class LocalSparseContextExtractor(nn.Module):
    """
    Local map-aware context extractor.

    For each vehicle query, keep only nearby map nodes:
    - optional radius filter
    - top-k nearest filter
    Then apply multi-head cross-attention on the masked keys.
    """
    def __init__(self, d_model, num_heads=4, dropout=0.1, local_topk=32, local_radius=None):
        super().__init__()
        self.cross_attn = CrossAttention(d_model, num_heads, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.local_topk = int(local_topk) if local_topk is not None else 32
        self.local_radius = float(local_radius) if local_radius is not None else None

    def _build_local_mask(self, vehicle_xy, map_xy, vehicle_mask=None):
        """
        Args:
            vehicle_xy: (B, N, 2)
            map_xy: (B, M, 2)
            vehicle_mask: (B, N) optional
        Returns:
            attn_mask: (B, N, M), bool
        """
        # Pairwise distance vehicle -> map nodes
        # dist2: (B, N, M)
        dist2 = torch.sum((vehicle_xy.unsqueeze(2) - map_xy.unsqueeze(1)) ** 2, dim=-1)
        B, N, M = dist2.shape

        attn_mask = torch.ones((B, N, M), dtype=torch.bool, device=dist2.device)

        # Radius-based locality
        if self.local_radius is not None and self.local_radius > 0:
            radius2 = self.local_radius * self.local_radius
            attn_mask = attn_mask & (dist2 <= radius2)

        # Top-k nearest map nodes per agent
        if self.local_topk > 0 and self.local_topk < M:
            k = self.local_topk
            topk_idx = torch.topk(dist2, k=k, dim=-1, largest=False).indices  # (B, N, k)
            topk_mask = torch.zeros_like(attn_mask)
            topk_mask.scatter_(dim=2, index=topk_idx, value=True)
            attn_mask = attn_mask & topk_mask

        # Guarantee at least one valid key for every query to avoid degenerate masks
        any_valid = attn_mask.any(dim=-1, keepdim=True)  # (B, N, 1)
        nearest_idx = torch.argmin(dist2, dim=-1, keepdim=True)  # (B, N, 1)
        fallback_mask = torch.zeros_like(attn_mask)
        fallback_mask.scatter_(dim=2, index=nearest_idx, value=True)
        attn_mask = torch.where(any_valid, attn_mask, fallback_mask)

        if vehicle_mask is not None:
            attn_mask = attn_mask & vehicle_mask.unsqueeze(-1).bool()

        return attn_mask

    def forward(self, vehicle_state, graph_embeddings, graph_positions, vehicle_xy, vehicle_mask=None):
        """
        Args:
            vehicle_state: (B, N, d)
            graph_embeddings: (B, M, d)
            graph_positions: (B, M, 2)
            vehicle_xy: (B, N, 2)
            vehicle_mask: (B, N) optional
        """
        mask = self._build_local_mask(
            vehicle_xy=vehicle_xy,
            map_xy=graph_positions,
            vehicle_mask=vehicle_mask,
        )  # (B, N, M)

        attn_out, _ = self.cross_attn(vehicle_state, graph_embeddings, graph_embeddings, mask=mask)
        local_context = self.norm(vehicle_state + self.dropout(attn_out))

        if vehicle_mask is not None:
            local_context = local_context * vehicle_mask.unsqueeze(-1).float()

        return local_context


class DIGIR(BaseDIGIR):
    """
    DIGIR with local sparse map cross-attention.

    Keep all original training/eval/loss logic. Only replace the map context extraction
    part in encode_scene.
    """
    def __init__(self, config):
        super().__init__(config)
        self.local_context_extractor = LocalSparseContextExtractor(
            d_model=self.d_model,
            num_heads=config.get("num_heads", 4),
            dropout=config.get("dropout", 0.1),
            local_topk=config.get("local_map_topk", 32),
            local_radius=config.get("local_map_radius", None),
        )

    def encode_scene(self, trajectories, kg_data, vehicle_masks=None):
        """
        Override only scene encoding:
        - trajectory encoder unchanged
        - graph encoder unchanged
        - local context extraction switched to sparse local map attention
        - scene transformer unchanged
        """
        batch_size, _, _, _ = trajectories.shape

        # 1. Motion summaries (unchanged)
        motion_summaries = self.traj_encoder(trajectories)
        if vehicle_masks is not None:
            motion_summaries = motion_summaries * vehicle_masks.unsqueeze(-1).float()

        # 2-3. KG + local sparse cross-attention
        if self.ablate_cross_attn:
            local_contexts = motion_summaries
        else:
            graph_embeddings = self.graph_encoder(
                kg_data["facility_types"],
                kg_data["positions"],
                kg_data["edge_index"],
                kg_data.get("edge_types"),
            )  # (B, M, d)

            graph_pos = kg_data["positions"]
            if graph_pos.dim() == 2:
                graph_pos = graph_pos.unsqueeze(0).expand(batch_size, -1, -1)

            # Use the latest observed position as vehicle query center in the normalized frame
            vehicle_xy = trajectories[:, :, -1, :2]

            local_contexts = self.local_context_extractor(
                vehicle_state=motion_summaries,
                graph_embeddings=graph_embeddings,
                graph_positions=graph_pos,
                vehicle_xy=vehicle_xy,
                vehicle_mask=vehicle_masks,
            )

        # 4. Scene pooling (unchanged)
        scene_intent, local_contexts = self.scene_transformer(
            local_contexts,
            vehicle_mask=vehicle_masks,
        )
        return scene_intent, local_contexts, motion_summaries


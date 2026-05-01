"""
DIGIR variant: official Mamba homotopy decoder with geometry-aware KG corridors.

This file does NOT overwrite existing model scripts. It keeps the official Mamba
Goal/Spatial/Temporal decoder from digir_goal_cascade_homotopy_official_mamba.py,
but changes the corridor endpoint selection:

    old: start/goal KG nodes selected only by embedding similarity
    new: start/goal KG nodes selected by embedding similarity + geometric distance

The goal is to make homotopy corridors physically closer to the observed agent
position and predicted endpoint, especially in per-agent normalized training.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from models.digir_goal_cascade_homotopy_official_mamba import (
    DIGIR as OfficialMambaHomotopyDIGIR,
)


class DIGIR(OfficialMambaHomotopyDIGIR):
    """
    Geometry-aware KG corridor variant.

    The training script injects kg_data["agent_anchor_points"] in the same frame as
    kg_data["positions"]. For each agent, map nodes are shifted by that anchor so
    the local agent origin is (0,0), matching the decoder's local goal coordinates.
    """

    def __init__(self, config):
        super().__init__(config)
        self.geo_embed_weight = float(config.get("geo_corridor_embed_weight", 1.0))
        self.geo_dist_weight = float(config.get("geo_corridor_dist_weight", 2.0))
        self.geo_path_goal_weight = float(config.get("geo_path_goal_weight", 0.2))
        self._agent_anchor_points = None

    def _build_backbone(self, trajectories, kg_data, vehicle_masks=None):
        outputs = super()._build_backbone(trajectories, kg_data, vehicle_masks=vehicle_masks)
        self._agent_anchor_points = kg_data.get("agent_anchor_points", None)
        return outputs

    @staticmethod
    def _safe_dist_scale(dist: torch.Tensor) -> torch.Tensor:
        return dist.detach().mean().clamp_min(1.0)

    def _build_geo_homotopy_corridor_sequences(
        self,
        graph_embeddings,
        local_contexts,
        top_goal_h,
        top_goals,
        kg_positions,
        kg_edge_index,
        agent_anchor_points=None,
    ):
        """
        Unroll H homotopy paths using geometry-aware start/goal node selection.

        Returns:
            corridor_seq:  (B,N,G,H,L,D)
            corridor_feat: (B,N,G,H,D)
            path_score:   (B,N,G,H)
        """
        b, n, g, d = top_goal_h.shape
        m = graph_embeddings.shape[1]
        h_cls = self.homotopy_classes
        l = self.corridor_seq_len
        device = graph_embeddings.device

        graph_norm = F.normalize(graph_embeddings, dim=-1)
        local_norm = F.normalize(local_contexts, dim=-1)
        goal_norm = F.normalize(top_goal_h, dim=-1)
        start_embed_score = torch.einsum("bnd,bmd->bnm", local_norm, graph_norm)
        goal_embed_score = torch.einsum("bngd,bmd->bngm", goal_norm, graph_norm)

        corridor_seq = torch.zeros((b, n, g, h_cls, l, d), device=device, dtype=graph_embeddings.dtype)
        path_score = torch.zeros((b, n, g, h_cls), device=device, dtype=graph_embeddings.dtype)

        for bi in range(b):
            edge_b = self._resolve_edge_index_for_batch(kg_edge_index, bi)
            pos_b = self._resolve_positions_for_batch(kg_positions, bi)
            if pos_b.dim() != 2:
                pos_b = pos_b.squeeze(0)
            pos_xy = pos_b[:, :2]
            adj = self._build_adjacency(edge_b, num_nodes=m, node_pos_b=pos_b)

            for ni in range(n):
                if agent_anchor_points is not None:
                    anchor = agent_anchor_points[bi, ni, :2].to(device=device, dtype=pos_xy.dtype)
                    local_nodes = pos_xy - anchor.unsqueeze(0)
                    start_dist = torch.norm(local_nodes, dim=-1)
                    start_score = (
                        self.geo_embed_weight * start_embed_score[bi, ni]
                        - self.geo_dist_weight * start_dist / self._safe_dist_scale(start_dist)
                    )
                    start = int(start_score.argmax().item())
                else:
                    local_nodes = pos_xy
                    start = int(start_embed_score[bi, ni].argmax().item())
                start = max(0, min(m - 1, start))

                for gi in range(g):
                    if agent_anchor_points is not None:
                        goal_local = top_goals[bi, ni, gi, :2].to(device=device, dtype=pos_xy.dtype)
                        goal_dist = torch.norm(local_nodes - goal_local.unsqueeze(0), dim=-1)
                        goal_score = (
                            self.geo_embed_weight * goal_embed_score[bi, ni, gi]
                            - self.geo_dist_weight * goal_dist / self._safe_dist_scale(goal_dist)
                        )
                        goal = int(goal_score.argmax().item())
                        selected_goal_dist = float(goal_dist[goal].detach().item())
                        goal_dist_scale = float(self._safe_dist_scale(goal_dist).detach().item())
                    else:
                        goal = int(goal_embed_score[bi, ni, gi].argmax().item())
                        selected_goal_dist = 0.0
                        goal_dist_scale = 1.0
                    goal = max(0, min(m - 1, goal))

                    node_penalty = [0.0 for _ in range(m)]
                    edge_penalty = {}

                    for hi in range(h_cls):
                        if any(adj[start]):
                            dist, prev = self._dijkstra_with_penalty(
                                start=start,
                                goal=goal,
                                adj=adj,
                                node_penalty=node_penalty,
                                edge_penalty=edge_penalty,
                            )
                            raw_path = self._reconstruct_path(prev, start, goal)
                            score = float(dist[goal]) if math.isfinite(dist[goal]) else 0.0
                        else:
                            raw_path = [start, goal] if start != goal else [start]
                            score = 0.0

                        raw_path = [max(0, min(m - 1, p)) for p in raw_path]
                        path = self._resample_indices(raw_path, l)
                        idx = torch.tensor(path, device=device, dtype=torch.long)
                        corridor_seq[bi, ni, gi, hi] = graph_embeddings[bi, idx]
                        path_score[bi, ni, gi, hi] = -float(score) - (
                            self.geo_path_goal_weight * selected_goal_dist / max(goal_dist_scale, 1.0)
                        )

                        # Penalize current path so the next pass searches a different class.
                        for u in raw_path[1:-1]:
                            node_penalty[u] += self.homotopy_node_penalty
                        for a, z in zip(raw_path[:-1], raw_path[1:]):
                            edge_penalty[(a, z)] = edge_penalty.get((a, z), 0.0) + self.homotopy_edge_penalty
                            edge_penalty[(z, a)] = edge_penalty.get((z, a), 0.0) + self.homotopy_edge_penalty

        path_score = path_score - path_score.max(dim=-1, keepdim=True).values
        corridor_feat = corridor_seq.mean(dim=4)
        return corridor_seq, corridor_feat, path_score

    def _route_decode_dual_ssm(
        self,
        fused_conditions,
        intent_priors,
        interaction_features,
        local_contexts,
        graph_embeddings,
        goal_h,
        goal_xy,
        goal_logits,
        kg_positions,
        kg_edge_index,
        num_points,
    ):
        """
        Same decoder as the homotopy Mamba variant, but corridors use geometry-aware
        start/goal KG nodes.
        """
        b, n, d = fused_conditions.shape
        h_cls = self.homotopy_classes
        k_target = self.route_modes
        t = int(num_points)

        goal_slots = max(1, min(goal_xy.shape[2], int(math.ceil(float(k_target) / float(h_cls)))))
        top = goal_logits.topk(goal_slots, dim=-1, largest=True, sorted=True)
        top_goal_indices = top.indices
        top_goal_logits = top.values

        gather_xy_idx = top_goal_indices.unsqueeze(-1).expand(b, n, goal_slots, 2)
        top_goals = goal_xy.gather(dim=2, index=gather_xy_idx)

        gather_h_idx = top_goal_indices.unsqueeze(-1).expand(b, n, goal_slots, d)
        top_goal_h = goal_h.gather(dim=2, index=gather_h_idx)

        corridor_seq, corridor_feat, path_score = self._build_geo_homotopy_corridor_sequences(
            graph_embeddings=graph_embeddings,
            local_contexts=local_contexts,
            top_goal_h=top_goal_h,
            top_goals=top_goals,
            kg_positions=kg_positions,
            kg_edge_index=kg_edge_index,
            agent_anchor_points=self._agent_anchor_points,
        )

        g = goal_slots
        h = h_cls
        c = g * h
        l = self.corridor_seq_len

        mode_goals_exp = top_goals.unsqueeze(3).expand(b, n, g, h, 2).reshape(b, n, c, 2)
        mode_seq_exp = corridor_seq.reshape(b, n, c, l, d)
        mode_corr_exp = corridor_feat.reshape(b, n, c, d)
        goal_logits_exp = top_goal_logits.unsqueeze(3).expand(b, n, g, h).reshape(b, n, c)
        path_score_exp = path_score.reshape(b, n, c)

        hid = torch.arange(h, device=top_goals.device, dtype=torch.long).view(1, 1, 1, h).expand(b, n, g, h)
        gid = torch.arange(g, device=top_goals.device, dtype=torch.long).view(1, 1, g, 1).expand(b, n, g, h)
        mode_hid_exp = hid.reshape(b, n, c)
        mode_gid_exp = gid.reshape(b, n, c)

        (
            sel_goal_logits,
            sel_path_score,
            mode_goals,
            mode_seq,
            mode_corr,
            mode_hid,
            mode_gid,
        ) = self._select_multipath_modes(
            goal_logits_exp=goal_logits_exp,
            path_score_exp=path_score_exp,
            mode_goals_exp=mode_goals_exp,
            mode_seq_exp=mode_seq_exp,
            mode_corr_exp=mode_corr_exp,
            mode_hid_exp=mode_hid_exp,
            mode_gid_exp=mode_gid_exp,
            k_target=k_target,
        )

        route_goal = self.goal_to_route(
            torch.cat([fused_conditions.unsqueeze(2).expand(-1, -1, k_target, -1), mode_goals], dim=-1)
        )
        route_goal = route_goal + self.homotopy_embed(mode_hid)
        map_pool = graph_embeddings.mean(dim=1).unsqueeze(1).unsqueeze(2).expand(b, n, k_target, d)
        base_seed = route_goal + 0.5 * intent_priors.unsqueeze(2) + 0.5 * interaction_features.unsqueeze(2)

        spatial_state, _ = self._spatial_scan(mode_seq, base_seed)
        route_base = self.route_seed_proj(
            torch.cat(
                [
                    route_goal,
                    spatial_state,
                    mode_corr,
                    intent_priors.unsqueeze(2).expand(-1, -1, k_target, -1),
                    interaction_features.unsqueeze(2).expand(-1, -1, k_target, -1)
                    + local_contexts.unsqueeze(2).expand(-1, -1, k_target, -1)
                    + map_pool,
                ],
                dim=-1,
            )
        )

        time = torch.linspace(0.0, 1.0, steps=t, device=route_base.device, dtype=route_base.dtype)
        time_feat = self.time_mlp(time.view(t, 1)).view(1, 1, 1, t, d)
        seq = route_base.unsqueeze(3) + time_feat + spatial_state.unsqueeze(3)
        seq = seq.reshape(b * n * k_target, t, d)

        for block in self.route_ssm:
            seq = block(seq)

        step_delta = self.step_delta_head(seq).view(b, n, k_target, t, 2)
        traj_modes = torch.cumsum(step_delta, dim=-2)
        traj_modes = self._apply_cv_residual(traj_modes)

        goal_expand = mode_goals.unsqueeze(-2)
        alpha = torch.linspace(0.0, 1.0, t, device=traj_modes.device, dtype=traj_modes.dtype).view(1, 1, 1, t, 1)
        traj_modes = traj_modes + alpha * (goal_expand - traj_modes[..., -1:, :])

        final_state = seq[:, -1].view(b, n, k_target, d)
        route_logits = self.route_score_head(final_state).squeeze(-1)
        corridor_score = F.cosine_similarity(final_state, spatial_state, dim=-1)
        mode_logits = (
            sel_goal_logits
            + route_logits
            + self.homotopy_score_weight * sel_path_score
            + self.corridor_score_weight * corridor_score
        )
        return traj_modes, mode_logits, mode_goals, spatial_state, final_state, mode_hid, mode_gid

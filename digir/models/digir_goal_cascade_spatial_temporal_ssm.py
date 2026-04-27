"""
DIGIR variant: Spatial-Temporal Dual-SSM Decoder.

Pipeline:
    history + KG -> dual-granularity encoder -> SSM Goal Decoder
    -> topology corridor sequence -> Spatial SSM over corridor nodes
    -> Temporal SSM over future steps -> multi-modal trajectories.

This file does NOT overwrite existing model scripts.
"""
from __future__ import annotations

import heapq
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.digir_goal_cascade_mamba_decoder import DIGIR as MambaDIGIR
from models.digir_goal_cascade_mamba_decoder import SelectiveSSMBlock


class DIGIR(MambaDIGIR):
    """
    Goal-cascade model where SSM scans both spatial topology and temporal rollout.
    """

    def __init__(self, config):
        super().__init__(config)

        d = self.d_model
        dropout = float(config.get("dropout", 0.1))
        expansion = int(config.get("mamba_expansion", 2))
        spatial_layers = int(config.get("spatial_ssm_layers", 2))
        temporal_layers = int(config.get("mamba_route_layers", 4))

        self.corridor_seq_len = max(2, int(config.get("corridor_seq_len", 16)))
        self.corridor_score_weight = float(config.get("corridor_score_weight", 0.2))
        self.lambda_corridor = float(config.get("lambda_corridor", 0.2))

        # Goal decoder is kept from MambaDIGIR, but this variant also exposes goal_h.
        self.spatial_ssm = nn.ModuleList(
            [SelectiveSSMBlock(d, expansion=expansion, dropout=dropout) for _ in range(spatial_layers)]
        )
        self.spatial_pos_mlp = nn.Sequential(
            nn.Linear(2, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )
        self.spatial_context_proj = nn.Linear(d * 2, d)

        # Replace temporal seed so it can consume spatial topology state.
        self.route_seed_proj = nn.Sequential(
            nn.Linear(d * 5, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.route_ssm = nn.ModuleList(
            [SelectiveSSMBlock(d, expansion=expansion, dropout=dropout) for _ in range(temporal_layers)]
        )

    def _build_backbone(self, trajectories, kg_data, vehicle_masks=None):
        outputs = super()._build_backbone(trajectories, kg_data, vehicle_masks=vehicle_masks)
        outputs["kg_positions"] = kg_data["positions"]
        outputs["kg_edge_index"] = kg_data["edge_index"]
        return outputs

    def _goal_decode_with_state(self, fused_conditions, local_contexts, graph_embeddings):
        """
        SSM goal decoder with hidden goal states.

        Returns:
            goal_h: (B,N,Q,D)
            goal_xy: (B,N,Q,2)
            goal_logits: (B,N,Q)
        """
        b, n, d = fused_conditions.shape
        q = self.goal_query_count

        map_pool = graph_embeddings.mean(dim=1).unsqueeze(1).expand(b, n, d)
        ctx = self.goal_context_proj(torch.cat([fused_conditions, local_contexts, map_pool], dim=-1))

        goal_seq = self.goal_queries.view(1, 1, q, d).expand(b, n, q, d)
        goal_seq = goal_seq + ctx.unsqueeze(2)
        goal_seq = goal_seq.reshape(b * n, q, d)

        for block in self.goal_ssm:
            goal_seq = block(goal_seq)

        goal_h = goal_seq.view(b, n, q, d)
        goal_xy = self.goal_pos_head(goal_h)
        goal_logits = self.goal_score_head(goal_h).squeeze(-1)
        return goal_h, goal_xy, goal_logits

    @staticmethod
    def _gather_feat(feat_modes, mode_idx):
        """
        feat_modes: (B,N,K,D), mode_idx: (B,N) -> (B,N,D)
        """
        b, n, _, d = feat_modes.shape
        idx = mode_idx.view(b, n, 1, 1).expand(b, n, 1, d)
        return feat_modes.gather(2, idx).squeeze(2)

    @staticmethod
    def _resolve_edge_index_for_batch(edge_index, b_idx: int):
        if edge_index.dim() == 3:
            return edge_index[b_idx]
        return edge_index

    @staticmethod
    def _resolve_positions_for_batch(positions, b_idx: int):
        if positions.dim() == 3:
            return positions[b_idx]
        return positions

    @staticmethod
    def _build_adjacency(edge_index_b: torch.Tensor, num_nodes: int, node_pos_b: torch.Tensor):
        ei = edge_index_b.long().cpu()
        pos = node_pos_b.detach().cpu()
        adj: List[List[Tuple[int, float]]] = [[] for _ in range(num_nodes)]
        if ei.numel() == 0:
            return adj

        src = ei[0].tolist()
        dst = ei[1].tolist()
        for u, v in zip(src, dst):
            if u < 0 or v < 0 or u >= num_nodes or v >= num_nodes:
                continue
            dx = float(pos[u, 0] - pos[v, 0])
            dy = float(pos[u, 1] - pos[v, 1])
            w = (dx * dx + dy * dy) ** 0.5 + 1e-3
            adj[u].append((v, w))
            adj[v].append((u, w))
        return adj

    @staticmethod
    def _dijkstra_tree(start: int, adj: List[List[Tuple[int, float]]]):
        n = len(adj)
        dist = [float("inf")] * n
        prev = [-1] * n
        dist[start] = 0.0
        pq: List[Tuple[float, int]] = [(0.0, start)]

        while pq:
            du, u = heapq.heappop(pq)
            if du > dist[u]:
                continue
            for v, w in adj[u]:
                nd = du + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        return dist, prev

    @staticmethod
    def _reconstruct_path(prev: List[int], start: int, goal: int):
        if start == goal:
            return [start]
        if goal < 0 or goal >= len(prev):
            return [start]
        if prev[goal] == -1:
            return [start, goal] if goal != start else [start]

        path = []
        cur = goal
        while cur != -1:
            path.append(cur)
            if cur == start:
                break
            cur = prev[cur]
        path = list(reversed(path))
        if not path or path[0] != start:
            return [start, goal]
        return path

    @staticmethod
    def _resample_indices(path: List[int], target_len: int):
        if len(path) <= 0:
            return [0 for _ in range(target_len)]
        if len(path) == 1:
            return [path[0] for _ in range(target_len)]
        if len(path) == target_len:
            return path

        idx = []
        for i in range(target_len):
            pos = round(i * (len(path) - 1) / max(target_len - 1, 1))
            idx.append(path[int(pos)])
        return idx

    def _build_corridor_sequences(
        self,
        graph_embeddings,
        local_contexts,
        top_goal_h,
        kg_positions,
        kg_edge_index,
    ):
        """
        Retrieve a topology path and return node-feature sequences.

        Returns:
            corridor_seq: (B,N,K,L,D)
            corridor_feat: (B,N,K,D)
        """
        b, n, k, d = top_goal_h.shape
        m = graph_embeddings.shape[1]
        l = self.corridor_seq_len
        device = graph_embeddings.device

        start_scores = torch.einsum("bnd,bmd->bnm", local_contexts, graph_embeddings)
        start_nodes = start_scores.argmax(dim=-1)

        goal_scores = torch.einsum("bnkd,bmd->bnkm", top_goal_h, graph_embeddings)
        goal_nodes = goal_scores.argmax(dim=-1)

        corridor_seq = torch.zeros((b, n, k, l, d), device=device, dtype=graph_embeddings.dtype)

        for bi in range(b):
            edge_b = self._resolve_edge_index_for_batch(kg_edge_index, bi)
            pos_b = self._resolve_positions_for_batch(kg_positions, bi)
            if pos_b.dim() != 2:
                pos_b = pos_b.squeeze(0)
            adj = self._build_adjacency(edge_b, num_nodes=m, node_pos_b=pos_b)

            for ni in range(n):
                s = int(start_nodes[bi, ni].item())
                s = max(0, min(m - 1, s))
                if any(adj[s]):
                    _, prev = self._dijkstra_tree(s, adj)
                else:
                    prev = [-1] * m

                for ki in range(k):
                    g = int(goal_nodes[bi, ni, ki].item())
                    g = max(0, min(m - 1, g))
                    path = self._reconstruct_path(prev, s, g)
                    path = [max(0, min(m - 1, p)) for p in self._resample_indices(path, l)]
                    idx = torch.tensor(path, device=device, dtype=torch.long)
                    corridor_seq[bi, ni, ki] = graph_embeddings[bi, idx]

        corridor_feat = corridor_seq.mean(dim=3)
        return corridor_seq, corridor_feat

    def _spatial_scan(self, corridor_seq, mode_seed):
        """
        corridor_seq: (B,N,K,L,D)
        mode_seed: (B,N,K,D)
        return:
            spatial_state: (B,N,K,D)
            spatial_seq: (B,N,K,L,D)
        """
        b, n, k, l, d = corridor_seq.shape
        x = self.spatial_context_proj(
            torch.cat([corridor_seq, mode_seed.unsqueeze(3).expand(-1, -1, -1, l, -1)], dim=-1)
        )
        x = x.reshape(b * n * k, l, d)
        for block in self.spatial_ssm:
            x = block(x)
        spatial_seq = x.view(b, n, k, l, d)
        spatial_state = spatial_seq[:, :, :, -1, :]
        return spatial_state, spatial_seq

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
        Spatial SSM over corridor nodes, then temporal SSM over future steps.
        """
        b, n, d = fused_conditions.shape
        k = min(self.route_modes, goal_xy.shape[2])
        t = int(num_points)

        top = goal_logits.topk(k, dim=-1, largest=True, sorted=True)
        top_goal_indices = top.indices
        top_goal_logits = top.values

        gather_xy_idx = top_goal_indices.unsqueeze(-1).expand(b, n, k, 2)
        top_goals = goal_xy.gather(dim=2, index=gather_xy_idx)

        gather_h_idx = top_goal_indices.unsqueeze(-1).expand(b, n, k, d)
        top_goal_h = goal_h.gather(dim=2, index=gather_h_idx)

        route_goal = self.goal_to_route(
            torch.cat([fused_conditions.unsqueeze(2).expand(-1, -1, k, -1), top_goals], dim=-1)
        )
        map_pool = graph_embeddings.mean(dim=1).unsqueeze(1).unsqueeze(2).expand(b, n, k, d)
        base_seed = route_goal + 0.5 * intent_priors.unsqueeze(2) + 0.5 * interaction_features.unsqueeze(2)

        corridor_seq, corridor_feat = self._build_corridor_sequences(
            graph_embeddings=graph_embeddings,
            local_contexts=local_contexts,
            top_goal_h=top_goal_h,
            kg_positions=kg_positions,
            kg_edge_index=kg_edge_index,
        )
        spatial_state, _ = self._spatial_scan(corridor_seq, base_seed)

        route_base = self.route_seed_proj(
            torch.cat(
                [
                    route_goal,
                    spatial_state,
                    corridor_feat,
                    intent_priors.unsqueeze(2).expand(-1, -1, k, -1),
                    interaction_features.unsqueeze(2).expand(-1, -1, k, -1)
                    + local_contexts.unsqueeze(2).expand(-1, -1, k, -1)
                    + map_pool,
                ],
                dim=-1,
            )
        )

        time = torch.linspace(0.0, 1.0, steps=t, device=route_base.device, dtype=route_base.dtype)
        time_feat = self.time_mlp(time.view(t, 1)).view(1, 1, 1, t, d)
        seq = route_base.unsqueeze(3) + time_feat + spatial_state.unsqueeze(3)
        seq = seq.reshape(b * n * k, t, d)

        for block in self.route_ssm:
            seq = block(seq)

        step_delta = self.step_delta_head(seq).view(b, n, k, t, 2)
        traj_modes = torch.cumsum(step_delta, dim=-2)

        goal_expand = top_goals.unsqueeze(-2)
        alpha = torch.linspace(
            0.0, 1.0, t, device=traj_modes.device, dtype=traj_modes.dtype
        ).view(1, 1, 1, t, 1)
        traj_modes = traj_modes + alpha * (goal_expand - traj_modes[..., -1:, :])

        final_state = seq[:, -1].view(b, n, k, d)
        route_logits = self.route_score_head(final_state).squeeze(-1)
        corridor_score = F.cosine_similarity(final_state, spatial_state, dim=-1)
        mode_logits = top_goal_logits + route_logits + self.corridor_score_weight * corridor_score
        return traj_modes, mode_logits, top_goals, spatial_state, final_state

    def forward(self, trajectories, kg_data, future_traj=None, mode="train", vehicle_masks=None):
        outputs = self._build_backbone(trajectories, kg_data, vehicle_masks=vehicle_masks)
        goal_h, goal_xy, goal_logits = self._goal_decode_with_state(
            outputs["fused_conditions"],
            outputs["local_contexts"],
            outputs["graph_embeddings"],
        )
        outputs["goal_xy"] = goal_xy
        outputs["goal_logits"] = goal_logits

        if mode == "train" and future_traj is not None:
            b, n, t, _ = future_traj.shape
            traj_modes, mode_logits, top_goals, spatial_state, final_state = self._route_decode_dual_ssm(
                outputs["fused_conditions"],
                outputs["intent_priors"],
                outputs["interaction_features"],
                outputs["local_contexts"],
                outputs["graph_embeddings"],
                goal_h,
                goal_xy,
                goal_logits,
                outputs["kg_positions"],
                outputs["kg_edge_index"],
                num_points=t,
            )
            outputs["mode_logits"] = mode_logits
            outputs["traj_modes"] = traj_modes
            outputs["spatial_state"] = spatial_state

            dist = torch.norm(traj_modes - future_traj.unsqueeze(2), dim=-1).mean(dim=-1)
            best_idx = dist.argmin(dim=-1)
            best_traj = self._gather_mode(traj_modes, best_idx)
            best_goal = self._gather_goal(top_goals, best_idx)
            outputs["traj_pred_train"] = best_traj
            outputs["best_mode_idx"] = best_idx

            if vehicle_masks is None:
                valid = torch.ones((b, n), dtype=torch.bool, device=future_traj.device)
            else:
                valid = vehicle_masks.bool()
            valid_flat = valid.view(-1)

            if valid.any():
                traj_loss = F.smooth_l1_loss(best_traj[valid], future_traj[valid], reduction="mean")
                goal_gt = future_traj[:, :, -1, :]
                goal_loss = F.smooth_l1_loss(best_goal[valid], goal_gt[valid], reduction="mean")
                mode_cls = F.cross_entropy(
                    mode_logits.view(b * n, -1)[valid_flat],
                    best_idx.view(-1)[valid_flat],
                    reduction="mean",
                )
                smooth_loss = self._temporal_smooth_loss(best_traj, valid)
                best_spatial = self._gather_feat(spatial_state, best_idx)
                best_final = self._gather_feat(final_state, best_idx)
                corr_align = 1.0 - F.cosine_similarity(best_final[valid], best_spatial[valid], dim=-1).mean()
            else:
                z = torch.tensor(0.0, device=future_traj.device, dtype=future_traj.dtype)
                traj_loss = z
                goal_loss = z
                mode_cls = z
                smooth_loss = z
                corr_align = z

            total = (
                traj_loss
                + self.lambda_goal * goal_loss
                + self.lambda_mode_cls * mode_cls
                + self.lambda_temporal_smooth * smooth_loss
                + self.lambda_corridor * corr_align
            )
            outputs["diffusion_loss"] = total
            outputs["loss_route_head"] = traj_loss
            outputs["loss_goal_head"] = goal_loss
            outputs["loss_mode_cls"] = mode_cls
            outputs["loss_temporal_smooth"] = smooth_loss
            outputs["loss_corridor_align"] = corr_align

        return outputs

    def generate(
        self,
        trajectories,
        kg_data,
        num_points=12,
        num_samples=20,
        sampling="ddim",
        step=20,
        bestof=True,
        vehicle_masks=None,
    ):
        del sampling, step

        outputs = self._build_backbone(trajectories, kg_data, vehicle_masks=vehicle_masks)
        goal_h, goal_xy, goal_logits = self._goal_decode_with_state(
            outputs["fused_conditions"],
            outputs["local_contexts"],
            outputs["graph_embeddings"],
        )
        traj_modes, mode_logits, _, _, _ = self._route_decode_dual_ssm(
            outputs["fused_conditions"],
            outputs["intent_priors"],
            outputs["interaction_features"],
            outputs["local_contexts"],
            outputs["graph_embeddings"],
            goal_h,
            goal_xy,
            goal_logits,
            outputs["kg_positions"],
            outputs["kg_edge_index"],
            num_points=int(num_points),
        )

        if int(num_samples) <= 1:
            idx = self._sample_mode_index(mode_logits, bestof=bestof)
            return self._gather_mode(traj_modes, idx)

        k = traj_modes.shape[2]
        s = int(num_samples)
        preds = []

        if bestof:
            top = min(s, k)
            top_idx = mode_logits.argsort(dim=-1, descending=True)[:, :, :top]
            b, n, _, t, c = traj_modes.shape
            gather_idx = top_idx.unsqueeze(-1).unsqueeze(-1).expand(b, n, top, t, c)
            top_traj = traj_modes.gather(2, gather_idx).permute(2, 0, 1, 3, 4).contiguous()
            preds.append(top_traj)
            if top < s:
                for _ in range(s - top):
                    idx = self._sample_mode_index(mode_logits, bestof=False)
                    preds.append(self._gather_mode(traj_modes, idx).unsqueeze(0))
            return torch.cat(preds, dim=0)

        for _ in range(s):
            idx = self._sample_mode_index(mode_logits, bestof=False)
            preds.append(self._gather_mode(traj_modes, idx))
        return torch.stack(preds, dim=0)


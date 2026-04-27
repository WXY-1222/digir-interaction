"""
DIGIR variant: Homotopy Multi-Path Unrolling + Spatial-Temporal Dual-SSM.

Pipeline:
    history + KG -> dual-granularity encoder -> SSM Goal Decoder
    -> homotopy multi-path unrolling -> Spatial SSM over each 1D path
    -> Temporal SSM over future steps -> multi-modal trajectories.

Instead of forcing a branching road graph into one sequence, this variant
unrolls the graph into multiple topology-distinct 1D corridor sequences.
"""
from __future__ import annotations

import heapq
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.digir_goal_cascade_spatial_temporal_ssm import DIGIR as SpatialTemporalSSM


class DIGIR(SpatialTemporalSSM):
    """
    Spatial-temporal SSM with homotopy-class multi-path unrolling.
    """

    def __init__(self, config):
        super().__init__(config)

        self.homotopy_classes = max(2, int(config.get("homotopy_classes", 3)))
        self.homotopy_node_penalty = float(config.get("homotopy_node_penalty", 3.0))
        self.homotopy_edge_penalty = float(config.get("homotopy_edge_penalty", 4.0))
        self.homotopy_score_weight = float(config.get("homotopy_score_weight", 0.2))
        self.lambda_homotopy = float(config.get("lambda_homotopy", 0.1))
        self.homotopy_margin = float(config.get("homotopy_margin", 0.6))

        self.homotopy_embed = nn.Embedding(self.homotopy_classes, self.d_model)

    @staticmethod
    def _dijkstra_with_penalty(
        start: int,
        goal: int,
        adj: List[List[Tuple[int, float]]],
        node_penalty: List[float],
        edge_penalty: dict,
    ):
        n = len(adj)
        dist = [float("inf")] * n
        prev = [-1] * n
        dist[start] = 0.0
        pq: List[Tuple[float, int]] = [(0.0, start)]

        while pq:
            du, u = heapq.heappop(pq)
            if du > dist[u]:
                continue
            if u == goal:
                break
            for v, w in adj[u]:
                penalty = float(node_penalty[v]) + float(edge_penalty.get((u, v), 0.0))
                nd = du + float(w) + penalty
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        return dist, prev

    def _build_homotopy_corridor_sequences(
        self,
        graph_embeddings,
        local_contexts,
        top_goal_h,
        kg_positions,
        kg_edge_index,
    ):
        """
        Unroll each goal-conditioned corridor graph into H independent 1D paths.

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

        start_scores = torch.einsum("bnd,bmd->bnm", local_contexts, graph_embeddings)
        start_nodes = start_scores.argmax(dim=-1)

        goal_scores = torch.einsum("bngd,bmd->bngm", top_goal_h, graph_embeddings)
        goal_nodes = goal_scores.argmax(dim=-1)

        corridor_seq = torch.zeros((b, n, g, h_cls, l, d), device=device, dtype=graph_embeddings.dtype)
        path_score = torch.zeros((b, n, g, h_cls), device=device, dtype=graph_embeddings.dtype)

        for bi in range(b):
            edge_b = self._resolve_edge_index_for_batch(kg_edge_index, bi)
            pos_b = self._resolve_positions_for_batch(kg_positions, bi)
            if pos_b.dim() != 2:
                pos_b = pos_b.squeeze(0)
            adj = self._build_adjacency(edge_b, num_nodes=m, node_pos_b=pos_b)

            for ni in range(n):
                start = int(start_nodes[bi, ni].item())
                start = max(0, min(m - 1, start))

                for gi in range(g):
                    goal = int(goal_nodes[bi, ni, gi].item())
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
                        path_score[bi, ni, gi, hi] = -float(score)

                        # Penalize current path so the next pass searches a different class.
                        for u in raw_path[1:-1]:
                            node_penalty[u] += self.homotopy_node_penalty
                        for a, z in zip(raw_path[:-1], raw_path[1:]):
                            edge_penalty[(a, z)] = edge_penalty.get((a, z), 0.0) + self.homotopy_edge_penalty
                            edge_penalty[(z, a)] = edge_penalty.get((z, a), 0.0) + self.homotopy_edge_penalty

        path_score = path_score - path_score.max(dim=-1, keepdim=True).values
        corridor_feat = corridor_seq.mean(dim=4)
        return corridor_seq, corridor_feat, path_score

    @staticmethod
    def _select_multipath_modes(
        goal_logits_exp,
        path_score_exp,
        mode_goals_exp,
        mode_seq_exp,
        mode_corr_exp,
        mode_hid_exp,
        mode_gid_exp,
        k_target: int,
    ):
        b, n, c = goal_logits_exp.shape
        if c == k_target:
            return (
                goal_logits_exp,
                path_score_exp,
                mode_goals_exp,
                mode_seq_exp,
                mode_corr_exp,
                mode_hid_exp,
                mode_gid_exp,
            )

        if c > k_target:
            combo = goal_logits_exp + path_score_exp
            top = combo.topk(k_target, dim=-1, largest=True, sorted=True)
            idx = top.indices

            idx2 = idx.unsqueeze(-1).expand(b, n, k_target, 2)
            idxd = idx.unsqueeze(-1).expand(b, n, k_target, mode_corr_exp.shape[-1])
            idxseq = idx.unsqueeze(-1).unsqueeze(-1).expand(
                b, n, k_target, mode_seq_exp.shape[-2], mode_seq_exp.shape[-1]
            )
            return (
                goal_logits_exp.gather(2, idx),
                path_score_exp.gather(2, idx),
                mode_goals_exp.gather(2, idx2),
                mode_seq_exp.gather(2, idxseq),
                mode_corr_exp.gather(2, idxd),
                mode_hid_exp.gather(2, idx),
                mode_gid_exp.gather(2, idx),
            )

        rep = int(math.ceil(float(k_target) / float(c)))
        return (
            goal_logits_exp.repeat(1, 1, rep)[:, :, :k_target],
            path_score_exp.repeat(1, 1, rep)[:, :, :k_target],
            mode_goals_exp.repeat(1, 1, rep, 1)[:, :, :k_target, :],
            mode_seq_exp.repeat(1, 1, rep, 1, 1)[:, :, :k_target, :, :],
            mode_corr_exp.repeat(1, 1, rep, 1)[:, :, :k_target, :],
            mode_hid_exp.repeat(1, 1, rep)[:, :, :k_target],
            mode_gid_exp.repeat(1, 1, rep)[:, :, :k_target],
        )

    def _homotopy_bifurcation_loss(self, traj_modes, mode_hid, mode_gid, vehicle_masks):
        """
        Encourage same-goal / different-homotopy trajectories to separate.
        """
        b, n, k, _, _ = traj_modes.shape
        if k < 2:
            return torch.tensor(0.0, device=traj_modes.device, dtype=traj_modes.dtype)

        traj_i = traj_modes.unsqueeze(3)
        traj_j = traj_modes.unsqueeze(2)
        pair_dist = torch.norm(traj_i - traj_j, dim=-1).mean(dim=-1)

        same_goal = mode_gid.unsqueeze(3) == mode_gid.unsqueeze(2)
        diff_h = mode_hid.unsqueeze(3) != mode_hid.unsqueeze(2)
        upper = torch.triu(
            torch.ones((k, k), device=traj_modes.device, dtype=torch.bool),
            diagonal=1,
        ).view(1, 1, k, k)
        valid_agent = vehicle_masks.bool().unsqueeze(-1).unsqueeze(-1)
        mask = same_goal & diff_h & upper & valid_agent
        if not mask.any():
            return torch.tensor(0.0, device=traj_modes.device, dtype=traj_modes.dtype)
        return torch.relu(float(self.homotopy_margin) - pair_dist)[mask].mean()

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
        Homotopy paths are unrolled into independent 1D sequences for Spatial SSM.
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

        corridor_seq, corridor_feat, path_score = self._build_homotopy_corridor_sequences(
            graph_embeddings=graph_embeddings,
            local_contexts=local_contexts,
            top_goal_h=top_goal_h,
            kg_positions=kg_positions,
            kg_edge_index=kg_edge_index,
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

        goal_expand = mode_goals.unsqueeze(-2)
        alpha = torch.linspace(
            0.0, 1.0, t, device=traj_modes.device, dtype=traj_modes.dtype
        ).view(1, 1, 1, t, 1)
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
            (
                traj_modes,
                mode_logits,
                mode_goals,
                spatial_state,
                final_state,
                mode_hid,
                mode_gid,
            ) = self._route_decode_dual_ssm(
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
            outputs["mode_homotopy_ids"] = mode_hid
            outputs["mode_goal_slots"] = mode_gid

            dist = torch.norm(traj_modes - future_traj.unsqueeze(2), dim=-1).mean(dim=-1)
            best_idx = dist.argmin(dim=-1)
            best_traj = self._gather_mode(traj_modes, best_idx)
            best_goal = self._gather_goal(mode_goals, best_idx)
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
                homotopy_loss = self._homotopy_bifurcation_loss(traj_modes, mode_hid, mode_gid, valid)
            else:
                z = torch.tensor(0.0, device=future_traj.device, dtype=future_traj.dtype)
                traj_loss = z
                goal_loss = z
                mode_cls = z
                smooth_loss = z
                corr_align = z
                homotopy_loss = z

            total = (
                traj_loss
                + self.lambda_goal * goal_loss
                + self.lambda_mode_cls * mode_cls
                + self.lambda_temporal_smooth * smooth_loss
                + self.lambda_corridor * corr_align
                + self.lambda_homotopy * homotopy_loss
            )
            outputs["diffusion_loss"] = total
            outputs["loss_route_head"] = traj_loss
            outputs["loss_goal_head"] = goal_loss
            outputs["loss_mode_cls"] = mode_cls
            outputs["loss_temporal_smooth"] = smooth_loss
            outputs["loss_corridor_align"] = corr_align
            outputs["loss_homotopy"] = homotopy_loss

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
        traj_modes, mode_logits, _, _, _, _, _ = self._route_decode_dual_ssm(
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


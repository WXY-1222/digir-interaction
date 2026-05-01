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


class DirectedPairInteractionHead(nn.Module):
    """
    Predict directed pairwise interaction order for each agent pair.

    Classes for pair (i, j):
        0: no close interaction
        1: agent i reaches the conflict region before agent j
        2: agent j reaches the conflict region before agent i
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 3 + 3, d_model),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 3),
        )

    def forward(self, agent_features, agent_positions, vehicle_masks=None):
        b, n, d = agent_features.shape
        hi = agent_features.unsqueeze(2).expand(b, n, n, d)
        hj = agent_features.unsqueeze(1).expand(b, n, n, d)
        rel = agent_positions.unsqueeze(2) - agent_positions.unsqueeze(1)
        dist = torch.norm(rel, dim=-1, keepdim=True)
        feat = torch.cat([hi, hj, hi - hj, rel, dist], dim=-1)
        logits = self.mlp(feat)

        if vehicle_masks is not None:
            valid = vehicle_masks.bool()
            pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
            logits = logits.masked_fill(~pair_valid.unsqueeze(-1), 0.0)
        return logits


class InteractionSpatialModulator(nn.Module):
    """
    Inject social interaction context into each homotopy corridor state.

    The modulator is intentionally light: pairwise interaction predictions are
    first pooled into a per-agent social context, then this context FiLM-modulates
    every route-mode spatial state before the Temporal SSM rollout.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, scale: float = 0.2):
        super().__init__()
        self.scale = float(scale)
        self.out = nn.Linear(d_model, d_model * 2 + 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        with torch.no_grad():
            self.out.bias[-1].fill_(-4.0)
        self.film = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            self.out,
        )

    def forward(self, spatial_state, route_goal, social_context):
        b, n, k, d = spatial_state.shape
        social = social_context.unsqueeze(2).expand(b, n, k, d)
        params = self.film(torch.cat([spatial_state, route_goal, social], dim=-1))
        gamma, beta, gate_logits = torch.split(params, [d, d, 1], dim=-1)
        gate = torch.sigmoid(gate_logits)
        delta = torch.tanh(gamma) * spatial_state + beta
        return spatial_state + self.scale * gate * delta


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
        self.lambda_interaction_graph = float(config.get("lambda_interaction_graph", 0.0))
        self.interaction_dist_threshold = float(config.get("interaction_dist_threshold", 2.5))
        self.use_cv_residual = bool(config.get("use_cv_residual", True))
        self.cv_residual_weight = float(config.get("cv_residual_weight", 1.0))
        self.use_interaction_spatial_modulation = bool(
            config.get("use_interaction_spatial_modulation", True)
        )
        self.interaction_modulation_scale = float(config.get("interaction_modulation_scale", 0.2))
        self._cv_prior = None
        self._social_context = None

        self.homotopy_embed = nn.Embedding(self.homotopy_classes, self.d_model)
        self.pair_interaction_head = DirectedPairInteractionHead(
            d_model=self.d_model,
            dropout=float(config.get("dropout", 0.1)),
        )
        self.interaction_spatial_modulator = InteractionSpatialModulator(
            d_model=self.d_model,
            dropout=float(config.get("dropout", 0.1)),
            scale=self.interaction_modulation_scale,
        )

    def _current_pair_positions(self, trajectories, kg_data):
        anchor = kg_data.get("agent_anchor_points", None) if isinstance(kg_data, dict) else None
        if anchor is not None:
            return anchor[..., :2].to(device=trajectories.device, dtype=trajectories.dtype)
        return trajectories[:, :, -1, :2]

    def _build_constant_velocity_prior(self, trajectories, num_points: int):
        if not self.use_cv_residual:
            return None
        b, n, hist, _ = trajectories.shape
        t = int(num_points)
        if hist < 2 or t <= 0:
            return torch.zeros((b, n, max(t, 0), 2), device=trajectories.device, dtype=trajectories.dtype)

        velocity = trajectories[:, :, -1, :2] - trajectories[:, :, -2, :2]
        steps = torch.arange(1, t + 1, device=trajectories.device, dtype=trajectories.dtype)
        return steps.view(1, 1, t, 1) * velocity.unsqueeze(2)

    def _apply_cv_residual(self, traj_modes):
        cv_prior = getattr(self, "_cv_prior", None)
        if cv_prior is None or not self.use_cv_residual:
            return traj_modes
        if cv_prior.shape[:2] != traj_modes.shape[:2] or cv_prior.shape[2] != traj_modes.shape[3]:
            return traj_modes
        return traj_modes + self.cv_residual_weight * cv_prior.unsqueeze(2).to(
            device=traj_modes.device,
            dtype=traj_modes.dtype,
        )

    def _build_social_context(self, interaction_features, pair_logits, vehicle_masks=None):
        probs = F.softmax(pair_logits, dim=-1)
        interact_prob = 1.0 - probs[..., 0]
        signed_yield = probs[..., 2] - probs[..., 1]
        weights = interact_prob * (1.0 + 0.5 * signed_yield).clamp(min=0.1, max=1.9)

        b, n, d = interaction_features.shape
        eye = torch.eye(n, dtype=torch.bool, device=interaction_features.device).view(1, n, n)
        weights = weights.masked_fill(eye, 0.0)

        if vehicle_masks is not None:
            valid = vehicle_masks.bool()
            pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
            weights = weights.masked_fill(~pair_valid, 0.0)

        neigh_feat = interaction_features.unsqueeze(1).expand(b, n, n, d)
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return (weights.unsqueeze(-1) * neigh_feat).sum(dim=2) / denom

    def _apply_interaction_spatial_modulation(self, spatial_state, route_goal):
        if not self.use_interaction_spatial_modulation:
            return spatial_state
        social_context = getattr(self, "_social_context", None)
        if social_context is None:
            return spatial_state
        if social_context.shape[:2] != spatial_state.shape[:2]:
            return spatial_state
        return self.interaction_spatial_modulator(
            spatial_state,
            route_goal,
            social_context.to(device=spatial_state.device, dtype=spatial_state.dtype),
        )

    def _build_pair_interaction_targets(self, future_traj, current_positions, vehicle_masks=None):
        b, n, t, _ = future_traj.shape
        device = future_traj.device
        dtype = future_traj.dtype
        if self.config.get("coord_frame", "global") == "per_agent":
            future_global = future_traj + current_positions.unsqueeze(2).to(device=device, dtype=dtype)
        else:
            future_global = future_traj

        fi = future_global.unsqueeze(2).unsqueeze(4)
        fj = future_global.unsqueeze(1).unsqueeze(3)
        dist = torch.norm(fi - fj, dim=-1)
        flat_dist = dist.view(b, n, n, t * t)
        min_dist, flat_idx = flat_dist.min(dim=-1)
        ti = torch.div(flat_idx, t, rounding_mode="floor")
        tj = flat_idx.remainder(t)

        labels = torch.zeros((b, n, n), device=device, dtype=torch.long)
        close = min_dist < float(self.interaction_dist_threshold)
        labels[(close) & (ti < tj)] = 1
        labels[(close) & (ti > tj)] = 2

        if t > 1:
            speed = torch.norm(future_global[:, :, 1:] - future_global[:, :, :-1], dim=-1).mean(dim=-1)
            faster_i = speed.unsqueeze(2) >= speed.unsqueeze(1)
            labels[(close) & (ti == tj) & faster_i] = 1
            labels[(close) & (ti == tj) & (~faster_i)] = 2

        valid = torch.ones((b, n), dtype=torch.bool, device=device) if vehicle_masks is None else vehicle_masks.bool()
        pair_mask = valid.unsqueeze(2) & valid.unsqueeze(1)
        diag = torch.eye(n, dtype=torch.bool, device=device).view(1, n, n)
        pair_mask = pair_mask & (~diag)
        return labels, pair_mask

    def _pair_interaction_loss(self, pair_logits, future_traj, current_positions, vehicle_masks=None):
        labels, pair_mask = self._build_pair_interaction_targets(
            future_traj=future_traj,
            current_positions=current_positions,
            vehicle_masks=vehicle_masks,
        )
        if not pair_mask.any():
            return torch.tensor(0.0, device=future_traj.device, dtype=future_traj.dtype), labels, pair_mask
        loss = F.cross_entropy(pair_logits[pair_mask], labels[pair_mask], reduction="mean")
        return loss, labels, pair_mask

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
        spatial_state = self._apply_interaction_spatial_modulation(spatial_state, route_goal)
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
        current_positions = self._current_pair_positions(trajectories, kg_data)
        pair_logits = self.pair_interaction_head(
            outputs["interaction_features"],
            current_positions,
            vehicle_masks=vehicle_masks,
        )
        outputs["pair_interaction_logits"] = pair_logits
        self._social_context = self._build_social_context(
            outputs["interaction_features"],
            pair_logits,
            vehicle_masks=vehicle_masks,
        )
        outputs["social_context"] = self._social_context
        goal_h, goal_xy, goal_logits = self._goal_decode_with_state(
            outputs["fused_conditions"],
            outputs["local_contexts"],
            outputs["graph_embeddings"],
        )
        outputs["goal_xy"] = goal_xy
        outputs["goal_logits"] = goal_logits

        if mode == "train" and future_traj is not None:
            b, n, t, _ = future_traj.shape
            self._cv_prior = self._build_constant_velocity_prior(trajectories, num_points=t)
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

            if self.lambda_interaction_graph > 0.0:
                interaction_graph_loss, pair_labels, pair_mask = self._pair_interaction_loss(
                    pair_logits=pair_logits,
                    future_traj=future_traj,
                    current_positions=current_positions,
                    vehicle_masks=vehicle_masks,
                )
                outputs["pair_interaction_labels"] = pair_labels
                outputs["pair_interaction_mask"] = pair_mask
            else:
                interaction_graph_loss = torch.tensor(0.0, device=future_traj.device, dtype=future_traj.dtype)

            total = (
                traj_loss
                + self.lambda_goal * goal_loss
                + self.lambda_mode_cls * mode_cls
                + self.lambda_temporal_smooth * smooth_loss
                + self.lambda_corridor * corr_align
                + self.lambda_homotopy * homotopy_loss
                + self.lambda_interaction_graph * interaction_graph_loss
            )
            outputs["diffusion_loss"] = total
            outputs["loss_route_head"] = traj_loss
            outputs["loss_goal_head"] = goal_loss
            outputs["loss_mode_cls"] = mode_cls
            outputs["loss_temporal_smooth"] = smooth_loss
            outputs["loss_corridor_align"] = corr_align
            outputs["loss_homotopy"] = homotopy_loss
            outputs["loss_interaction_graph"] = interaction_graph_loss

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
        self._cv_prior = self._build_constant_velocity_prior(trajectories, num_points=int(num_points))
        current_positions = self._current_pair_positions(trajectories, kg_data)
        pair_logits = self.pair_interaction_head(
            outputs["interaction_features"],
            current_positions,
            vehicle_masks=vehicle_masks,
        )
        self._social_context = self._build_social_context(
            outputs["interaction_features"],
            pair_logits,
            vehicle_masks=vehicle_masks,
        )
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

    def compute_losses(self, outputs, future_traj, intent_labels, vehicle_masks=None):
        losses, total_loss = super().compute_losses(
            outputs,
            future_traj,
            intent_labels,
            vehicle_masks=vehicle_masks,
        )
        for key in (
            "loss_route_head",
            "loss_goal_head",
            "loss_mode_cls",
            "loss_temporal_smooth",
            "loss_corridor_align",
            "loss_homotopy",
            "loss_interaction_graph",
        ):
            if key in outputs:
                losses[key] = float(outputs[key].detach().item())
        losses["lambda_interaction_graph"] = float(self.lambda_interaction_graph)
        losses["use_cv_residual"] = float(self.use_cv_residual)
        losses["use_interaction_spatial_modulation"] = float(self.use_interaction_spatial_modulation)
        return losses, total_loss

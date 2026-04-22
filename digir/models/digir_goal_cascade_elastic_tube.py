"""
DIGIR variant: Goal-to-Trajectory Cascaded Decoding with Dynamic
Spatio-Temporal Elastic Tubes.

Upgrade over topology-corridor guided routing:
- Keep static topology corridor retrieval (map-aware lane sequence).
- Lift 2D corridor to a 3D (x, y, t) tube by building time-indexed centerline.
- Predict dynamic risk and apply time-local "pinch" on tube radius.
- Constrain routing outputs inside the elastic tube with differentiable penalties.

This file does NOT overwrite existing model scripts.
"""
from __future__ import annotations

import heapq
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.digir import DIGIR as BaseDIGIR


class DIGIR(BaseDIGIR):
    """
    Cascaded decoder + topology corridor + dynamic elastic tube.
    """

    def __init__(self, config):
        super().__init__(config)

        self.goal_query_count = int(config.get("goal_query_count", 32))
        self.route_modes = int(config.get("route_modes", 6))
        self.route_temperature = float(config.get("route_temperature", 1.0))

        # Corridor retrieval / static constraint knobs.
        self.corridor_expand_hops = int(config.get("corridor_expand_hops", 1))
        self.corridor_score_weight = float(config.get("corridor_score_weight", 0.2))
        self.lambda_corridor = float(config.get("lambda_corridor", 0.2))

        # Dynamic elastic tube knobs.
        self.tube_base_radius = float(config.get("tube_base_radius", 2.5))         # meters
        self.tube_min_radius = float(config.get("tube_min_radius", 0.6))           # meters
        self.tube_pinch_strength = float(config.get("tube_pinch_strength", 0.7))   # [0,1]
        self.tube_score_weight = float(config.get("tube_score_weight", 0.25))
        self.lambda_tube = float(config.get("lambda_tube", 0.3))
        self.lambda_tube_smooth = float(config.get("lambda_tube_smooth", 0.05))

        self.lambda_goal = float(config.get("lambda_goal", 0.5))
        self.lambda_mode_cls = float(config.get("lambda_mode_cls", 0.3))

        d = self.d_model
        nhead = int(config.get("num_heads", 4))
        dropout = float(config.get("dropout", 0.1))
        goal_layers = int(config.get("goal_decoder_layers", 3))
        route_layers = int(config.get("route_decoder_layers", 3))
        ffn_dim = int(config.get("cascade_ffn_dim", d * 4))

        # -------- Stage 1: Goal Decoder --------
        self.goal_queries = nn.Parameter(torch.randn(self.goal_query_count, d) * 0.02)
        self.goal_query_proj = nn.Linear(d, d)

        goal_layer = nn.TransformerDecoderLayer(
            d_model=d,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.goal_decoder = nn.TransformerDecoder(goal_layer, num_layers=goal_layers)
        self.goal_pos_head = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 2),
        )
        self.goal_score_head = nn.Linear(d, 1)

        # -------- Stage 2: Routing Decoder --------
        self.goal_to_route = nn.Sequential(
            nn.Linear(d + 2, d),
            nn.ReLU(),
            nn.Linear(d, d),
        )
        self.corridor_proj = nn.Linear(d, d)

        route_layer = nn.TransformerDecoderLayer(
            d_model=d,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.route_decoder = nn.TransformerDecoder(route_layer, num_layers=route_layers)
        self.route_traj_head = nn.Linear(d, self.prediction_horizon * 2)
        self.route_score_head = nn.Linear(d, 1)

        # Dynamic tube parameter heads.
        self.dynamic_risk_head = nn.Sequential(
            nn.Linear(d * 3, d),
            nn.ReLU(),
            nn.Linear(d, 1),
        )
        self.pinch_param_head = nn.Linear(d, 2)  # center, width

        # This variant does not use diffusion.
        self.diffusion = None

    @staticmethod
    def _gather_mode(traj_modes, mode_idx):
        """
        traj_modes: (B,N,K,T,2), mode_idx: (B,N) -> (B,N,T,2)
        """
        b, n, _, t, c = traj_modes.shape
        idx = mode_idx.view(b, n, 1, 1, 1).expand(b, n, 1, t, c)
        return traj_modes.gather(2, idx).squeeze(2)

    @staticmethod
    def _gather_goal(goal_modes, mode_idx):
        """
        goal_modes: (B,N,K,2), mode_idx: (B,N) -> (B,N,2)
        """
        b, n, _, c = goal_modes.shape
        idx = mode_idx.view(b, n, 1, 1).expand(b, n, 1, c)
        return goal_modes.gather(2, idx).squeeze(2)

    @staticmethod
    def _gather_feat(feat_modes, mode_idx):
        """
        feat_modes: (B,N,K,D), mode_idx: (B,N) -> (B,N,D)
        """
        b, n, _, d = feat_modes.shape
        idx = mode_idx.view(b, n, 1, 1).expand(b, n, 1, d)
        return feat_modes.gather(2, idx).squeeze(2)

    @staticmethod
    def _gather_time2d(feat_modes, mode_idx):
        """
        feat_modes: (B,N,K,T,2), mode_idx: (B,N) -> (B,N,T,2)
        """
        b, n, _, t, c = feat_modes.shape
        idx = mode_idx.view(b, n, 1, 1, 1).expand(b, n, 1, t, c)
        return feat_modes.gather(2, idx).squeeze(2)

    @staticmethod
    def _gather_time1d(feat_modes, mode_idx):
        """
        feat_modes: (B,N,K,T), mode_idx: (B,N) -> (B,N,T)
        """
        b, n, _, t = feat_modes.shape
        idx = mode_idx.view(b, n, 1, 1).expand(b, n, 1, t)
        return feat_modes.gather(2, idx).squeeze(2)

    def _sample_mode_index(self, mode_logits, bestof=False):
        if bestof:
            return mode_logits.argmax(dim=-1)

        k = mode_logits.shape[-1]
        temp = max(self.route_temperature, 1e-4)
        probs = torch.softmax(mode_logits / temp, dim=-1)
        probs = torch.nan_to_num(
            probs,
            nan=1.0 / float(k),
            posinf=1.0 / float(k),
            neginf=1.0 / float(k),
        )
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        idx = torch.multinomial(probs.view(-1, k), num_samples=1).view(mode_logits.shape[0], mode_logits.shape[1])
        return idx

    def _resize_traj(self, traj_modes, num_points):
        t0 = traj_modes.shape[-2]
        if t0 == num_points:
            return traj_modes

        b, n, k, _, c = traj_modes.shape
        x = traj_modes.permute(0, 1, 2, 4, 3).reshape(b * n * k, c, t0)
        x = F.interpolate(x, size=int(num_points), mode="linear", align_corners=True)
        x = x.view(b, n, k, c, int(num_points)).permute(0, 1, 2, 4, 3)
        return x

    def _build_backbone(self, trajectories, kg_data, vehicle_masks=None):
        """
        DIGIR dual-granularity backbone with explicit graph embeddings.
        """
        motion_summaries = self.traj_encoder(trajectories)
        if vehicle_masks is not None:
            motion_summaries = motion_summaries * vehicle_masks.unsqueeze(-1).float()

        graph_embeddings = self.graph_encoder(
            kg_data["facility_types"],
            kg_data["positions"],
            kg_data["edge_index"],
            kg_data.get("edge_types"),
        )  # (B,M,D)

        if self.ablate_cross_attn:
            local_contexts = motion_summaries
        else:
            local_contexts = self.local_context_extractor(
                motion_summaries, graph_embeddings, vehicle_mask=vehicle_masks
            )

        scene_intent, local_contexts = self.scene_transformer(
            local_contexts,
            vehicle_mask=vehicle_masks,
        )
        intent_priors, intent_logits = self.cross_granularity_mapping(
            scene_intent,
            local_contexts,
            vehicle_masks=vehicle_masks,
        )
        fused_conditions, interaction_features, gate_weights = self.agent_level_modeling(
            motion_summaries,
            intent_priors,
            vehicle_masks=vehicle_masks,
        )

        return {
            "scene_intent": scene_intent,
            "local_contexts": local_contexts,
            "motion_summaries": motion_summaries,
            "intent_priors": intent_priors,
            "intent_logits": intent_logits,
            "interaction_features": interaction_features,
            "fused_conditions": fused_conditions,
            "gate_weights": gate_weights,
            "graph_embeddings": graph_embeddings,
            "kg_positions": kg_data["positions"],
            "kg_edge_index": kg_data["edge_index"],
        }

    def _goal_decode(self, fused_conditions, local_contexts, graph_embeddings):
        """
        Returns:
            goal_h: (B,N,Q,D)
            goal_xy: (B,N,Q,2)
            goal_logits: (B,N,Q)
        """
        b, n, d = fused_conditions.shape
        q = self.goal_query_count
        m = graph_embeddings.shape[1]

        base_q = self.goal_queries.view(1, 1, q, d).expand(b, n, q, d)
        goal_query = self.goal_query_proj(base_q + fused_conditions.unsqueeze(2) + 0.5 * local_contexts.unsqueeze(2))

        tgt = goal_query.reshape(b * n, q, d)
        mem_map = graph_embeddings.unsqueeze(1).expand(b, n, m, d).reshape(b * n, m, d)
        mem_agent = torch.stack([fused_conditions, local_contexts], dim=2).reshape(b * n, 2, d)
        memory = torch.cat([mem_agent, mem_map], dim=1)

        goal_h = self.goal_decoder(tgt=tgt, memory=memory).view(b, n, q, d)
        goal_xy = self.goal_pos_head(goal_h)
        goal_logits = self.goal_score_head(goal_h).squeeze(-1)
        return goal_h, goal_xy, goal_logits

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
        """
        Build undirected weighted adjacency for corridor retrieval.
        """
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
    def _expand_path_nodes(path_nodes: List[int], adj: List[List[Tuple[int, float]]], hops: int):
        nodes = set(path_nodes)
        frontier = set(path_nodes)
        for _ in range(max(0, int(hops))):
            new_nodes = set()
            for u in frontier:
                for v, _ in adj[u]:
                    if v not in nodes:
                        new_nodes.add(v)
            if not new_nodes:
                break
            nodes.update(new_nodes)
            frontier = new_nodes
        return sorted(nodes)

    @staticmethod
    def _resample_polyline(poly: torch.Tensor, num_points: int):
        """
        poly: (L,2) -> resampled (T,2) by arc-length.
        """
        l = int(poly.shape[0])
        t = int(num_points)
        if l <= 1:
            return poly[:1].expand(t, 2).contiguous()

        seg = poly[1:] - poly[:-1]  # (L-1,2)
        seg_len = torch.norm(seg, dim=-1).clamp_min(1e-6)  # (L-1,)
        cum = torch.zeros(l, device=poly.device, dtype=poly.dtype)
        cum[1:] = torch.cumsum(seg_len, dim=0)
        total = float(cum[-1].item())
        if total <= 1e-6:
            return poly[:1].expand(t, 2).contiguous()

        s = torch.linspace(0.0, total, steps=t, device=poly.device, dtype=poly.dtype)
        idx = torch.searchsorted(cum[1:], s, right=False)
        idx = idx.clamp(0, l - 2)

        s0 = cum[idx]
        s1 = cum[idx + 1]
        p0 = poly[idx]
        p1 = poly[idx + 1]
        w = ((s - s0) / (s1 - s0).clamp_min(1e-6)).unsqueeze(-1)
        return p0 + w * (p1 - p0)

    def _compute_corridor_with_centerline(
        self,
        graph_embeddings,
        local_contexts,
        top_goal_h,
        kg_positions,
        kg_edge_index,
        num_points: int,
    ):
        """
        Build per-mode corridor features and time-indexed centerlines.

        Returns:
            corridor_feat:   (B,N,K,D)
            corridor_center: (B,N,K,T,2)
        """
        b, n, k, d = top_goal_h.shape
        m = graph_embeddings.shape[1]
        t = int(num_points)
        device = graph_embeddings.device
        dtype = graph_embeddings.dtype

        start_scores = torch.einsum("bnd,bmd->bnm", local_contexts, graph_embeddings)
        start_nodes = start_scores.argmax(dim=-1)  # (B,N)

        goal_scores = torch.einsum("bnkd,bmd->bnkm", top_goal_h, graph_embeddings)
        goal_nodes = goal_scores.argmax(dim=-1)  # (B,N,K)

        corridor_feat = torch.zeros((b, n, k, d), device=device, dtype=dtype)
        corridor_center = torch.zeros((b, n, k, t, 2), device=device, dtype=dtype)

        for bi in range(b):
            edge_b = self._resolve_edge_index_for_batch(kg_edge_index, bi)
            pos_b = self._resolve_positions_for_batch(kg_positions, bi)
            if pos_b.dim() != 2:
                pos_b = pos_b.squeeze(0)
            pos_b = pos_b.to(device=device, dtype=dtype)
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
                    path = self._expand_path_nodes(path, adj, hops=self.corridor_expand_hops)
                    if not path:
                        path = [s]

                    idx = torch.tensor(path, device=device, dtype=torch.long)
                    nodes = pos_b[idx, :2]  # (L,2)
                    corridor_feat[bi, ni, ki] = graph_embeddings[bi, idx].mean(dim=0)
                    corridor_center[bi, ni, ki] = self._resample_polyline(nodes, num_points=t)

        return corridor_feat, corridor_center

    def _build_dynamic_tube(
        self,
        route_h_view,
        interaction_features,
        intent_priors,
        num_points: int,
    ):
        """
        Build time-varying tube radii with pinch effects.

        Returns:
            tube_radius: (B,N,K,T)
            tube_risk:   (B,N,K,1)
            pinch_meta:  (B,N,K,2)  # center, width in normalized time
        """
        b, n, k, d = route_h_view.shape
        t = int(num_points)

        inter = interaction_features.unsqueeze(2).expand(-1, -1, k, -1)
        prior = intent_priors.unsqueeze(2).expand(-1, -1, k, -1)
        risk_in = torch.cat([route_h_view, inter, prior], dim=-1)
        tube_risk = torch.sigmoid(self.dynamic_risk_head(risk_in))  # (B,N,K,1)

        pinch_raw = self.pinch_param_head(route_h_view)  # (B,N,K,2)
        pinch_center = torch.sigmoid(pinch_raw[..., 0:1])                   # [0,1]
        pinch_width = 0.08 + 0.32 * torch.sigmoid(pinch_raw[..., 1:2])      # [0.08,0.40]
        pinch_meta = torch.cat([pinch_center, pinch_width], dim=-1)

        tau = torch.linspace(
            0.0, 1.0, steps=t, device=route_h_view.device, dtype=route_h_view.dtype
        ).view(1, 1, 1, t)
        center = pinch_center.expand(-1, -1, -1, t)
        width = pinch_width.expand(-1, -1, -1, t)
        pinch = torch.exp(-0.5 * ((tau - center) / width.clamp_min(1e-4)) ** 2)  # (B,N,K,T)

        pinch_scale = self.tube_pinch_strength * tube_risk  # (B,N,K,1)
        radius = self.tube_base_radius * (1.0 - pinch_scale * pinch)
        radius = radius.clamp(min=self.tube_min_radius)
        return radius, tube_risk, pinch_meta

    @staticmethod
    def _tube_violation(traj_modes, centerline, radius):
        """
        traj_modes: (B,N,K,T,2)
        centerline: (B,N,K,T,2)
        radius: (B,N,K,T)
        return:
            violation: (B,N,K,T) = relu(dist - radius)
        """
        dist = torch.norm(traj_modes - centerline, dim=-1)
        return torch.relu(dist - radius)

    def _route_decode_elastic_tube(
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
        Stage 2 routing with topology corridor + dynamic elastic tube.
        Returns:
            traj_modes:       (B,N,K,T,2)
            mode_logits:      (B,N,K)
            top_goals:        (B,N,K,2)
            corridor_feat:    (B,N,K,D)
            route_h_view:     (B,N,K,D)
            corridor_center:  (B,N,K,T,2)
            tube_radius:      (B,N,K,T)
            tube_risk:        (B,N,K,1)
        """
        b, n, d = fused_conditions.shape
        q = goal_xy.shape[2]
        k = min(self.route_modes, q)
        t = int(num_points)

        top = goal_logits.topk(k, dim=-1, largest=True, sorted=True)
        top_goal_indices = top.indices   # (B,N,K)
        top_goal_logits = top.values     # (B,N,K)

        gather_xy_idx = top_goal_indices.unsqueeze(-1).expand(b, n, k, 2)
        top_goals = goal_xy.gather(dim=2, index=gather_xy_idx)  # (B,N,K,2)

        gather_h_idx = top_goal_indices.unsqueeze(-1).expand(b, n, k, d)
        top_goal_h = goal_h.gather(dim=2, index=gather_h_idx)  # (B,N,K,D)

        corridor_feat, corridor_center = self._compute_corridor_with_centerline(
            graph_embeddings=graph_embeddings,
            local_contexts=local_contexts,
            top_goal_h=top_goal_h,
            kg_positions=kg_positions,
            kg_edge_index=kg_edge_index,
            num_points=t,
        )

        route_in = torch.cat(
            [fused_conditions.unsqueeze(2).expand(-1, -1, k, -1), top_goals],
            dim=-1,
        )
        route_q = self.goal_to_route(route_in)
        route_q = route_q + 0.5 * intent_priors.unsqueeze(2) + 0.5 * interaction_features.unsqueeze(2)

        # Corridor context injection.
        route_q = route_q + self.corridor_proj(corridor_feat)
        corridor_gate = torch.sigmoid(
            (route_q * corridor_feat).sum(dim=-1, keepdim=True) / math.sqrt(float(d))
        )
        route_q = route_q * (0.5 + corridor_gate)

        tgt = route_q.reshape(b * n, k, d)

        map_pool = graph_embeddings.mean(dim=1).unsqueeze(1).unsqueeze(1).expand(b, n, 1, d).reshape(b * n, 1, d)
        corridor_pool = corridor_feat.mean(dim=2).reshape(b * n, 1, d)
        mem_tokens = torch.stack(
            [fused_conditions, intent_priors, interaction_features, local_contexts],
            dim=2,
        ).reshape(b * n, 4, d)
        memory = torch.cat([mem_tokens, corridor_pool, map_pool], dim=1)

        route_h = self.route_decoder(tgt=tgt, memory=memory)
        route_h_view = route_h.view(b, n, k, d)

        traj_modes = self.route_traj_head(route_h).view(b, n, k, self.prediction_horizon, 2)
        traj_modes = self._resize_traj(traj_modes, num_points=t)

        # Endpoint alignment.
        goal_expand = top_goals.unsqueeze(-2)  # (B,N,K,1,2)
        alpha = torch.linspace(
            0.0, 1.0, t, device=traj_modes.device, dtype=traj_modes.dtype
        ).view(1, 1, 1, t, 1)
        traj_modes = traj_modes + alpha * (goal_expand - traj_modes[..., -1:, :])

        # Dynamic tube.
        tube_radius, tube_risk, pinch_meta = self._build_dynamic_tube(
            route_h_view=route_h_view,
            interaction_features=interaction_features,
            intent_priors=intent_priors,
            num_points=t,
        )
        tube_violation = self._tube_violation(traj_modes, corridor_center, tube_radius)

        route_logits = self.route_score_head(route_h).view(b, n, k)
        corridor_score = F.cosine_similarity(route_h_view, corridor_feat, dim=-1)  # (B,N,K)
        tube_penalty = tube_violation.mean(dim=-1)  # (B,N,K)
        mode_logits = (
            top_goal_logits
            + route_logits
            + self.corridor_score_weight * corridor_score
            - self.tube_score_weight * tube_penalty
        )

        return (
            traj_modes,
            mode_logits,
            top_goals,
            corridor_feat,
            route_h_view,
            corridor_center,
            tube_radius,
            tube_risk,
            pinch_meta,
            tube_violation,
        )

    def forward(self, trajectories, kg_data, future_traj=None, mode="train", vehicle_masks=None):
        """
        Interface-compatible forward.
        In train mode:
        - outputs['diffusion_loss'] is repurposed as elastic-tube training loss.
        """
        outputs = self._build_backbone(trajectories, kg_data, vehicle_masks=vehicle_masks)

        goal_h, goal_xy, goal_logits = self._goal_decode(
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
                top_goals,
                corridor_feat,
                route_h_view,
                corridor_center,
                tube_radius,
                tube_risk,
                pinch_meta,
                tube_violation,
            ) = self._route_decode_elastic_tube(
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
            outputs["corridor_feat"] = corridor_feat
            outputs["corridor_centerline"] = corridor_center
            outputs["tube_radius"] = tube_radius
            outputs["tube_risk"] = tube_risk
            outputs["pinch_meta"] = pinch_meta

            # Best-of-K supervision.
            dist = torch.norm(traj_modes - future_traj.unsqueeze(2), dim=-1).mean(dim=-1)  # (B,N,K)
            best_idx = dist.argmin(dim=-1)  # (B,N)
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

                best_route_h = self._gather_feat(route_h_view, best_idx)
                best_corr = self._gather_feat(corridor_feat, best_idx)
                corr_align = 1.0 - F.cosine_similarity(
                    best_route_h[valid], best_corr[valid], dim=-1
                ).mean()

                best_tube_v = self._gather_time1d(tube_violation, best_idx)    # (B,N,T)
                mask_t = valid.unsqueeze(-1).expand_as(best_tube_v)
                tube_loss = best_tube_v[mask_t].mean() if mask_t.any() else torch.tensor(0.0, device=future_traj.device)

                best_radius = self._gather_time1d(tube_radius, best_idx)       # (B,N,T)
                if t >= 3:
                    d2 = best_radius[..., 2:] - 2.0 * best_radius[..., 1:-1] + best_radius[..., :-2]
                    mask_d2 = valid.unsqueeze(-1).expand_as(d2)
                    tube_smooth = d2.abs()[mask_d2].mean() if mask_d2.any() else torch.tensor(0.0, device=future_traj.device)
                else:
                    tube_smooth = torch.tensor(0.0, device=future_traj.device)
            else:
                z = torch.tensor(0.0, device=future_traj.device, dtype=future_traj.dtype)
                traj_loss = z
                goal_loss = z
                mode_cls = z
                corr_align = z
                tube_loss = z
                tube_smooth = z

            total = (
                traj_loss
                + self.lambda_goal * goal_loss
                + self.lambda_mode_cls * mode_cls
                + self.lambda_corridor * corr_align
                + self.lambda_tube * tube_loss
                + self.lambda_tube_smooth * tube_smooth
            )
            outputs["diffusion_loss"] = total
            outputs["loss_route_head"] = traj_loss
            outputs["loss_goal_head"] = goal_loss
            outputs["loss_mode_cls"] = mode_cls
            outputs["loss_corridor_align"] = corr_align
            outputs["loss_tube"] = tube_loss
            outputs["loss_tube_smooth"] = tube_smooth

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
        """
        Interface-compatible generation.
        - num_samples == 1 -> (B,N,T,2)
        - num_samples > 1  -> (S,B,N,T,2)
        """
        del sampling, step

        outputs = self._build_backbone(trajectories, kg_data, vehicle_masks=vehicle_masks)
        goal_h, goal_xy, goal_logits = self._goal_decode(
            outputs["fused_conditions"],
            outputs["local_contexts"],
            outputs["graph_embeddings"],
        )
        (
            traj_modes,
            mode_logits,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = self._route_decode_elastic_tube(
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


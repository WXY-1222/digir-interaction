"""
DIGIR variant: Goal-to-Trajectory Cascaded Decoding with
Homotopy-Class Guided Topological Bifurcation.

Upgrade over topology-corridor guided routing:
- For each goal, generate multiple topology-distinct corridor classes.
- Inject different homotopy corridor features into different routing queries.
- Encourage same-goal / different-homotopy trajectory bifurcation.

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
    Cascaded decoder + homotopy-class topological bifurcation.
    """

    def __init__(self, config):
        super().__init__(config)

        self.goal_query_count = int(config.get("goal_query_count", 32))
        self.route_modes = int(config.get("route_modes", 6))
        self.route_temperature = float(config.get("route_temperature", 1.0))

        # Homotopy controls.
        self.homotopy_classes = max(2, int(config.get("homotopy_classes", 3)))
        self.corridor_expand_hops = int(config.get("corridor_expand_hops", 1))
        self.homotopy_node_penalty = float(config.get("homotopy_node_penalty", 3.0))
        self.homotopy_edge_penalty = float(config.get("homotopy_edge_penalty", 4.0))
        self.homotopy_score_weight = float(config.get("homotopy_score_weight", 0.2))
        self.lambda_homotopy = float(config.get("lambda_homotopy", 0.1))
        self.homotopy_margin = float(config.get("homotopy_margin", 0.6))

        # Base cascade loss terms.
        self.lambda_corridor = float(config.get("lambda_corridor", 0.2))
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
        self.homotopy_embed = nn.Embedding(self.homotopy_classes, d)

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

    def _compute_homotopy_corridors(
        self,
        graph_embeddings,
        local_contexts,
        top_goal_h,
        kg_positions,
        kg_edge_index,
    ):
        """
        Build per-goal multi-homotopy corridor features.

        Returns:
            corridor_feat: (B,N,G,H,D)
            path_score: (B,N,G,H)
        """
        b, n, g, d = top_goal_h.shape
        m = graph_embeddings.shape[1]
        h_cls = self.homotopy_classes
        device = graph_embeddings.device

        # Start-node & goal-node proposal in graph feature space.
        start_scores = torch.einsum("bnd,bmd->bnm", local_contexts, graph_embeddings)
        start_nodes = start_scores.argmax(dim=-1)  # (B,N)

        goal_scores = torch.einsum("bngd,bmd->bngm", top_goal_h, graph_embeddings)
        goal_nodes = goal_scores.argmax(dim=-1)  # (B,N,G)

        corridor_feat = torch.zeros((b, n, g, h_cls, d), device=device, dtype=graph_embeddings.dtype)
        path_score = torch.zeros((b, n, g, h_cls), device=device, dtype=graph_embeddings.dtype)

        for bi in range(b):
            edge_b = self._resolve_edge_index_for_batch(kg_edge_index, bi)
            pos_b = self._resolve_positions_for_batch(kg_positions, bi)
            if pos_b.dim() != 2:
                pos_b = pos_b.squeeze(0)
            adj = self._build_adjacency(edge_b, num_nodes=m, node_pos_b=pos_b)

            for ni in range(n):
                s = int(start_nodes[bi, ni].item())
                s = max(0, min(m - 1, s))

                for gi in range(g):
                    gnode = int(goal_nodes[bi, ni, gi].item())
                    gnode = max(0, min(m - 1, gnode))

                    node_penalty = [0.0 for _ in range(m)]
                    edge_penalty = {}

                    for hi in range(h_cls):
                        if any(adj[s]):
                            dist, prev = self._dijkstra_with_penalty(
                                start=s,
                                goal=gnode,
                                adj=adj,
                                node_penalty=node_penalty,
                                edge_penalty=edge_penalty,
                            )
                            path = self._reconstruct_path(prev, s, gnode)
                            dscore = float(dist[gnode]) if math.isfinite(dist[gnode]) else 0.0
                        else:
                            path = [s, gnode] if s != gnode else [s]
                            dscore = 0.0

                        path = self._expand_path_nodes(path, adj, hops=self.corridor_expand_hops)
                        if not path:
                            path = [s]

                        idx = torch.tensor(path, device=device, dtype=torch.long)
                        corridor_feat[bi, ni, gi, hi] = graph_embeddings[bi, idx].mean(dim=0)
                        path_score[bi, ni, gi, hi] = -float(dscore)  # larger is better

                        # Penalize selected path internals to force another homotopy branch.
                        for u in path[1:-1]:
                            if 0 <= u < m:
                                node_penalty[u] += self.homotopy_node_penalty
                        for a, b2 in zip(path[:-1], path[1:]):
                            edge_penalty[(a, b2)] = edge_penalty.get((a, b2), 0.0) + self.homotopy_edge_penalty
                            edge_penalty[(b2, a)] = edge_penalty.get((b2, a), 0.0) + self.homotopy_edge_penalty

        # Normalize path scores per (B,N,G).
        pmax = path_score.max(dim=-1, keepdim=True).values
        path_score = path_score - pmax
        return corridor_feat, path_score

    @staticmethod
    def _select_mode_combinations(
        goal_logits_exp,
        path_score_exp,
        mode_goals_exp,
        mode_corr_exp,
        mode_hid_exp,
        mode_gid_exp,
        k_target: int,
    ):
        """
        Select up to k_target combinations from expanded (goal,homotopy) pairs.
        Shapes:
            goal_logits_exp: (B,N,C)
            path_score_exp:  (B,N,C)
            mode_goals_exp:  (B,N,C,2)
            mode_corr_exp:   (B,N,C,D)
            mode_hid_exp:    (B,N,C)
            mode_gid_exp:    (B,N,C)
        Returns:
            selected tensors with C' = k_target.
        """
        b, n, c = goal_logits_exp.shape
        if c == k_target:
            return (
                goal_logits_exp,
                path_score_exp,
                mode_goals_exp,
                mode_corr_exp,
                mode_hid_exp,
                mode_gid_exp,
            )

        if c > k_target:
            combo = goal_logits_exp + path_score_exp
            top = combo.topk(k_target, dim=-1, largest=True, sorted=True)
            idx = top.indices  # (B,N,K)

            g_idx2 = idx.unsqueeze(-1).expand(b, n, k_target, 2)
            g_idxd = idx.unsqueeze(-1).expand(b, n, k_target, mode_corr_exp.shape[-1])
            sel_goal_logits = goal_logits_exp.gather(2, idx)
            sel_path_score = path_score_exp.gather(2, idx)
            sel_goals = mode_goals_exp.gather(2, g_idx2)
            sel_corr = mode_corr_exp.gather(2, g_idxd)
            sel_hid = mode_hid_exp.gather(2, idx)
            sel_gid = mode_gid_exp.gather(2, idx)
            return sel_goal_logits, sel_path_score, sel_goals, sel_corr, sel_hid, sel_gid

        # c < k_target: repeat from front.
        rep = int(math.ceil(float(k_target) / float(c)))
        sel_goal_logits = goal_logits_exp.repeat(1, 1, rep)[:, :, :k_target]
        sel_path_score = path_score_exp.repeat(1, 1, rep)[:, :, :k_target]
        sel_goals = mode_goals_exp.repeat(1, 1, rep, 1)[:, :, :k_target, :]
        sel_corr = mode_corr_exp.repeat(1, 1, rep, 1)[:, :, :k_target, :]
        sel_hid = mode_hid_exp.repeat(1, 1, rep)[:, :, :k_target]
        sel_gid = mode_gid_exp.repeat(1, 1, rep)[:, :, :k_target]
        return sel_goal_logits, sel_path_score, sel_goals, sel_corr, sel_hid, sel_gid

    def _route_decode_homotopy(
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
        Stage 2 routing with homotopy-class topological bifurcation.

        Returns:
            traj_modes: (B,N,K,T,2)
            mode_logits: (B,N,K)
            mode_goals: (B,N,K,2)
            mode_corr: (B,N,K,D)
            route_h_view: (B,N,K,D)
            mode_hid: (B,N,K) homotopy class id
            mode_gid: (B,N,K) goal slot id
        """
        b, n, d = fused_conditions.shape
        q = goal_xy.shape[2]
        h_cls = self.homotopy_classes
        k_target = self.route_modes

        goal_slots = max(1, min(q, int(math.ceil(float(k_target) / float(h_cls)))))

        top = goal_logits.topk(goal_slots, dim=-1, largest=True, sorted=True)
        top_goal_indices = top.indices   # (B,N,G)
        top_goal_logits = top.values     # (B,N,G)

        gather_xy_idx = top_goal_indices.unsqueeze(-1).expand(b, n, goal_slots, 2)
        top_goals = goal_xy.gather(dim=2, index=gather_xy_idx)  # (B,N,G,2)

        gather_h_idx = top_goal_indices.unsqueeze(-1).expand(b, n, goal_slots, d)
        top_goal_h = goal_h.gather(dim=2, index=gather_h_idx)  # (B,N,G,D)

        corridor_feat, path_score = self._compute_homotopy_corridors(
            graph_embeddings=graph_embeddings,
            local_contexts=local_contexts,
            top_goal_h=top_goal_h,
            kg_positions=kg_positions,
            kg_edge_index=kg_edge_index,
        )  # corridor: (B,N,G,H,D), path_score: (B,N,G,H)

        # Expand (goal, homotopy) combinations.
        g = goal_slots
        h = h_cls
        c = g * h

        mode_goals_exp = top_goals.unsqueeze(3).expand(b, n, g, h, 2).reshape(b, n, c, 2)
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
            mode_corr,
            mode_hid,
            mode_gid,
        ) = self._select_mode_combinations(
            goal_logits_exp=goal_logits_exp,
            path_score_exp=path_score_exp,
            mode_goals_exp=mode_goals_exp,
            mode_corr_exp=mode_corr_exp,
            mode_hid_exp=mode_hid_exp,
            mode_gid_exp=mode_gid_exp,
            k_target=k_target,
        )

        # Build route queries.
        route_in = torch.cat(
            [fused_conditions.unsqueeze(2).expand(-1, -1, k_target, -1), mode_goals],
            dim=-1,
        )
        route_q = self.goal_to_route(route_in)
        route_q = route_q + 0.5 * intent_priors.unsqueeze(2) + 0.5 * interaction_features.unsqueeze(2)
        route_q = route_q + self.corridor_proj(mode_corr)
        route_q = route_q + self.homotopy_embed(mode_hid)

        tgt = route_q.reshape(b * n, k_target, d)

        map_pool = graph_embeddings.mean(dim=1).unsqueeze(1).unsqueeze(1).expand(b, n, 1, d).reshape(b * n, 1, d)
        corridor_pool = mode_corr.mean(dim=2).reshape(b * n, 1, d)
        mem_tokens = torch.stack(
            [fused_conditions, intent_priors, interaction_features, local_contexts],
            dim=2,
        ).reshape(b * n, 4, d)
        memory = torch.cat([mem_tokens, corridor_pool, map_pool], dim=1)

        route_h = self.route_decoder(tgt=tgt, memory=memory)
        route_h_view = route_h.view(b, n, k_target, d)

        traj_modes = self.route_traj_head(route_h).view(b, n, k_target, self.prediction_horizon, 2)
        traj_modes = self._resize_traj(traj_modes, num_points=int(num_points))

        # Endpoint alignment to selected goals.
        goal_expand = mode_goals.unsqueeze(-2)  # (B,N,K,1,2)
        alpha = torch.linspace(
            0.0, 1.0, int(num_points), device=traj_modes.device, dtype=traj_modes.dtype
        ).view(1, 1, 1, int(num_points), 1)
        traj_modes = traj_modes + alpha * (goal_expand - traj_modes[..., -1:, :])

        route_logits = self.route_score_head(route_h).view(b, n, k_target)
        corridor_score = F.cosine_similarity(route_h_view, mode_corr, dim=-1)  # (B,N,K)
        mode_logits = (
            sel_goal_logits
            + route_logits
            + self.homotopy_score_weight * sel_path_score
            + 0.2 * corridor_score
        )

        return traj_modes, mode_logits, mode_goals, mode_corr, route_h_view, mode_hid, mode_gid

    def _homotopy_bifurcation_loss(self, traj_modes, mode_hid, mode_gid, vehicle_masks):
        """
        Encourage same-goal / different-homotopy trajectories to stay separated.

        traj_modes: (B,N,K,T,2)
        mode_hid: (B,N,K)
        mode_gid: (B,N,K)
        """
        b, n, k, _, _ = traj_modes.shape
        if k < 2:
            return torch.tensor(0.0, device=traj_modes.device, dtype=traj_modes.dtype)

        traj_i = traj_modes.unsqueeze(3)  # (B,N,K,1,T,2)
        traj_j = traj_modes.unsqueeze(2)  # (B,N,1,K,T,2)
        pair_dist = torch.norm(traj_i - traj_j, dim=-1).mean(dim=-1)  # (B,N,K,K)

        same_goal = mode_gid.unsqueeze(3) == mode_gid.unsqueeze(2)
        diff_h = mode_hid.unsqueeze(3) != mode_hid.unsqueeze(2)
        upper = torch.triu(
            torch.ones((k, k), device=traj_modes.device, dtype=torch.bool),
            diagonal=1,
        ).view(1, 1, k, k)

        valid_agent = vehicle_masks.bool().unsqueeze(-1).unsqueeze(-1)  # (B,N,1,1)
        mask = same_goal & diff_h & upper & valid_agent
        if not mask.any():
            return torch.tensor(0.0, device=traj_modes.device, dtype=traj_modes.dtype)

        pen = torch.relu(float(self.homotopy_margin) - pair_dist)
        return pen[mask].mean()

    def forward(self, trajectories, kg_data, future_traj=None, mode="train", vehicle_masks=None):
        """
        Interface-compatible forward.
        In train mode:
        - outputs['diffusion_loss'] is repurposed as homotopy-cascade training loss.
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
                mode_goals,
                mode_corr,
                route_h_view,
                mode_hid,
                mode_gid,
            ) = self._route_decode_homotopy(
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

            # Best-of-K supervision.
            dist = torch.norm(traj_modes - future_traj.unsqueeze(2), dim=-1).mean(dim=-1)  # (B,N,K)
            best_idx = dist.argmin(dim=-1)  # (B,N)
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
                best_route_h = self._gather_feat(route_h_view, best_idx)
                best_corr = self._gather_feat(mode_corr, best_idx)
                corr_align = 1.0 - F.cosine_similarity(
                    best_route_h[valid], best_corr[valid], dim=-1
                ).mean()
                homotopy_loss = self._homotopy_bifurcation_loss(
                    traj_modes=traj_modes,
                    mode_hid=mode_hid,
                    mode_gid=mode_gid,
                    vehicle_masks=valid,
                )
            else:
                z = torch.tensor(0.0, device=future_traj.device, dtype=future_traj.dtype)
                traj_loss = z
                goal_loss = z
                mode_cls = z
                corr_align = z
                homotopy_loss = z

            total = (
                traj_loss
                + self.lambda_goal * goal_loss
                + self.lambda_mode_cls * mode_cls
                + self.lambda_corridor * corr_align
                + self.lambda_homotopy * homotopy_loss
            )
            outputs["diffusion_loss"] = total
            outputs["loss_route_head"] = traj_loss
            outputs["loss_goal_head"] = goal_loss
            outputs["loss_mode_cls"] = mode_cls
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
        ) = self._route_decode_homotopy(
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


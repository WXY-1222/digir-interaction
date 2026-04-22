"""
DIGIR variant: Goal-to-Trajectory Cascaded Decoding with
Differentiable Graph Soft-Routing.

Upgrade over topology-corridor guided routing:
- Remove hard path search (Dijkstra/BFS).
- Learn a differentiable soft corridor as node/edge probability heatmaps.
- Feed soft topological heatmap features into routing decoder.

This file does NOT overwrite existing model scripts.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.digir import DIGIR as BaseDIGIR


class DIGIR(BaseDIGIR):
    """
    Cascaded decoder + differentiable graph soft-routing.
    """

    def __init__(self, config):
        super().__init__(config)

        self.goal_query_count = int(config.get("goal_query_count", 32))
        self.route_modes = int(config.get("route_modes", 6))
        self.route_temperature = float(config.get("route_temperature", 1.0))

        # Soft-routing controls.
        self.soft_route_blend = float(config.get("soft_route_blend", 0.45))  # blend node prior and edge-induced mass
        self.soft_route_tau = float(config.get("soft_route_tau", 1.0))
        self.corridor_score_weight = float(config.get("corridor_score_weight", 0.2))
        self.heat_score_weight = float(config.get("heat_score_weight", 0.2))

        # Loss weights.
        self.lambda_goal = float(config.get("lambda_goal", 0.5))
        self.lambda_mode_cls = float(config.get("lambda_mode_cls", 0.3))
        self.lambda_corridor = float(config.get("lambda_corridor", 0.2))
        self.lambda_soft_entropy = float(config.get("lambda_soft_entropy", 0.05))
        self.lambda_soft_consistency = float(config.get("lambda_soft_consistency", 0.1))

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
        self.heat_proj = nn.Linear(d, d)

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

    @staticmethod
    def _gather_heat(heat_modes, mode_idx):
        """
        heat_modes: (B,N,K,M), mode_idx: (B,N) -> (B,N,M)
        """
        b, n, _, m = heat_modes.shape
        idx = mode_idx.view(b, n, 1, 1).expand(b, n, 1, m)
        return heat_modes.gather(2, idx).squeeze(2)

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

    def _compute_soft_corridor(
        self,
        graph_embeddings,
        local_contexts,
        top_goal_h,
        edge_index,
    ):
        """
        Differentiable graph soft-routing on nodes and edges.

        Returns:
            corridor_feat:    (B,N,K,D)
            node_heatmap:     (B,N,K,M)
            edge_prob:        (B,N,K,Emax)
            node_from_edges:  (B,N,K,M)
        """
        b, m, d = graph_embeddings.shape
        _, n, k, _ = top_goal_h.shape
        eps = 1e-8
        tau = max(float(self.soft_route_tau), 1e-4)

        # Route query for soft graph diffusion (agent state + goal state).
        route_query = F.normalize(
            top_goal_h + local_contexts.unsqueeze(2).expand(-1, -1, k, -1),
            dim=-1,
        )  # (B,N,K,D)

        # Node prior probability from query-node affinity.
        node_logits = torch.einsum("bnkd,bmd->bnkm", route_query, graph_embeddings) / math.sqrt(float(d))
        node_prior = torch.softmax(node_logits / tau, dim=-1)  # (B,N,K,M)

        # Prepare edge prob container.
        if edge_index.dim() == 3:
            emax = int(edge_index.shape[-1])
        else:
            emax = int(edge_index.shape[-1]) if edge_index.numel() > 0 else 0
        edge_prob = torch.zeros((b, n, k, emax), device=graph_embeddings.device, dtype=graph_embeddings.dtype)

        node_from_edges = torch.zeros_like(node_prior)
        blend = max(0.0, min(1.0, float(self.soft_route_blend)))

        for bi in range(b):
            ei = self._resolve_edge_index_for_batch(edge_index, bi)
            if ei.numel() == 0:
                node_from_edges[bi] = node_prior[bi]
                continue

            src_all = ei[0].long()
            dst_all = ei[1].long()
            valid = (
                (src_all >= 0)
                & (dst_all >= 0)
                & (src_all < m)
                & (dst_all < m)
            )
            valid_idx = torch.where(valid)[0]
            if valid_idx.numel() == 0:
                node_from_edges[bi] = node_prior[bi]
                continue

            src = src_all[valid_idx]
            dst = dst_all[valid_idx]
            nodes = graph_embeddings[bi]  # (M,D)
            edge_feat = 0.5 * (nodes[src] + nodes[dst])  # (E,D)

            q_flat = route_query[bi].reshape(n * k, d)  # (NK,D)
            edge_logits = torch.matmul(q_flat, edge_feat.t()) / math.sqrt(float(d))  # (NK,E)

            # Encourage edges connected to high-prior nodes.
            prior_flat = node_prior[bi].reshape(n * k, m)
            bias = 0.5 * (prior_flat[:, src] + prior_flat[:, dst]).clamp_min(eps)
            edge_logits = edge_logits + bias.log()

            edge_p = torch.softmax(edge_logits / tau, dim=-1)  # (NK,E)
            edge_p = torch.nan_to_num(edge_p, nan=0.0, posinf=0.0, neginf=0.0)

            # Scatter edge mass to nodes.
            node_mass = torch.zeros((n * k, m), device=nodes.device, dtype=nodes.dtype)
            node_mass.index_add_(1, src, edge_p)
            node_mass.index_add_(1, dst, edge_p)
            node_mass = node_mass / node_mass.sum(dim=-1, keepdim=True).clamp_min(eps)

            node_from_edges[bi] = node_mass.view(n, k, m)
            edge_prob[bi, :, :, valid_idx] = edge_p.view(n, k, -1)

        node_heat = (1.0 - blend) * node_prior + blend * node_from_edges
        node_heat = node_heat / node_heat.sum(dim=-1, keepdim=True).clamp_min(eps)

        corridor_feat = torch.einsum("bnkm,bmd->bnkd", node_heat, graph_embeddings)  # (B,N,K,D)
        return corridor_feat, node_heat, edge_prob, node_from_edges

    def _route_decode_soft_graph(
        self,
        fused_conditions,
        intent_priors,
        interaction_features,
        local_contexts,
        graph_embeddings,
        goal_h,
        goal_xy,
        goal_logits,
        edge_index,
        num_points,
    ):
        """
        Stage 2 routing with differentiable graph soft-routing.

        Returns:
            traj_modes: (B,N,K,T,2)
            mode_logits: (B,N,K)
            top_goals: (B,N,K,2)
            top_goal_h: (B,N,K,D)
            corridor_feat: (B,N,K,D)
            route_h_view: (B,N,K,D)
            node_heatmap: (B,N,K,M)
            edge_prob: (B,N,K,Emax)
            node_from_edges: (B,N,K,M)
        """
        b, n, d = fused_conditions.shape
        q = goal_xy.shape[2]
        k = min(self.route_modes, q)

        top = goal_logits.topk(k, dim=-1, largest=True, sorted=True)
        top_goal_indices = top.indices
        top_goal_logits = top.values

        gather_xy_idx = top_goal_indices.unsqueeze(-1).expand(b, n, k, 2)
        top_goals = goal_xy.gather(dim=2, index=gather_xy_idx)  # (B,N,K,2)

        gather_h_idx = top_goal_indices.unsqueeze(-1).expand(b, n, k, d)
        top_goal_h = goal_h.gather(dim=2, index=gather_h_idx)   # (B,N,K,D)

        corridor_feat, node_heatmap, edge_prob, node_from_edges = self._compute_soft_corridor(
            graph_embeddings=graph_embeddings,
            local_contexts=local_contexts,
            top_goal_h=top_goal_h,
            edge_index=edge_index,
        )

        route_in = torch.cat(
            [fused_conditions.unsqueeze(2).expand(-1, -1, k, -1), top_goals],
            dim=-1,
        )
        route_q = self.goal_to_route(route_in)
        route_q = route_q + 0.5 * intent_priors.unsqueeze(2) + 0.5 * interaction_features.unsqueeze(2)

        # Inject soft-corridor feature and heatmap token.
        heat_token = torch.einsum(
            "bnkm,bmd->bnkd",
            node_heatmap,
            graph_embeddings,
        )
        route_q = route_q + self.corridor_proj(corridor_feat) + self.heat_proj(heat_token)

        # Confidence gate from heatmap concentration.
        heat_conf = node_heatmap.max(dim=-1).values.unsqueeze(-1)  # (B,N,K,1)
        route_q = route_q * (0.5 + heat_conf)

        tgt = route_q.reshape(b * n, k, d)

        map_pool = graph_embeddings.mean(dim=1).unsqueeze(1).unsqueeze(1).expand(b, n, 1, d).reshape(b * n, 1, d)
        corr_pool = corridor_feat.mean(dim=2).reshape(b * n, 1, d)
        heat_pool = heat_token.mean(dim=2).reshape(b * n, 1, d)
        mem_tokens = torch.stack(
            [fused_conditions, intent_priors, interaction_features, local_contexts],
            dim=2,
        ).reshape(b * n, 4, d)
        memory = torch.cat([mem_tokens, corr_pool, heat_pool, map_pool], dim=1)  # (BN,7,D)

        route_h = self.route_decoder(tgt=tgt, memory=memory)
        route_h_view = route_h.view(b, n, k, d)

        traj_modes = self.route_traj_head(route_h).view(b, n, k, self.prediction_horizon, 2)
        traj_modes = self._resize_traj(traj_modes, num_points=int(num_points))

        # Endpoint alignment to selected goals.
        goal_expand = top_goals.unsqueeze(-2)  # (B,N,K,1,2)
        alpha = torch.linspace(
            0.0, 1.0, int(num_points), device=traj_modes.device, dtype=traj_modes.dtype
        ).view(1, 1, 1, int(num_points), 1)
        traj_modes = traj_modes + alpha * (goal_expand - traj_modes[..., -1:, :])

        route_logits = self.route_score_head(route_h).view(b, n, k)
        corridor_score = F.cosine_similarity(route_h_view, corridor_feat, dim=-1)  # (B,N,K)
        mode_logits = (
            top_goal_logits
            + route_logits
            + self.corridor_score_weight * corridor_score
            + self.heat_score_weight * heat_conf.squeeze(-1)
        )

        return (
            traj_modes,
            mode_logits,
            top_goals,
            top_goal_h,
            corridor_feat,
            route_h_view,
            node_heatmap,
            edge_prob,
            node_from_edges,
        )

    def forward(self, trajectories, kg_data, future_traj=None, mode="train", vehicle_masks=None):
        """
        Interface-compatible forward.
        In train mode:
        - outputs['diffusion_loss'] is repurposed as soft-graph-route training loss.
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
                _,
                corridor_feat,
                route_h_view,
                node_heatmap,
                edge_prob,
                node_from_edges,
            ) = self._route_decode_soft_graph(
                outputs["fused_conditions"],
                outputs["intent_priors"],
                outputs["interaction_features"],
                outputs["local_contexts"],
                outputs["graph_embeddings"],
                goal_h,
                goal_xy,
                goal_logits,
                outputs["kg_edge_index"],
                num_points=t,
            )
            outputs["mode_logits"] = mode_logits
            outputs["traj_modes"] = traj_modes
            outputs["corridor_feat"] = corridor_feat
            outputs["node_heatmap"] = node_heatmap
            outputs["edge_routing_prob"] = edge_prob

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

                # Soft-routing regularizers on the chosen mode heatmap.
                eps = 1e-8
                best_heat = self._gather_heat(node_heatmap, best_idx)  # (B,N,M)
                best_edge_node = self._gather_heat(node_from_edges, best_idx)  # (B,N,M)
                heat_entropy = -(best_heat.clamp_min(eps) * best_heat.clamp_min(eps).log()).sum(dim=-1)
                heat_entropy = heat_entropy[valid].mean()

                # KL(best_heat || best_edge_node) for topology consistency.
                p = best_heat[valid].clamp_min(eps)
                q = best_edge_node[valid].clamp_min(eps)
                q = q / q.sum(dim=-1, keepdim=True).clamp_min(eps)
                soft_cons = (p * (p.log() - q.log())).sum(dim=-1).mean()
            else:
                z = torch.tensor(0.0, device=future_traj.device, dtype=future_traj.dtype)
                traj_loss = z
                goal_loss = z
                mode_cls = z
                corr_align = z
                heat_entropy = z
                soft_cons = z

            total = (
                traj_loss
                + self.lambda_goal * goal_loss
                + self.lambda_mode_cls * mode_cls
                + self.lambda_corridor * corr_align
                + self.lambda_soft_entropy * heat_entropy
                + self.lambda_soft_consistency * soft_cons
            )
            outputs["diffusion_loss"] = total
            outputs["loss_route_head"] = traj_loss
            outputs["loss_goal_head"] = goal_loss
            outputs["loss_mode_cls"] = mode_cls
            outputs["loss_corridor_align"] = corr_align
            outputs["loss_soft_entropy"] = heat_entropy
            outputs["loss_soft_consistency"] = soft_cons

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
        ) = self._route_decode_soft_graph(
            outputs["fused_conditions"],
            outputs["intent_priors"],
            outputs["interaction_features"],
            outputs["local_contexts"],
            outputs["graph_embeddings"],
            goal_h,
            goal_xy,
            goal_logits,
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


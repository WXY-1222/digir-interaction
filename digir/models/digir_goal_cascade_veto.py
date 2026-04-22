"""
DIGIR variant: Goal-to-Trajectory Cascaded Decoding with Bottom-Up Veto Feedback.

Core idea:
- Stage-1 (Goal Decoder): predict candidate endpoints and goal scores.
- Stage-2 (Routing Decoder): predict trajectories + route feasibility/cost.
- Bottom-up veto: use routing feasibility to rescore top-down goals.

This file does NOT overwrite existing model scripts.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.digir import DIGIR as BaseDIGIR


class DIGIR(BaseDIGIR):
    """
    Cascaded decoder + routing-aware veto feedback.
    """

    def __init__(self, config):
        super().__init__(config)

        self.goal_query_count = int(config.get("goal_query_count", 32))
        self.route_modes = int(config.get("route_modes", 6))
        self.route_temperature = float(config.get("route_temperature", 1.0))

        # Bottom-up veto parameters
        self.veto_scale = float(config.get("veto_scale", 1.0))
        self.veto_map_weight = float(config.get("veto_map_weight", 0.3))
        self.veto_dyn_weight = float(config.get("veto_dyn_weight", 0.2))
        self.veto_map_margin = float(config.get("veto_map_margin", 2.5))

        # Loss weights
        self.lambda_goal = float(config.get("lambda_goal", 0.5))
        self.lambda_mode_cls = float(config.get("lambda_mode_cls", 0.3))
        self.lambda_veto = float(config.get("lambda_veto", 0.3))

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
        self.route_cost_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.ReLU(),
            nn.Linear(d // 2, 1),
        )

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
        """
        traj_modes: (B,N,K,T0,2) -> (B,N,K,T,2)
        """
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
        b, _, _, _ = trajectories.shape

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

        goal_h = self.goal_decoder(tgt=tgt, memory=memory)
        goal_xy = self.goal_pos_head(goal_h).view(b, n, q, 2)
        goal_logits = self.goal_score_head(goal_h).view(b, n, q)
        return goal_xy, goal_logits

    def _map_cost(self, traj_modes, kg_positions):
        """
        Simple off-road proxy cost:
        average positive excess distance to nearest map node beyond margin.
        traj_modes: (B,N,K,T,2)
        kg_positions: (B,M,2) or (M,2)
        returns: (B,N,K)
        """
        b, n, k, t, _ = traj_modes.shape
        if kg_positions.dim() == 2:
            kg_positions = kg_positions.unsqueeze(0).expand(b, -1, -1)

        costs = []
        for bi in range(b):
            nodes = kg_positions[bi, :, :2]  # (M,2)
            pts = traj_modes[bi].reshape(n * k * t, 2)  # (N*K*T,2)
            d = torch.cdist(pts, nodes)  # (N*K*T, M)
            min_d = d.min(dim=1).values.reshape(n, k, t)
            c = F.relu(min_d - self.veto_map_margin).mean(dim=-1)  # (N,K)
            costs.append(c)
        return torch.stack(costs, dim=0)  # (B,N,K)

    @staticmethod
    def _dyn_cost(traj_modes):
        """
        Kinematic roughness proxy based on acceleration magnitude.
        traj_modes: (B,N,K,T,2)
        returns: (B,N,K)
        """
        if traj_modes.shape[-2] < 3:
            return torch.zeros(traj_modes.shape[:3], device=traj_modes.device, dtype=traj_modes.dtype)
        vel = traj_modes[:, :, :, 1:, :] - traj_modes[:, :, :, :-1, :]
        acc = vel[:, :, :, 1:, :] - vel[:, :, :, :-1, :]
        return torch.norm(acc, dim=-1).mean(dim=-1)

    def _route_decode_with_veto(
        self,
        fused_conditions,
        intent_priors,
        interaction_features,
        local_contexts,
        graph_embeddings,
        goal_xy,
        goal_logits,
        kg_positions,
        num_points,
    ):
        """
        Stage 2 routing + veto rescoring.
        Returns:
            traj_modes: (B,N,K,T,2)
            mode_logits: (B,N,K) final rescored
            top_goals: (B,N,K,2)
            veto_cost: (B,N,K)
            route_logits: (B,N,K)
            top_goal_logits: (B,N,K)
        """
        b, n, d = fused_conditions.shape
        k = min(self.route_modes, goal_xy.shape[2])

        # Hard goal candidates for execution layer (keeps explicit top-down intent path).
        top = goal_logits.topk(k, dim=-1, largest=True, sorted=True)
        top_goal_indices = top.indices
        top_goal_logits = top.values
        gather_idx = top_goal_indices.unsqueeze(-1).expand(b, n, k, 2)
        top_goals = goal_xy.gather(dim=2, index=gather_idx)  # (B,N,K,2)

        route_in = torch.cat(
            [fused_conditions.unsqueeze(2).expand(-1, -1, k, -1), top_goals],
            dim=-1,
        )
        route_q = self.goal_to_route(route_in)
        route_q = route_q + 0.5 * intent_priors.unsqueeze(2) + 0.5 * interaction_features.unsqueeze(2)

        tgt = route_q.reshape(b * n, k, d)

        map_pool = graph_embeddings.mean(dim=1).unsqueeze(1).unsqueeze(1).expand(b, n, 1, d).reshape(b * n, 1, d)
        mem_tokens = torch.stack(
            [fused_conditions, intent_priors, interaction_features, local_contexts],
            dim=2,
        ).reshape(b * n, 4, d)
        memory = torch.cat([mem_tokens, map_pool], dim=1)

        route_h = self.route_decoder(tgt=tgt, memory=memory)  # (BN,K,D)
        traj_modes = self.route_traj_head(route_h).view(b, n, k, self.prediction_horizon, 2)
        traj_modes = self._resize_traj(traj_modes, int(num_points))

        # Endpoint alignment to selected goals.
        goal_expand = top_goals.unsqueeze(-2)
        alpha = torch.linspace(
            0.0, 1.0, int(num_points), device=traj_modes.device, dtype=traj_modes.dtype
        ).view(1, 1, 1, int(num_points), 1)
        traj_modes = traj_modes + alpha * (goal_expand - traj_modes[..., -1:, :])

        route_h_view = route_h.view(b, n, k, d)
        route_logits = self.route_score_head(route_h).view(b, n, k)

        # Bottom-up veto cost (execution feasibility).
        raw_cost = F.softplus(self.route_cost_head(route_h).view(b, n, k))
        map_cost = self._map_cost(traj_modes, kg_positions)
        dyn_cost = self._dyn_cost(traj_modes)
        veto_cost = raw_cost + self.veto_map_weight * map_cost + self.veto_dyn_weight * dyn_cost

        # Closed-loop rescoring:
        # P_final ∝ P(goal|scene) * P(traj|goal,obstacles)
        # score space approximation:
        # log P_final = goal_score + route_score - veto_scale * cost
        mode_logits = top_goal_logits + route_logits - self.veto_scale * veto_cost

        return traj_modes, mode_logits, top_goals, veto_cost, route_logits, top_goal_logits

    def forward(self, trajectories, kg_data, future_traj=None, mode="train", vehicle_masks=None):
        """
        Interface-compatible forward.
        In train mode:
        - outputs['diffusion_loss'] is repurposed as veto-cascade training loss.
        """
        outputs = self._build_backbone(trajectories, kg_data, vehicle_masks=vehicle_masks)

        goal_xy, goal_logits = self._goal_decode(
            outputs["fused_conditions"],
            outputs["local_contexts"],
            outputs["graph_embeddings"],
        )
        outputs["goal_xy"] = goal_xy
        outputs["goal_logits"] = goal_logits

        if mode == "train" and future_traj is not None:
            b, n, t, _ = future_traj.shape
            traj_modes, mode_logits, top_goals, veto_cost, _, _ = self._route_decode_with_veto(
                outputs["fused_conditions"],
                outputs["intent_priors"],
                outputs["interaction_features"],
                outputs["local_contexts"],
                outputs["graph_embeddings"],
                goal_xy,
                goal_logits,
                outputs["kg_positions"],
                num_points=t,
            )
            outputs["traj_modes"] = traj_modes
            outputs["mode_logits"] = mode_logits
            outputs["veto_cost"] = veto_cost
            outputs["route_confidence"] = torch.exp(-veto_cost)

            # Best-of-K supervision.
            dist = torch.norm(traj_modes - future_traj.unsqueeze(2), dim=-1).mean(dim=-1)  # (B,N,K)
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
            valid_k = valid.unsqueeze(-1).expand_as(veto_cost)

            if valid.any():
                traj_loss = F.smooth_l1_loss(best_traj[valid], future_traj[valid], reduction="mean")
                goal_gt = future_traj[:, :, -1, :]
                goal_loss = F.smooth_l1_loss(best_goal[valid], goal_gt[valid], reduction="mean")
                mode_cls = F.cross_entropy(
                    mode_logits.view(b * n, -1)[valid_flat],
                    best_idx.view(-1)[valid_flat],
                    reduction="mean",
                )
            else:
                z = torch.tensor(0.0, device=future_traj.device, dtype=future_traj.dtype)
                traj_loss = z
                goal_loss = z
                mode_cls = z

            # Calibrate veto cost with realized trajectory error (detached target).
            target_cost = dist.detach()
            denom = target_cost.mean(dim=-1, keepdim=True).clamp_min(1e-6)
            target_cost = target_cost / denom
            if valid_k.any():
                veto_loss = F.smooth_l1_loss(veto_cost[valid_k], target_cost[valid_k], reduction="mean")
            else:
                veto_loss = torch.tensor(0.0, device=future_traj.device, dtype=future_traj.dtype)

            total = (
                traj_loss
                + self.lambda_goal * goal_loss
                + self.lambda_mode_cls * mode_cls
                + self.lambda_veto * veto_loss
            )
            outputs["diffusion_loss"] = total
            outputs["loss_route_head"] = traj_loss
            outputs["loss_goal_head"] = goal_loss
            outputs["loss_mode_cls"] = mode_cls
            outputs["loss_veto"] = veto_loss

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
        goal_xy, goal_logits = self._goal_decode(
            outputs["fused_conditions"],
            outputs["local_contexts"],
            outputs["graph_embeddings"],
        )
        traj_modes, mode_logits, _, veto_cost, _, _ = self._route_decode_with_veto(
            outputs["fused_conditions"],
            outputs["intent_priors"],
            outputs["interaction_features"],
            outputs["local_contexts"],
            outputs["graph_embeddings"],
            goal_xy,
            goal_logits,
            outputs["kg_positions"],
            num_points=int(num_points),
        )

        # expose confidence for debug if needed
        _ = torch.exp(-veto_cost)

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


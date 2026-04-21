"""
DIGIR variant: Goal-to-Trajectory Cascaded Decoding with Differentiable Soft-Routing.

Compared to hard Top-K cascade:
- Stage-1 still predicts many goal candidates.
- Stage-2 no longer hard-selects Top-K goals.
  Each routing query softly attends all goals (or Gumbel-Softmax straight-through),
  so routing loss gradients can flow back to goal decoder.

This file does NOT overwrite the original DIGIR implementation.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.digir import DIGIR as BaseDIGIR


class DIGIR(BaseDIGIR):
    """
    Cascaded decoder with differentiable soft routing.
    """

    def __init__(self, config):
        super().__init__(config)

        self.goal_query_count = int(config.get("goal_query_count", 32))
        self.route_modes = int(config.get("route_modes", 6))
        self.route_temperature = float(config.get("route_temperature", 1.0))

        # Soft-routing controls.
        self.use_gumbel = bool(config.get("soft_route_use_gumbel", True))
        self.gumbel_tau = float(config.get("soft_route_gumbel_tau", 0.8))
        self.gumbel_hard = bool(config.get("soft_route_gumbel_hard", True))

        # Loss weights.
        self.lambda_goal = float(config.get("lambda_goal", 0.5))
        self.lambda_mode_cls = float(config.get("lambda_mode_cls", 0.3))
        self.lambda_goal_aux = float(config.get("lambda_goal_aux", 0.2))

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

        # -------- Stage 2: Routing Decoder (soft routing) --------
        self.route_query_tokens = nn.Parameter(torch.randn(self.route_modes, d) * 0.02)
        self.route_seed_proj = nn.Linear(d, d)
        self.route_goal_xy_proj = nn.Linear(2, d)

        self.route_goal_attn = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.route_goal_norm = nn.LayerNorm(d)
        self.route_goal_ffn = nn.Sequential(
            nn.Linear(d, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d),
        )
        self.route_goal_ffn_norm = nn.LayerNorm(d)

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
        traj_modes: (B,N,K,T,2)
        mode_idx: (B,N)
        return: (B,N,T,2)
        """
        b, n, _, t, c = traj_modes.shape
        idx = mode_idx.view(b, n, 1, 1, 1).expand(b, n, 1, t, c)
        return traj_modes.gather(2, idx).squeeze(2)

    @staticmethod
    def _gather_goal(goal_modes, mode_idx):
        """
        goal_modes: (B,N,K,2)
        mode_idx: (B,N)
        return: (B,N,2)
        """
        b, n, _, c = goal_modes.shape
        idx = mode_idx.view(b, n, 1, 1).expand(b, n, 1, c)
        return goal_modes.gather(2, idx).squeeze(2)

    def _sample_mode_index(self, mode_logits, bestof=False):
        """
        mode_logits: (B,N,K) -> (B,N)
        """
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
        }

    def _goal_decode(self, fused_conditions, local_contexts, graph_embeddings):
        """
        Stage 1 goal decoder.
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

    def _soft_goal_assignment(self, route_q, goal_h, goal_xy, goal_logits):
        """
        Differentiable soft routing from route queries to all goals.

        Args:
            route_q: (BN,K,D)
            goal_h: (BN,Q,D)
            goal_xy: (BN,Q,2)
            goal_logits: (BN,Q)
        Returns:
            goal_weights: (BN,K,Q)
            soft_goal_h: (BN,K,D)
            soft_goal_xy: (BN,K,2)
            goal_logit_expect: (BN,K)
        """
        d = route_q.shape[-1]
        # compatibility + goal prior
        compat = torch.matmul(route_q, goal_h.transpose(1, 2)) / math.sqrt(float(d))  # (BN,K,Q)
        compat = compat + goal_logits.unsqueeze(1)

        if self.use_gumbel and self.training:
            w = F.gumbel_softmax(
                compat,
                tau=max(self.gumbel_tau, 1e-4),
                hard=self.gumbel_hard,
                dim=-1,
            )
        else:
            tau = max(self.gumbel_tau if self.use_gumbel else 1.0, 1e-4)
            w = torch.softmax(compat / tau, dim=-1)
        w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        soft_goal_h = torch.matmul(w, goal_h)   # (BN,K,D)
        soft_goal_xy = torch.matmul(w, goal_xy) # (BN,K,2)
        goal_logit_expect = torch.matmul(w, goal_logits.unsqueeze(-1)).squeeze(-1)  # (BN,K)
        return w, soft_goal_h, soft_goal_xy, goal_logit_expect

    def _route_decode_soft(
        self,
        fused_conditions,
        intent_priors,
        interaction_features,
        local_contexts,
        graph_embeddings,
        goal_h,
        goal_xy,
        goal_logits,
        num_points,
    ):
        """
        Stage 2 routing decoder with differentiable soft routing.
        Returns:
            traj_modes: (B,N,K,T,2)
            mode_logits: (B,N,K)
            soft_goals: (B,N,K,2)
            goal_weights: (B,N,K,Q)
        """
        b, n, d = fused_conditions.shape
        k = self.route_modes
        q = goal_h.shape[2]
        bn = b * n

        # Route seed queries (mode-specific).
        base = self.route_query_tokens.view(1, 1, k, d).expand(b, n, k, d)
        route_seed = (
            base
            + fused_conditions.unsqueeze(2)
            + 0.5 * intent_priors.unsqueeze(2)
            + 0.5 * interaction_features.unsqueeze(2)
        )
        route_seed = self.route_seed_proj(route_seed).reshape(bn, k, d)  # (BN,K,D)

        # Goal memory per agent.
        goal_h_bn = goal_h.reshape(bn, q, d)
        goal_xy_bn = goal_xy.reshape(bn, q, 2)
        goal_logits_bn = goal_logits.reshape(bn, q)

        # Cross-attn route->all-goals (differentiable).
        attn_out, _ = self.route_goal_attn(route_seed, goal_h_bn, goal_h_bn, need_weights=False)
        route_q = self.route_goal_norm(route_seed + attn_out)
        route_q = self.route_goal_ffn_norm(route_q + self.route_goal_ffn(route_q))

        # Soft (or ST-Gumbel) assignment over all goals.
        goal_w, soft_goal_h, soft_goal_xy, goal_logit_expect = self._soft_goal_assignment(
            route_q, goal_h_bn, goal_xy_bn, goal_logits_bn
        )

        # Fuse soft goal condition into route query.
        route_q = route_q + soft_goal_h + self.route_goal_xy_proj(soft_goal_xy)

        # Route transformer decode.
        tgt = route_q  # (BN,K,D)
        map_pool = graph_embeddings.mean(dim=1).unsqueeze(1).unsqueeze(1).expand(b, n, 1, d).reshape(bn, 1, d)
        mem_tokens = torch.stack(
            [fused_conditions, intent_priors, interaction_features, local_contexts],
            dim=2,
        ).reshape(bn, 4, d)
        memory = torch.cat([mem_tokens, map_pool], dim=1)

        route_h = self.route_decoder(tgt=tgt, memory=memory)  # (BN,K,D)
        traj_modes = self.route_traj_head(route_h).view(b, n, k, self.prediction_horizon, 2)
        traj_modes = self._resize_traj(traj_modes, num_points=int(num_points))

        soft_goals = soft_goal_xy.view(b, n, k, 2)

        # Endpoint alignment to soft selected goals.
        goal_expand = soft_goals.unsqueeze(-2)  # (B,N,K,1,2)
        alpha = torch.linspace(
            0.0, 1.0, int(num_points), device=traj_modes.device, dtype=traj_modes.dtype
        ).view(1, 1, 1, int(num_points), 1)
        traj_modes = traj_modes + alpha * (goal_expand - traj_modes[..., -1:, :])

        route_logits = self.route_score_head(route_h).view(b, n, k)
        mode_logits = route_logits + goal_logit_expect.view(b, n, k)
        goal_weights = goal_w.view(b, n, k, q)
        return traj_modes, mode_logits, soft_goals, goal_weights

    def _goal_aux_losses(self, goal_xy, goal_logits, goal_gt, valid_mask):
        """
        Auxiliary objective for stage-1 goal decoder.
        goal_xy: (B,N,Q,2)
        goal_logits: (B,N,Q)
        goal_gt: (B,N,2)
        valid_mask: (B,N) bool
        """
        b, n, q, _ = goal_xy.shape
        d = torch.norm(goal_xy - goal_gt.unsqueeze(2), dim=-1)  # (B,N,Q)
        closest_idx = d.argmin(dim=-1)  # (B,N)
        closest_goal = self._gather_goal(goal_xy, closest_idx)  # (B,N,2)

        if valid_mask.any():
            valid_flat = valid_mask.view(-1)
            cls = F.cross_entropy(
                goal_logits.view(b * n, q)[valid_flat],
                closest_idx.view(-1)[valid_flat],
                reduction="mean",
            )
            reg = F.smooth_l1_loss(closest_goal[valid_mask], goal_gt[valid_mask], reduction="mean")
        else:
            z = torch.tensor(0.0, device=goal_xy.device, dtype=goal_xy.dtype)
            cls = z
            reg = z
        return cls, reg

    def forward(self, trajectories, kg_data, future_traj=None, mode="train", vehicle_masks=None):
        """
        Interface-compatible forward.
        In train mode:
        - outputs["diffusion_loss"] is repurposed as soft-cascade training loss.
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
            traj_modes, mode_logits, soft_goals, goal_weights = self._route_decode_soft(
                outputs["fused_conditions"],
                outputs["intent_priors"],
                outputs["interaction_features"],
                outputs["local_contexts"],
                outputs["graph_embeddings"],
                goal_h,
                goal_xy,
                goal_logits,
                num_points=t,
            )
            outputs["traj_modes"] = traj_modes
            outputs["mode_logits"] = mode_logits
            outputs["soft_goals"] = soft_goals
            outputs["goal_weights"] = goal_weights

            # Best-of-K supervision.
            dist = torch.norm(traj_modes - future_traj.unsqueeze(2), dim=-1).mean(dim=-1)  # (B,N,K)
            best_idx = dist.argmin(dim=-1)  # (B,N)
            best_traj = self._gather_mode(traj_modes, best_idx)  # (B,N,T,2)
            best_goal = self._gather_goal(soft_goals, best_idx)  # (B,N,2)
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
                    mode_logits.view(b * n, self.route_modes)[valid_flat],
                    best_idx.view(-1)[valid_flat],
                    reduction="mean",
                )
            else:
                z = torch.tensor(0.0, device=future_traj.device, dtype=future_traj.dtype)
                traj_loss = z
                goal_loss = z
                mode_cls = z

            goal_gt = future_traj[:, :, -1, :]
            goal_aux_cls, goal_aux_reg = self._goal_aux_losses(
                goal_xy, goal_logits, goal_gt, valid
            )
            goal_aux = goal_aux_cls + goal_aux_reg

            total = (
                traj_loss
                + self.lambda_goal * goal_loss
                + self.lambda_mode_cls * mode_cls
                + self.lambda_goal_aux * goal_aux
            )
            outputs["diffusion_loss"] = total
            outputs["loss_route_head"] = traj_loss
            outputs["loss_goal_head"] = goal_loss
            outputs["loss_mode_cls"] = mode_cls
            outputs["loss_goal_aux_cls"] = goal_aux_cls
            outputs["loss_goal_aux_reg"] = goal_aux_reg

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
        del sampling, step  # not used by this non-diffusion variant

        outputs = self._build_backbone(trajectories, kg_data, vehicle_masks=vehicle_masks)
        goal_h, goal_xy, goal_logits = self._goal_decode(
            outputs["fused_conditions"],
            outputs["local_contexts"],
            outputs["graph_embeddings"],
        )
        traj_modes, mode_logits, _, _ = self._route_decode_soft(
            outputs["fused_conditions"],
            outputs["intent_priors"],
            outputs["interaction_features"],
            outputs["local_contexts"],
            outputs["graph_embeddings"],
            goal_h,
            goal_xy,
            goal_logits,
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


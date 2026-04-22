"""
DIGIR variant: Goal-to-Trajectory Cascaded Decoding with Spatio-Temporal
Decoupled Bezier routing.

Upgrade over standard goal cascade:
- Path Head: predicts spatial Bezier control points (geometry only).
- Velocity Head: predicts monotonic s(t) profile (timing only).
- Final trajectory is composed by Y(t) = Path(s(t)).

This file does NOT overwrite existing model scripts.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.digir import DIGIR as BaseDIGIR


class DIGIR(BaseDIGIR):
    """
    Cascaded decoder + spatio-temporal decoupled Bezier routing.
    """

    def __init__(self, config):
        super().__init__(config)

        self.goal_query_count = int(config.get("goal_query_count", 32))
        self.route_modes = int(config.get("route_modes", 6))
        self.route_temperature = float(config.get("route_temperature", 1.0))

        # Spatio-temporal decoupled heads.
        self.bezier_ctrl_points = max(2, int(config.get("bezier_ctrl_points", 4)))
        self.speed_min_delta = float(config.get("speed_min_delta", 1e-3))

        # Loss weights for cascade head.
        self.lambda_goal = float(config.get("lambda_goal", 0.5))
        self.lambda_mode_cls = float(config.get("lambda_mode_cls", 0.3))
        self.lambda_speed_smooth = float(config.get("lambda_speed_smooth", 0.1))
        self.lambda_accel = float(config.get("lambda_accel", 0.05))

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

        # Path head (space): Bezier control points.
        self.path_ctrl_head = nn.Linear(d, self.bezier_ctrl_points * 2)
        # Velocity head (time): station increments for monotonic s(t).
        self.speed_delta_head = nn.Linear(d, self.prediction_horizon)
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

    @staticmethod
    def _gather_profile(profile_modes, mode_idx):
        """
        profile_modes: (B,N,K,T)
        mode_idx: (B,N)
        return: (B,N,T)
        """
        b, n, _, t = profile_modes.shape
        idx = mode_idx.view(b, n, 1, 1).expand(b, n, 1, t)
        return profile_modes.gather(2, idx).squeeze(2)

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

        flat = probs.view(-1, k)
        idx = torch.multinomial(flat, num_samples=1).view(mode_logits.shape[0], mode_logits.shape[1])
        return idx

    @staticmethod
    def _resize_time_last(x, target_t):
        """
        Resize tensor on the last dimension by linear interpolation.
        """
        t0 = int(x.shape[-1])
        if t0 == int(target_t):
            return x
        flat = x.reshape(-1, 1, t0)
        flat = F.interpolate(flat, size=int(target_t), mode="linear", align_corners=True)
        return flat.view(*x.shape[:-1], int(target_t))

    def _build_backbone(self, trajectories, kg_data, vehicle_masks=None):
        """
        Build DIGIR dual-granularity prompt/state with explicit graph embeddings.
        """
        motion_summaries = self.traj_encoder(trajectories)
        if vehicle_masks is not None:
            motion_summaries = motion_summaries * vehicle_masks.unsqueeze(-1).float()

        graph_embeddings = self.graph_encoder(
            kg_data["facility_types"],
            kg_data["positions"],
            kg_data["edge_index"],
            kg_data.get("edge_types"),
        )  # (B, M, D)

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

    @staticmethod
    def _bezier_curve(control_points, s_values):
        """
        Evaluate Bezier curve with arbitrary degree.

        control_points: (B,N,K,P,2)
        s_values: (B,N,K,T) in [0,1]
        return: (B,N,K,T,2)
        """
        p = control_points.shape[-2]
        degree = p - 1
        s = s_values.unsqueeze(-1)  # (B,N,K,T,1)
        one_minus = 1.0 - s

        basis_terms = []
        for i in range(p):
            coef = float(math.comb(degree, i))
            term = coef * (s ** i) * (one_minus ** (degree - i))
            basis_terms.append(term)
        basis = torch.cat(basis_terms, dim=-1)  # (B,N,K,T,P)

        return torch.einsum("bnktp,bnkpc->bnktc", basis, control_points)

    def _build_control_points(self, route_h_view, top_goals):
        """
        route_h_view: (B,N,K,D)
        top_goals: (B,N,K,2)
        return:
            control_points: (B,N,K,P,2)
        """
        b, n, k, _ = route_h_view.shape
        p = self.bezier_ctrl_points

        raw = self.path_ctrl_head(route_h_view).view(b, n, k, p, 2)
        start = torch.zeros_like(top_goals).unsqueeze(-2)  # local future starts near origin
        end = top_goals.unsqueeze(-2)

        if p == 2:
            return torch.cat([start, end], dim=-2)

        # Keep inner points learnable, but softly anchored to straight-line initialization.
        inner = raw[..., 1:-1, :]  # (B,N,K,P-2,2)
        if inner.shape[-2] > 0:
            frac = torch.linspace(
                1.0 / float(p - 1),
                float(p - 2) / float(p - 1),
                steps=p - 2,
                device=inner.device,
                dtype=inner.dtype,
            ).view(1, 1, 1, p - 2, 1)
            anchor = frac * top_goals.unsqueeze(-2)
            inner = anchor + 0.5 * inner

        return torch.cat([start, inner, end], dim=-2)

    def _build_speed_profile(self, route_h_view, num_points):
        """
        route_h_view: (B,N,K,D)
        return:
            s_profile: (B,N,K,T), monotonic in [0,1]
            delta_s: (B,N,K,T), positive increments before normalization
        """
        raw = self.speed_delta_head(route_h_view)  # (B,N,K,T0)
        raw = self._resize_time_last(raw, target_t=int(num_points))
        delta_s = F.softplus(raw) + self.speed_min_delta
        s_profile = torch.cumsum(delta_s, dim=-1)
        s_profile = s_profile / s_profile[..., -1:].clamp_min(1e-6)
        return s_profile, delta_s

    def _route_decode_bezier(
        self,
        fused_conditions,
        intent_priors,
        interaction_features,
        local_contexts,
        graph_embeddings,
        goal_xy,
        goal_logits,
        num_points,
    ):
        """
        Stage 2 routing decoder with spatial-temporal decoupling.
        Returns:
            traj_modes: (B,N,K,T,2)
            mode_logits: (B,N,K)
            top_goals: (B,N,K,2)
            control_points: (B,N,K,P,2)
            s_profile: (B,N,K,T)
        """
        b, n, d = fused_conditions.shape
        k = min(self.route_modes, goal_xy.shape[2])

        top = goal_logits.topk(k, dim=-1, largest=True, sorted=True)
        top_goal_indices = top.indices  # (B,N,K)
        top_goal_logits = top.values    # (B,N,K)
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

        route_h = self.route_decoder(tgt=tgt, memory=memory).view(b, n, k, d)

        # Space path: Bezier control points.
        control_points = self._build_control_points(route_h, top_goals)
        # Time profile: monotonic s(t).
        s_profile, _ = self._build_speed_profile(route_h, num_points=int(num_points))
        # Compose final trajectory.
        traj_modes = self._bezier_curve(control_points, s_profile)

        route_logits = self.route_score_head(route_h).view(b, n, k)
        mode_logits = top_goal_logits + route_logits
        return traj_modes, mode_logits, top_goals, control_points, s_profile

    @staticmethod
    def _masked_mean(x, mask, zero):
        """
        x: (...), mask broadcastable to x
        """
        if mask.any():
            return x[mask].mean()
        return zero

    def forward(self, trajectories, kg_data, future_traj=None, mode="train", vehicle_masks=None):
        """
        Interface-compatible forward.
        In train mode:
        - outputs['diffusion_loss'] is repurposed as cascade head training loss.
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

            traj_modes, mode_logits, top_goals, control_points, s_profile = self._route_decode_bezier(
                outputs["fused_conditions"],
                outputs["intent_priors"],
                outputs["interaction_features"],
                outputs["local_contexts"],
                outputs["graph_embeddings"],
                goal_xy,
                goal_logits,
                num_points=t,
            )
            outputs["traj_modes"] = traj_modes
            outputs["mode_logits"] = mode_logits
            outputs["bezier_control_points"] = control_points
            outputs["s_profile"] = s_profile

            # Best-of-K supervision.
            dist = torch.norm(traj_modes - future_traj.unsqueeze(2), dim=-1).mean(dim=-1)  # (B,N,K)
            best_idx = dist.argmin(dim=-1)  # (B,N)
            best_traj = self._gather_mode(traj_modes, best_idx)  # (B,N,T,2)
            best_goal = self._gather_goal(top_goals, best_idx)   # (B,N,2)
            best_s = self._gather_profile(s_profile, best_idx)   # (B,N,T)
            outputs["traj_pred_train"] = best_traj
            outputs["best_mode_idx"] = best_idx

            if vehicle_masks is None:
                valid = torch.ones((b, n), dtype=torch.bool, device=future_traj.device)
            else:
                valid = vehicle_masks.bool()
            valid_flat = valid.view(-1)

            zero = torch.tensor(0.0, device=future_traj.device, dtype=future_traj.dtype)
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
                traj_loss = zero
                goal_loss = zero
                mode_cls = zero

            # Velocity-profile smoothness regularization on s(t).
            if t >= 3:
                ds = best_s[..., 1:] - best_s[..., :-1]          # (B,N,T-1)
                d2s = ds[..., 1:] - ds[..., :-1]                 # (B,N,T-2)
                mask_d2s = valid.unsqueeze(-1).expand_as(d2s)
                speed_smooth = self._masked_mean(d2s.abs(), mask_d2s, zero)
            else:
                speed_smooth = zero

            # Physical smoothness: penalize large acceleration magnitude.
            if t >= 3:
                vel = best_traj[..., 1:, :] - best_traj[..., :-1, :]   # (B,N,T-1,2)
                acc = vel[..., 1:, :] - vel[..., :-1, :]                # (B,N,T-2,2)
                acc_norm = torch.norm(acc, dim=-1)                      # (B,N,T-2)
                mask_acc = valid.unsqueeze(-1).expand_as(acc_norm)
                accel_loss = self._masked_mean(acc_norm, mask_acc, zero)
            else:
                accel_loss = zero

            total = (
                traj_loss
                + self.lambda_goal * goal_loss
                + self.lambda_mode_cls * mode_cls
                + self.lambda_speed_smooth * speed_smooth
                + self.lambda_accel * accel_loss
            )
            outputs["diffusion_loss"] = total
            outputs["loss_route_head"] = traj_loss
            outputs["loss_goal_head"] = goal_loss
            outputs["loss_mode_cls"] = mode_cls
            outputs["loss_speed_smooth"] = speed_smooth
            outputs["loss_accel"] = accel_loss

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
        goal_xy, goal_logits = self._goal_decode(
            outputs["fused_conditions"],
            outputs["local_contexts"],
            outputs["graph_embeddings"],
        )
        traj_modes, mode_logits, _, _, _ = self._route_decode_bezier(
            outputs["fused_conditions"],
            outputs["intent_priors"],
            outputs["interaction_features"],
            outputs["local_contexts"],
            outputs["graph_embeddings"],
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


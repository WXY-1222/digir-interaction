"""
DIGIR variant: goal-conditioned query decoder (no diffusion).

Design goals:
- Keep training/evaluation interface compatible with train_digir_full.py.
- Replace diffusion generation with a query-based planning head.
- Preserve existing scene/intent/interaction modules to reuse your current pipeline.

This file does NOT overwrite the original DIGIR implementation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.digir import DIGIR as BaseDIGIR
from models.digir_local_map import LocalSparseContextExtractor


class DIGIR(BaseDIGIR):
    """
    Non-diffusion DIGIR:
    - scene/intention/interaction backbone stays
    - trajectory generation uses query-conditioned goal + control-point decoding
    """

    def __init__(self, config):
        super().__init__(config)

        self.num_modes = int(config.get("num_modes", 6))
        self.mode_temperature = float(config.get("mode_temperature", 1.0))
        self.residual_scale = float(config.get("plan_residual_scale", 0.40))
        self.lambda_mode_cls = float(config.get("lambda_mode_cls", 0.30))
        self.lambda_diversity = float(config.get("lambda_diversity", 0.03))
        self.diversity_margin = float(config.get("diversity_margin", 1.5))

        # Keep the improved local sparse map-attention for map conditioning.
        self.local_context_extractor = LocalSparseContextExtractor(
            d_model=self.d_model,
            num_heads=config.get("num_heads", 4),
            dropout=config.get("dropout", 0.1),
            local_topk=config.get("local_map_topk", 24),
            local_radius=config.get("local_map_radius", None),
        )

        # Query-conditioned planning head.
        self.mode_queries = nn.Parameter(torch.randn(self.num_modes, self.d_model) * 0.02)
        self.intent_to_query = nn.Linear(self.d_model, self.d_model)
        self.interaction_to_query = nn.Linear(self.d_model, self.d_model)
        self.query_norm = nn.LayerNorm(self.d_model)

        self.query_mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
        )

        # Decode two inner Bezier control points + endpoint goal.
        # Output order: [p1_x, p1_y, p2_x, p2_y, goal_x, goal_y]
        self.ctrl_head = nn.Linear(self.d_model, 6)

        # Residual per-timestep offsets to increase flexibility.
        self.residual_head = nn.Linear(self.d_model, self.prediction_horizon * 2)

        # Per-mode score.
        self.mode_score_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.d_model // 2, 1),
        )

        # This variant does not use diffusion.
        self.diffusion = None

    def encode_scene(self, trajectories, kg_data, vehicle_masks=None):
        """
        Override scene encoding to use local sparse map cross-attention.
        """
        batch_size, _, _, _ = trajectories.shape

        motion_summaries = self.traj_encoder(trajectories)
        if vehicle_masks is not None:
            motion_summaries = motion_summaries * vehicle_masks.unsqueeze(-1).float()

        if self.ablate_cross_attn:
            local_contexts = motion_summaries
        else:
            graph_embeddings = self.graph_encoder(
                kg_data["facility_types"],
                kg_data["positions"],
                kg_data["edge_index"],
                kg_data.get("edge_types"),
            )
            graph_pos = kg_data["positions"]
            if graph_pos.dim() == 2:
                graph_pos = graph_pos.unsqueeze(0).expand(batch_size, -1, -1)

            vehicle_xy = trajectories[:, :, -1, :2]
            local_contexts = self.local_context_extractor(
                vehicle_state=motion_summaries,
                graph_embeddings=graph_embeddings,
                graph_positions=graph_pos,
                vehicle_xy=vehicle_xy,
                vehicle_mask=vehicle_masks,
            )

        scene_intent, local_contexts = self.scene_transformer(
            local_contexts,
            vehicle_mask=vehicle_masks,
        )
        return scene_intent, local_contexts, motion_summaries

    def _build_bezier_trajectory(self, ctrl_params, num_points):
        """
        Build smooth trajectories from cubic Bezier control points.

        Args:
            ctrl_params: (..., 6) => p1(2), p2(2), goal(2)
            num_points: prediction horizon
        Returns:
            traj: (..., T, 2)
        """
        p1 = ctrl_params[..., 0:2]
        p2 = ctrl_params[..., 2:4]
        p3 = ctrl_params[..., 4:6]  # goal

        device = ctrl_params.device
        dtype = ctrl_params.dtype

        t = torch.linspace(1.0 / float(num_points), 1.0, num_points, device=device, dtype=dtype)
        expand_shape = [1] * (ctrl_params.dim() - 1) + [num_points, 1]
        t = t.view(*expand_shape)

        one_minus_t = 1.0 - t

        # P0 is fixed at origin (current position in local frame).
        traj = (
            3.0 * (one_minus_t ** 2) * t * p1.unsqueeze(-2)
            + 3.0 * one_minus_t * (t ** 2) * p2.unsqueeze(-2)
            + (t ** 3) * p3.unsqueeze(-2)
        )
        return traj

    def _resize_residual(self, residual, num_points):
        """
        Residual is decoded at training horizon by default; interpolate if needed.
        residual: (B, N, K, T0, 2)
        """
        t0 = residual.shape[-2]
        if t0 == num_points:
            return residual

        b, n, k, _, c = residual.shape
        r = residual.permute(0, 1, 2, 4, 3).reshape(b * n * k, c, t0)
        r = F.interpolate(r, size=num_points, mode="linear", align_corners=True)
        r = r.view(b, n, k, c, num_points).permute(0, 1, 2, 4, 3)
        return r

    def _decode_modes(self, fused_conditions, intent_priors, interaction_features, num_points):
        """
        Decode K mode trajectories and mode logits.

        Returns:
            traj_modes: (B, N, K, T, 2)
            mode_logits: (B, N, K)
        """
        b, n, _ = fused_conditions.shape
        k = self.num_modes

        q = self.mode_queries.view(1, 1, k, self.d_model)
        intent_bias = self.intent_to_query(intent_priors).unsqueeze(2)
        inter_bias = self.interaction_to_query(interaction_features).unsqueeze(2)
        base = fused_conditions.unsqueeze(2)

        query_ctx = self.query_norm(base + q + 0.5 * intent_bias + 0.5 * inter_bias)
        h = self.query_mlp(query_ctx)

        ctrl_params = self.ctrl_head(h)  # (B, N, K, 6)
        bezier_traj = self._build_bezier_trajectory(ctrl_params, num_points=num_points)

        residual = self.residual_head(h).view(
            b, n, k, self.prediction_horizon, 2
        )  # decoded at canonical horizon
        residual = self._resize_residual(residual, num_points=num_points)
        residual = torch.tanh(residual) * self.residual_scale

        traj_modes = bezier_traj + residual

        # Light endpoint alignment so final point follows decoded goal.
        goal = ctrl_params[..., 4:6].unsqueeze(-2)  # (B, N, K, 1, 2)
        alpha = torch.linspace(
            0.0, 1.0, num_points, device=traj_modes.device, dtype=traj_modes.dtype
        ).view(1, 1, 1, num_points, 1)
        traj_modes = traj_modes + alpha * (goal - traj_modes[..., -1:, :])

        mode_logits = self.mode_score_head(h).squeeze(-1)
        return traj_modes, mode_logits

    @staticmethod
    def _gather_one_mode(traj_modes, mode_idx):
        """
        Gather trajectory by mode index.
        traj_modes: (B, N, K, T, 2)
        mode_idx: (B, N)
        return: (B, N, T, 2)
        """
        b, n, _, t, c = traj_modes.shape
        idx = mode_idx.view(b, n, 1, 1, 1).expand(b, n, 1, t, c)
        return traj_modes.gather(2, idx).squeeze(2)

    def _sample_mode_index(self, mode_logits, bestof=False):
        """
        mode_logits: (B, N, K)
        return mode_idx: (B, N)
        """
        if bestof:
            return mode_logits.argmax(dim=-1)

        k = mode_logits.shape[-1]
        temp = max(float(self.mode_temperature), 1e-4)
        probs = torch.softmax(mode_logits / temp, dim=-1)
        probs = torch.nan_to_num(probs, nan=1.0 / k, posinf=1.0 / k, neginf=1.0 / k)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        flat = probs.view(-1, k)
        idx = torch.multinomial(flat, num_samples=1).view(mode_logits.shape[0], mode_logits.shape[1])
        return idx

    def _diversity_loss(self, traj_modes, vehicle_masks=None):
        """
        Encourage endpoint diversity across modes.
        traj_modes: (B, N, K, T, 2)
        """
        k = traj_modes.shape[2]
        if k < 2:
            return torch.tensor(0.0, device=traj_modes.device, dtype=traj_modes.dtype)

        endpoints = traj_modes[..., -1, :]  # (B, N, K, 2)
        pair_dist = torch.norm(
            endpoints.unsqueeze(3) - endpoints.unsqueeze(2), dim=-1
        )  # (B, N, K, K)

        tri = torch.triu_indices(k, k, offset=1, device=traj_modes.device)
        pair_dist = pair_dist[:, :, tri[0], tri[1]]  # (B, N, K*(K-1)/2)
        penalty = F.relu(self.diversity_margin - pair_dist)

        if vehicle_masks is not None:
            valid = vehicle_masks.bool().unsqueeze(-1).expand_as(penalty)
            if valid.any():
                return penalty[valid].mean()
            return torch.tensor(0.0, device=traj_modes.device, dtype=traj_modes.dtype)

        return penalty.mean()

    def forward(self, trajectories, kg_data, future_traj=None, mode="train", vehicle_masks=None):
        """
        Interface-compatible forward.
        - In train mode, outputs['diffusion_loss'] is repurposed as planning-head loss.
        """
        scene_intent, local_contexts, motion_summaries = self.encode_scene(
            trajectories, kg_data, vehicle_masks=vehicle_masks
        )
        intent_priors, intent_logits = self.cross_granularity_mapping(
            scene_intent, local_contexts, vehicle_masks=vehicle_masks
        )
        fused_conditions, interaction_features, gate_weights = self.agent_level_modeling(
            motion_summaries, intent_priors, vehicle_masks=vehicle_masks
        )

        outputs = {
            "scene_intent": scene_intent,
            "local_contexts": local_contexts,
            "motion_summaries": motion_summaries,
            "intent_priors": intent_priors,
            "intent_logits": intent_logits,
            "interaction_features": interaction_features,
            "fused_conditions": fused_conditions,
            "gate_weights": gate_weights,
        }

        if mode == "train" and future_traj is not None:
            b, n, t, _ = future_traj.shape

            traj_modes, mode_logits = self._decode_modes(
                fused_conditions=fused_conditions,
                intent_priors=intent_priors,
                interaction_features=interaction_features,
                num_points=t,
            )  # (B, N, K, T, 2), (B, N, K)
            outputs["mode_logits"] = mode_logits

            dist = torch.norm(traj_modes - future_traj.unsqueeze(2), dim=-1)  # (B, N, K, T)
            ade = dist.mean(dim=-1)  # (B, N, K)
            best_idx = ade.argmin(dim=-1)  # (B, N)

            best_traj = self._gather_one_mode(traj_modes, best_idx)  # (B, N, T, 2)
            outputs["traj_pred_train"] = best_traj
            outputs["best_mode_idx"] = best_idx

            if vehicle_masks is None:
                valid = torch.ones((b, n), dtype=torch.bool, device=future_traj.device)
            else:
                valid = vehicle_masks.bool()

            if valid.any():
                traj_loss = F.smooth_l1_loss(best_traj[valid], future_traj[valid], reduction="mean")
                logits_flat = mode_logits.view(b * n, self.num_modes)
                best_flat = best_idx.view(-1)
                valid_flat = valid.view(-1)
                mode_loss = F.cross_entropy(logits_flat[valid_flat], best_flat[valid_flat])
            else:
                z = torch.tensor(0.0, device=future_traj.device, dtype=future_traj.dtype)
                traj_loss = z
                mode_loss = z

            diversity_loss = self._diversity_loss(traj_modes, vehicle_masks=vehicle_masks)
            plan_loss = traj_loss + self.lambda_mode_cls * mode_loss + self.lambda_diversity * diversity_loss

            # Keep compatibility with existing training script and compute_losses().
            outputs["diffusion_loss"] = plan_loss
            outputs["loss_traj_head"] = traj_loss
            outputs["loss_mode_head"] = mode_loss
            outputs["loss_div_head"] = diversity_loss

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
        - num_samples=1: return (B, N, T, 2)
        - num_samples>1: return (S, B, N, T, 2)
        """
        del sampling, step  # unused in this non-diffusion variant

        scene_intent, local_contexts, motion_summaries = self.encode_scene(
            trajectories, kg_data, vehicle_masks=vehicle_masks
        )
        intent_priors, _ = self.cross_granularity_mapping(
            scene_intent, local_contexts, vehicle_masks=vehicle_masks
        )
        fused_conditions, interaction_features, _ = self.agent_level_modeling(
            motion_summaries, intent_priors, vehicle_masks=vehicle_masks
        )
        traj_modes, mode_logits = self._decode_modes(
            fused_conditions=fused_conditions,
            intent_priors=intent_priors,
            interaction_features=interaction_features,
            num_points=int(num_points),
        )  # (B, N, K, T, 2), (B, N, K)

        top = min(int(num_samples), self.num_modes)
        if top <= 1:
            idx = self._sample_mode_index(mode_logits, bestof=bestof)
            return self._gather_one_mode(traj_modes, idx)

        # Multi-sample output: take highest-score modes
        sort_idx = mode_logits.argsort(dim=-1, descending=True)[:, :, :top]  # (B, N, top)
        b, n, _, t, c = traj_modes.shape
        idx = sort_idx.unsqueeze(-1).unsqueeze(-1).expand(b, n, top, t, c)
        top_traj = traj_modes.gather(2, idx)  # (B, N, top, T, 2)
        top_traj = top_traj.permute(2, 0, 1, 3, 4).contiguous()  # (top, B, N, T, 2)
        if bestof:
            return top_traj[0]
        return top_traj


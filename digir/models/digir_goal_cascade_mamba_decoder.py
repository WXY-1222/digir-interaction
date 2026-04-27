"""
DIGIR variant: Goal-to-Trajectory Cascaded Decoding with a Selective
State-Space (Mamba-style) decoder.

Key idea:
- Remove TransformerDecoder from the cascaded decoder.
- Decode goals and future trajectories with lightweight selective SSM blocks.
- Generate future positions as a time-evolving state sequence.

This file does NOT overwrite existing model scripts.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.digir import DIGIR as BaseDIGIR


class SelectiveSSMBlock(nn.Module):
    """
    A compact Mamba-style selective state-space block.

    This is not the CUDA-optimized official mamba-ssm implementation. It keeps the
    same modeling spirit: input-dependent state update, causal depthwise mixing,
    and linear-time sequence processing without self-attention.
    """

    def __init__(self, d_model, expansion=2, dropout=0.1, conv_kernel=3):
        super().__init__()
        self.d_model = int(d_model)
        self.inner_dim = int(d_model) * int(expansion)

        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, self.inner_dim * 2)
        self.dw_conv = nn.Conv1d(
            self.inner_dim,
            self.inner_dim,
            kernel_size=int(conv_kernel),
            padding=int(conv_kernel) - 1,
            groups=self.inner_dim,
        )
        self.dt_proj = nn.Linear(self.inner_dim, self.inner_dim)
        self.b_proj = nn.Linear(self.inner_dim, self.inner_dim)
        self.c_proj = nn.Linear(self.inner_dim, self.inner_dim)
        self.out_proj = nn.Linear(self.inner_dim, d_model)
        self.dropout = nn.Dropout(dropout)

        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x):
        """
        x: (B,T,D)
        return: (B,T,D)
        """
        residual = x
        z = self.norm(x)
        u, gate = self.in_proj(z).chunk(2, dim=-1)

        # Causal local mixing. Conv padding produces extra future positions, so crop.
        u_conv = self.dw_conv(u.transpose(1, 2))[..., : u.shape[1]].transpose(1, 2)
        u_conv = F.silu(u_conv)

        dt = torch.sigmoid(self.dt_proj(u_conv))
        b_t = torch.tanh(self.b_proj(u_conv))
        c_t = torch.sigmoid(self.c_proj(u_conv))

        state = torch.zeros(
            u_conv.shape[0],
            self.inner_dim,
            device=u_conv.device,
            dtype=u_conv.dtype,
        )
        outs = []
        for t in range(u_conv.shape[1]):
            # Input-dependent leaky integration.
            state = (1.0 - dt[:, t]) * state + dt[:, t] * b_t[:, t]
            outs.append(c_t[:, t] * state + u_conv[:, t])
        y = torch.stack(outs, dim=1)
        y = y * torch.sigmoid(gate)
        x = residual + self.dropout(self.out_proj(y))

        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x


class DIGIR(BaseDIGIR):
    """
    Cascaded goal-to-trajectory model with Selective SSM decoders.
    """

    def __init__(self, config):
        super().__init__(config)

        self.goal_query_count = int(config.get("goal_query_count", 32))
        self.route_modes = int(config.get("route_modes", 6))
        self.route_temperature = float(config.get("route_temperature", 1.0))

        self.lambda_goal = float(config.get("lambda_goal", 0.5))
        self.lambda_mode_cls = float(config.get("lambda_mode_cls", 0.3))
        self.lambda_temporal_smooth = float(config.get("lambda_temporal_smooth", 0.05))

        d = self.d_model
        dropout = float(config.get("dropout", 0.1))
        goal_layers = int(config.get("mamba_goal_layers", 2))
        route_layers = int(config.get("mamba_route_layers", 4))
        expansion = int(config.get("mamba_expansion", 2))

        # -------- Stage 1: SSM Goal Decoder --------
        self.goal_queries = nn.Parameter(torch.randn(self.goal_query_count, d) * 0.02)
        self.goal_context_proj = nn.Linear(d * 3, d)
        self.goal_ssm = nn.ModuleList(
            [SelectiveSSMBlock(d, expansion=expansion, dropout=dropout) for _ in range(goal_layers)]
        )
        self.goal_pos_head = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 2),
        )
        self.goal_score_head = nn.Linear(d, 1)

        # -------- Stage 2: SSM Routing Decoder --------
        self.goal_to_route = nn.Sequential(
            nn.Linear(d + 2, d),
            nn.ReLU(),
            nn.Linear(d, d),
        )
        self.route_seed_proj = nn.Linear(d * 4, d)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )
        self.route_ssm = nn.ModuleList(
            [SelectiveSSMBlock(d, expansion=expansion, dropout=dropout) for _ in range(route_layers)]
        )
        self.step_delta_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, 2),
        )
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
        }

    def _goal_decode(self, fused_conditions, local_contexts, graph_embeddings):
        """
        SSM goal decoder.

        Returns:
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
        return goal_xy, goal_logits

    def _route_decode_mamba(
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
        SSM trajectory decoder.

        Returns:
            traj_modes: (B,N,K,T,2)
            mode_logits: (B,N,K)
            top_goals: (B,N,K,2)
        """
        b, n, d = fused_conditions.shape
        k = min(self.route_modes, goal_xy.shape[2])
        t = int(num_points)

        top = goal_logits.topk(k, dim=-1, largest=True, sorted=True)
        top_goal_indices = top.indices
        top_goal_logits = top.values
        gather_idx = top_goal_indices.unsqueeze(-1).expand(b, n, k, 2)
        top_goals = goal_xy.gather(dim=2, index=gather_idx)

        route_goal = self.goal_to_route(
            torch.cat([fused_conditions.unsqueeze(2).expand(-1, -1, k, -1), top_goals], dim=-1)
        )
        map_pool = graph_embeddings.mean(dim=1).unsqueeze(1).unsqueeze(2).expand(b, n, k, d)
        route_base = self.route_seed_proj(
            torch.cat(
                [
                    route_goal,
                    intent_priors.unsqueeze(2).expand(-1, -1, k, -1),
                    interaction_features.unsqueeze(2).expand(-1, -1, k, -1),
                    local_contexts.unsqueeze(2).expand(-1, -1, k, -1) + map_pool,
                ],
                dim=-1,
            )
        )

        time = torch.linspace(0.0, 1.0, steps=t, device=route_base.device, dtype=route_base.dtype)
        time_feat = self.time_mlp(time.view(t, 1)).view(1, 1, 1, t, d)
        seq = route_base.unsqueeze(3) + time_feat
        seq = seq.reshape(b * n * k, t, d)

        for block in self.route_ssm:
            seq = block(seq)

        step_delta = self.step_delta_head(seq).view(b, n, k, t, 2)
        traj_modes = torch.cumsum(step_delta, dim=-2)

        # Endpoint alignment to selected goals keeps the SSM decoder goal-conditioned.
        goal_expand = top_goals.unsqueeze(-2)
        alpha = torch.linspace(
            0.0, 1.0, t, device=traj_modes.device, dtype=traj_modes.dtype
        ).view(1, 1, 1, t, 1)
        traj_modes = traj_modes + alpha * (goal_expand - traj_modes[..., -1:, :])

        final_state = seq[:, -1].view(b, n, k, d)
        route_logits = self.route_score_head(final_state).squeeze(-1)
        mode_logits = top_goal_logits + route_logits
        return traj_modes, mode_logits, top_goals

    @staticmethod
    def _temporal_smooth_loss(best_traj, valid_mask):
        """
        Penalize abrupt acceleration in generated trajectories.
        """
        t = best_traj.shape[-2]
        if t < 3:
            return torch.tensor(0.0, device=best_traj.device, dtype=best_traj.dtype)
        vel = best_traj[..., 1:, :] - best_traj[..., :-1, :]
        acc = vel[..., 1:, :] - vel[..., :-1, :]
        acc_norm = torch.norm(acc, dim=-1)
        mask = valid_mask.unsqueeze(-1).expand_as(acc_norm)
        if not mask.any():
            return torch.tensor(0.0, device=best_traj.device, dtype=best_traj.dtype)
        return acc_norm[mask].mean()

    def forward(self, trajectories, kg_data, future_traj=None, mode="train", vehicle_masks=None):
        """
        Interface-compatible forward.
        In train mode:
        - outputs['diffusion_loss'] is repurposed as SSM decoder training loss.
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
            traj_modes, mode_logits, top_goals = self._route_decode_mamba(
                outputs["fused_conditions"],
                outputs["intent_priors"],
                outputs["interaction_features"],
                outputs["local_contexts"],
                outputs["graph_embeddings"],
                goal_xy,
                goal_logits,
                num_points=t,
            )
            outputs["mode_logits"] = mode_logits
            outputs["traj_modes"] = traj_modes

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
            else:
                z = torch.tensor(0.0, device=future_traj.device, dtype=future_traj.dtype)
                traj_loss = z
                goal_loss = z
                mode_cls = z
                smooth_loss = z

            total = (
                traj_loss
                + self.lambda_goal * goal_loss
                + self.lambda_mode_cls * mode_cls
                + self.lambda_temporal_smooth * smooth_loss
            )
            outputs["diffusion_loss"] = total
            outputs["loss_route_head"] = traj_loss
            outputs["loss_goal_head"] = goal_loss
            outputs["loss_mode_cls"] = mode_cls
            outputs["loss_temporal_smooth"] = smooth_loss

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
        traj_modes, mode_logits, _ = self._route_decode_mamba(
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


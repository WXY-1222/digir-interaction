"""
DIGIR variant: Joint Multi-Agent Interacting Queries (Game-Theoretic Joint Decoder).

Core idea:
- Build per-agent multi-modal queries first.
- Concatenate all agents' queries and run a joint self-attention layer at the
  decoder tail ("last mile"), so cross-agent mode queries can interact directly.
- Add conflict-aware attention bias from provisional endpoints to suppress
  colliding query combinations.

This file does NOT overwrite the original DIGIR implementation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.digir import DIGIR as BaseDIGIR


class DIGIR(BaseDIGIR):
    """
    Joint-decoding DIGIR with cross-agent query interaction.
    """

    def __init__(self, config):
        super().__init__(config)

        self.num_modes = int(config.get("num_modes", 6))
        self.mode_temperature = float(config.get("mode_temperature", 1.0))

        # Conflict-aware attention bias parameters.
        self.conflict_alpha = float(config.get("conflict_alpha", 2.0))
        self.conflict_sigma = float(config.get("conflict_sigma", 2.5))

        # Training loss weights.
        self.lambda_mode_cls = float(config.get("lambda_mode_cls", 0.3))
        self.lambda_joint_col = float(config.get("lambda_joint_col", 0.2))
        self.joint_col_margin = float(config.get("joint_col_margin", 2.0))

        d = self.d_model
        nhead = int(config.get("num_heads", 4))
        dropout = float(config.get("dropout", 0.1))
        ffn_dim = int(config.get("joint_ffn_dim", d * 4))

        # Base per-mode queries (agent-conditional).
        self.mode_queries = nn.Parameter(torch.randn(self.num_modes, d) * 0.02)
        self.intent_to_query = nn.Linear(d, d)
        self.inter_to_query = nn.Linear(d, d)
        self.motion_to_query = nn.Linear(d, d)

        self.query_mlp = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d, d),
            nn.GELU(),
        )
        self.query_norm = nn.LayerNorm(d)

        # Provisional trajectory head used to build conflict-aware attention bias.
        self.pre_traj_head = nn.Linear(d, self.prediction_horizon * 2)

        # Joint query-to-query interaction layer (across all agents and modes).
        self.joint_attn = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.joint_norm1 = nn.LayerNorm(d)
        self.joint_ffn = nn.Sequential(
            nn.Linear(d, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d),
        )
        self.joint_norm2 = nn.LayerNorm(d)

        # Final decoding heads.
        self.traj_head = nn.Linear(d, self.prediction_horizon * 2)
        self.mode_score_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.ReLU(),
            nn.Linear(d // 2, 1),
        )

        # This variant does not use diffusion.
        self.diffusion = None

    @staticmethod
    def _gather_mode(traj_modes, mode_idx):
        """
        traj_modes: (B,N,K,T,2)
        mode_idx: (B,N)
        -> (B,N,T,2)
        """
        b, n, _, t, c = traj_modes.shape
        idx = mode_idx.view(b, n, 1, 1, 1).expand(b, n, 1, t, c)
        return traj_modes.gather(2, idx).squeeze(2)

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

    def _sample_mode_index(self, mode_logits, bestof=False):
        """
        mode_logits: (B,N,K) -> (B,N)
        """
        if bestof:
            return mode_logits.argmax(dim=-1)

        k = mode_logits.shape[-1]
        temp = max(self.mode_temperature, 1e-4)
        probs = torch.softmax(mode_logits / temp, dim=-1)
        probs = torch.nan_to_num(
            probs,
            nan=1.0 / float(k),
            posinf=1.0 / float(k),
            neginf=1.0 / float(k),
        )
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        idx = torch.multinomial(probs.view(-1, k), 1).view(mode_logits.shape[0], mode_logits.shape[1])
        return idx

    def _build_conflict_attn_bias(self, endpoints, num_heads):
        """
        Build additive attention bias for query-to-query self-attention.
        Closer endpoints => larger negative bias => reduced attention probability.

        endpoints: (B, L, 2), where L = N*K
        return:
            attn_bias: (B*num_heads, L, L) float, or None
        """
        if self.conflict_alpha <= 0.0:
            return None

        b, l, _ = endpoints.shape
        sigma = max(self.conflict_sigma, 1e-3)

        dist = torch.cdist(endpoints, endpoints)  # (B,L,L)
        penalty = -self.conflict_alpha * torch.exp(-((dist / sigma) ** 2))  # (B,L,L)

        # Do not penalize self-attention on the diagonal.
        eye = torch.eye(l, dtype=penalty.dtype, device=penalty.device).unsqueeze(0)
        penalty = penalty * (1.0 - eye)

        # Expand to (B*H, L, L) for MultiheadAttention.
        penalty = penalty.unsqueeze(1).expand(b, num_heads, l, l).reshape(b * num_heads, l, l)
        return penalty

    def _joint_collision_loss(self, best_traj, vehicle_masks=None):
        """
        best_traj: (B,N,T,2)
        """
        b, n, _, _ = best_traj.shape
        if vehicle_masks is None:
            valid = torch.ones((b, n), dtype=torch.bool, device=best_traj.device)
        else:
            valid = vehicle_masks.bool()

        total = torch.tensor(0.0, device=best_traj.device, dtype=best_traj.dtype)
        for bi in range(b):
            idx = torch.where(valid[bi])[0]
            nv = int(idx.numel())
            if nv < 2:
                continue
            pos = best_traj[bi, idx]  # (nv,T,2)
            d = torch.norm(pos.unsqueeze(1) - pos.unsqueeze(2), dim=-1)  # (nv,nv,T)
            tri = torch.triu_indices(nv, nv, offset=1, device=best_traj.device)
            if tri.numel() == 0:
                continue
            pen = F.relu(self.joint_col_margin - d[tri[0], tri[1], :]).mean()
            total = total + pen

        return total / max(b, 1)

    def _build_backbone(self, trajectories, kg_data, vehicle_masks=None):
        """
        DIGIR dual-granularity backbone (preserved).
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
        return {
            "scene_intent": scene_intent,
            "local_contexts": local_contexts,
            "motion_summaries": motion_summaries,
            "intent_priors": intent_priors,
            "intent_logits": intent_logits,
            "interaction_features": interaction_features,
            "fused_conditions": fused_conditions,
            "gate_weights": gate_weights,
        }

    def _joint_decode(self, fused_conditions, intent_priors, interaction_features, motion_summaries, vehicle_masks, num_points):
        """
        Build agent-wise mode queries -> joint Q2Q interaction -> decode trajectories.
        Returns:
            traj_modes: (B,N,K,T,2)
            mode_logits: (B,N,K)
        """
        b, n, d = fused_conditions.shape
        k = self.num_modes
        nk = n * k
        nhead = self.joint_attn.num_heads

        # Per-agent multimodal query initialization.
        q_base = self.mode_queries.view(1, 1, k, d)
        q = (
            q_base
            + fused_conditions.unsqueeze(2)
            + 0.4 * self.intent_to_query(intent_priors).unsqueeze(2)
            + 0.4 * self.inter_to_query(interaction_features).unsqueeze(2)
            + 0.2 * self.motion_to_query(motion_summaries).unsqueeze(2)
        )  # (B,N,K,D)
        q = self.query_norm(q + self.query_mlp(q))

        # Provisional endpoint for conflict-aware attention bias.
        pre_traj = self.pre_traj_head(q).view(b, n, k, self.prediction_horizon, 2)
        pre_traj = self._resize_traj(pre_traj, int(num_points))
        pre_end = pre_traj[..., -1, :].reshape(b, nk, 2)  # (B,NK,2)

        # Flatten all agents and modes for joint interaction.
        q_flat = q.reshape(b, nk, d)
        if vehicle_masks is None:
            valid_q = torch.ones((b, nk), dtype=torch.bool, device=q_flat.device)
        else:
            valid_q = vehicle_masks.unsqueeze(-1).expand(-1, -1, k).reshape(b, nk).bool()

        attn_bias = self._build_conflict_attn_bias(pre_end, num_heads=nhead)
        # Merge key-padding behavior into additive attn bias to avoid mixed-mask dtype warnings.
        # invalid keys get a very negative score so they are ignored by softmax.
        if attn_bias is None:
            attn_bias = torch.zeros(
                (b * nhead, nk, nk), dtype=q_flat.dtype, device=q_flat.device
            )
        else:
            attn_bias = attn_bias.to(dtype=q_flat.dtype)
        invalid_key_bias = (~valid_q).to(dtype=q_flat.dtype).unsqueeze(1).unsqueeze(1) * (-1e4)
        attn_bias = (
            attn_bias.view(b, nhead, nk, nk) + invalid_key_bias
        ).reshape(b * nhead, nk, nk)

        # Keep invalid query tokens inert.
        q_flat = q_flat * valid_q.unsqueeze(-1).to(dtype=q_flat.dtype)

        q_attn, _ = self.joint_attn(
            q_flat, q_flat, q_flat,
            attn_mask=attn_bias,
            need_weights=False,
        )
        q_flat = self.joint_norm1(q_flat + q_attn)
        q_flat = self.joint_norm2(q_flat + self.joint_ffn(q_flat))

        q_joint = q_flat.view(b, n, k, d)

        # Final decode.
        traj_modes = self.traj_head(q_joint).view(b, n, k, self.prediction_horizon, 2)
        traj_modes = self._resize_traj(traj_modes, int(num_points))
        mode_logits = self.mode_score_head(q_joint).squeeze(-1)  # (B,N,K)

        # Suppress invalid agents' modes.
        if vehicle_masks is not None:
            invalid = (~vehicle_masks.bool()).unsqueeze(-1).expand_as(mode_logits)
            mode_logits = mode_logits.masked_fill(invalid, -1e4)
            # Explicit 5D broadcast: (B,N,1,1,1) aligns with (B,N,K,T,2).
            traj_modes = traj_modes * vehicle_masks.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float()

        return traj_modes, mode_logits

    def forward(self, trajectories, kg_data, future_traj=None, mode="train", vehicle_masks=None):
        """
        Interface-compatible forward.
        In train mode:
        - outputs["diffusion_loss"] is repurposed as joint-decoder loss.
        """
        outputs = self._build_backbone(trajectories, kg_data, vehicle_masks=vehicle_masks)

        if mode == "train" and future_traj is not None:
            b, n, t, _ = future_traj.shape

            traj_modes, mode_logits = self._joint_decode(
                outputs["fused_conditions"],
                outputs["intent_priors"],
                outputs["interaction_features"],
                outputs["motion_summaries"],
                vehicle_masks,
                num_points=t,
            )
            outputs["traj_modes"] = traj_modes
            outputs["mode_logits"] = mode_logits

            # Best-of-K supervision.
            dist = torch.norm(traj_modes - future_traj.unsqueeze(2), dim=-1).mean(dim=-1)  # (B,N,K)
            best_idx = dist.argmin(dim=-1)  # (B,N)
            best_traj = self._gather_mode(traj_modes, best_idx)  # (B,N,T,2)

            outputs["best_mode_idx"] = best_idx
            outputs["traj_pred_train"] = best_traj

            if vehicle_masks is None:
                valid = torch.ones((b, n), dtype=torch.bool, device=future_traj.device)
            else:
                valid = vehicle_masks.bool()
            valid_flat = valid.reshape(-1)

            if valid.any():
                traj_loss = F.smooth_l1_loss(best_traj[valid], future_traj[valid], reduction="mean")
                mode_loss = F.cross_entropy(
                    mode_logits.view(b * n, self.num_modes)[valid_flat],
                    best_idx.view(-1)[valid_flat],
                    reduction="mean",
                )
            else:
                z = torch.tensor(0.0, device=future_traj.device, dtype=future_traj.dtype)
                traj_loss = z
                mode_loss = z

            joint_col = self._joint_collision_loss(best_traj, vehicle_masks=vehicle_masks)
            joint_loss = traj_loss + self.lambda_mode_cls * mode_loss + self.lambda_joint_col * joint_col

            # Keep compatibility with train_digir_full.py / base compute_losses().
            outputs["diffusion_loss"] = joint_loss
            outputs["loss_joint_traj"] = traj_loss
            outputs["loss_joint_mode"] = mode_loss
            outputs["loss_joint_col"] = joint_col

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
        - num_samples <= 1: (B,N,T,2)
        - num_samples > 1 : (S,B,N,T,2)
        """
        del sampling, step  # unused in this non-diffusion decoder

        outputs = self._build_backbone(trajectories, kg_data, vehicle_masks=vehicle_masks)
        b, n, _, _ = trajectories.shape

        traj_modes, mode_logits = self._joint_decode(
            outputs["fused_conditions"],
            outputs["intent_priors"],
            outputs["interaction_features"],
            outputs["motion_summaries"],
            vehicle_masks,
            num_points=int(num_points),
        )

        s = int(num_samples)
        if s <= 1:
            idx = self._sample_mode_index(mode_logits, bestof=bestof)
            return self._gather_mode(traj_modes, idx)

        k = traj_modes.shape[2]
        preds = []

        if bestof:
            top = min(s, k)
            top_idx = mode_logits.argsort(dim=-1, descending=True)[:, :, :top]  # (B,N,top)
            gather_idx = top_idx.unsqueeze(-1).unsqueeze(-1).expand(b, n, top, int(num_points), 2)
            top_traj = traj_modes.gather(2, gather_idx).permute(2, 0, 1, 3, 4).contiguous()  # (top,B,N,T,2)
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

"""
DIGIR variant: Tree-Structured / Branching Query Decoding.

Motivation:
- Share early trajectory prefix (root query), then branch later in time.
- Better align with human driving intuition: late multimodality instead of
  fully independent trajectories from t=0.

Design:
1) Root query decodes shared prefix segment.
2) Branch-1 queries (e.g., left/keep/right) decode middle segment.
3) Branch-2 queries decode late segment (final multimodality).
4) A causal-masked tree refinement layer is applied in feature space.

This file does NOT overwrite the original DIGIR model.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.digir import DIGIR as BaseDIGIR


class DIGIR(BaseDIGIR):
    """
    Tree-structured decoder on top of DIGIR dual-granularity backbone.
    """

    def __init__(self, config):
        super().__init__(config)

        # Branching factors: 1 -> b1 -> b1*b2 modes
        self.branch_factor_1 = int(config.get("branch_factor_1", 3))
        self.branch_factor_2 = int(config.get("branch_factor_2", 2))
        self.num_modes = self.branch_factor_1 * self.branch_factor_2  # default 6

        # Branch time ratios over horizon (e.g. 2s and 5s on an 8s horizon -> 0.25 / 0.625).
        self.branch_ratio_1 = float(config.get("branch_ratio_1", 0.25))
        self.branch_ratio_2 = float(config.get("branch_ratio_2", 0.625))

        self.delta_scale = float(config.get("tree_delta_scale", 0.6))
        self.route_temperature = float(config.get("route_temperature", 1.0))

        self.lambda_goal = float(config.get("lambda_goal", 0.4))
        self.lambda_mode_cls = float(config.get("lambda_mode_cls", 0.3))

        d = self.d_model
        nhead = int(config.get("num_heads", 4))
        dropout = float(config.get("dropout", 0.1))
        ffn_dim = int(config.get("tree_ffn_dim", d * 4))

        # ---------- Branching attention layers ----------
        self.root_proj = nn.Linear(d, d)
        self.root_norm = nn.LayerNorm(d)

        self.branch_token_1 = nn.Parameter(torch.randn(self.branch_factor_1, d) * 0.02)
        self.branch_attn_1 = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
        self.branch_ffn_1 = nn.Sequential(
            nn.Linear(d, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d),
        )
        self.branch_norm_1a = nn.LayerNorm(d)
        self.branch_norm_1b = nn.LayerNorm(d)

        self.branch_token_2 = nn.Parameter(torch.randn(self.branch_factor_2, d) * 0.02)
        self.branch_attn_2 = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
        self.branch_ffn_2 = nn.Sequential(
            nn.Linear(d, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d),
        )
        self.branch_norm_2a = nn.LayerNorm(d)
        self.branch_norm_2b = nn.LayerNorm(d)

        # ---------- Causal-masked tree refinement ----------
        tree_layers = int(config.get("tree_refine_layers", 2))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.tree_refine = nn.TransformerEncoder(enc_layer, num_layers=tree_layers)

        # ---------- Segment decoders ----------
        # Decode step-wise deltas for up to prediction_horizon; actual segment lengths are sliced dynamically.
        self.root_delta_head = nn.Linear(d, self.prediction_horizon * 2)
        self.branch1_delta_head = nn.Linear(d, self.prediction_horizon * 2)
        self.branch2_delta_head = nn.Linear(d, self.prediction_horizon * 2)

        self.mode_score_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.ReLU(),
            nn.Linear(d // 2, 1),
        )

        # This variant does not use diffusion.
        self.diffusion = None

    @staticmethod
    def _causal_mask(length: int, device: torch.device):
        # True means masked for nn.TransformerEncoder with bool mask.
        return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)

    def _compute_branch_points(self, num_points: int):
        """
        Compute two branch cut points p1, p2 in (0, T), with p1 < p2.
        """
        t = int(num_points)
        if t < 3:
            # Degenerate fallback; not expected for your datasets.
            p1 = 1
            p2 = max(1, t - 1)
            return p1, p2

        p1 = int(round(t * self.branch_ratio_1))
        p2 = int(round(t * self.branch_ratio_2))

        p1 = max(1, min(t - 2, p1))
        p2 = max(p1 + 1, min(t - 1, p2))
        return p1, p2

    def _build_backbone(self, trajectories, kg_data, vehicle_masks=None):
        """
        DIGIR dual-granularity backbone (unchanged logic).
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

    def _branch_queries(self, fused_conditions, intent_priors, interaction_features):
        """
        Build tree queries:
        root -> branch1 -> branch2, then refine with causal mask.
        Returns:
            root_q: (BN,1,D)
            b1_q:   (BN,B1,D)
            b2_q:   (BN,K,D) where K=B1*B2
        """
        b, n, d = fused_conditions.shape
        bn = b * n
        b1 = self.branch_factor_1
        b2 = self.branch_factor_2
        k = self.num_modes

        # Root query from dual-granularity prompt.
        root = fused_conditions + 0.5 * intent_priors + 0.5 * interaction_features
        root_q = self.root_norm(self.root_proj(root)).view(bn, 1, d)  # (BN,1,D)

        # Branch-1: split root into semantic coarse branches.
        seed1 = self.branch_token_1.unsqueeze(0).expand(bn, -1, -1)  # (BN,B1,D)
        attn1, _ = self.branch_attn_1(seed1, root_q, root_q)
        b1_q = self.branch_norm_1a(seed1 + attn1 + root_q.expand(-1, b1, -1))
        b1_q = self.branch_norm_1b(b1_q + self.branch_ffn_1(b1_q))

        # Branch-2: split each branch-1 node.
        # Prepare (BN*B1,B2,D)
        parent = b1_q.view(bn * b1, 1, d)
        seed2 = self.branch_token_2.view(1, b2, d).expand(bn * b1, -1, -1)
        attn2, _ = self.branch_attn_2(seed2, parent, parent)
        b2_q_local = self.branch_norm_2a(seed2 + attn2 + parent.expand(-1, b2, -1))
        b2_q_local = self.branch_norm_2b(b2_q_local + self.branch_ffn_2(b2_q_local))
        b2_q = b2_q_local.view(bn, k, d)  # (BN,K,D)

        # Causal tree refinement in feature space:
        # token order = [root] [branch1 nodes] [branch2 nodes]
        tree_tokens = torch.cat([root_q, b1_q, b2_q], dim=1)  # (BN, 1+B1+K, D)
        mask = self._causal_mask(tree_tokens.shape[1], tree_tokens.device)
        tree_tokens = self.tree_refine(tree_tokens, mask=mask)

        root_ref = tree_tokens[:, :1, :]
        b1_ref = tree_tokens[:, 1:1 + b1, :]
        b2_ref = tree_tokens[:, 1 + b1:, :]
        return root_ref, b1_ref, b2_ref

    def _decode_tree_trajectories(self, root_q, b1_q, b2_q, num_points):
        """
        Decode trajectories with shared-prefix tree topology.
        Args:
            root_q: (BN,1,D)
            b1_q:   (BN,B1,D)
            b2_q:   (BN,K,D), K=B1*B2
        Returns:
            traj_modes: (B,N,K,T,2)
            mode_logits: (B,N,K)
            branch_points: (p1,p2)
        """
        bn, _, d = root_q.shape
        b1 = self.branch_factor_1
        b2 = self.branch_factor_2
        k = self.num_modes
        t = int(num_points)
        p1, p2 = self._compute_branch_points(t)
        l1, l2, l3 = p1, (p2 - p1), (t - p2)

        # Decode delta sequences (canonical horizon), then slice to segment lengths.
        root_d = torch.tanh(self.root_delta_head(root_q.squeeze(1))).view(bn, self.prediction_horizon, 2) * self.delta_scale
        b1_d = torch.tanh(self.branch1_delta_head(b1_q)).view(bn, b1, self.prediction_horizon, 2) * self.delta_scale
        b2_d = torch.tanh(self.branch2_delta_head(b2_q)).view(bn, k, self.prediction_horizon, 2) * self.delta_scale

        # If required horizon differs from canonical horizon, interpolate.
        if self.prediction_horizon != t:
            root_d_i = F.interpolate(root_d.permute(0, 2, 1), size=t, mode="linear", align_corners=True).permute(0, 2, 1)
            b1_d_i = F.interpolate(
                b1_d.permute(0, 1, 3, 2).reshape(bn * b1, 2, self.prediction_horizon),
                size=t, mode="linear", align_corners=True
            ).view(bn, b1, 2, t).permute(0, 1, 3, 2)
            b2_d_i = F.interpolate(
                b2_d.permute(0, 1, 3, 2).reshape(bn * k, 2, self.prediction_horizon),
                size=t, mode="linear", align_corners=True
            ).view(bn, k, 2, t).permute(0, 1, 3, 2)
            root_d, b1_d, b2_d = root_d_i, b1_d_i, b2_d_i

        # Segment 1 (shared root prefix)
        seg1 = torch.cumsum(root_d[:, :l1, :], dim=1)  # (BN,l1,2)
        seg1_end = seg1[:, -1, :] if l1 > 0 else torch.zeros(bn, 2, device=root_q.device)

        # Segment 2 (shared within branch-1 family)
        if l2 > 0:
            seg2_delta = b1_d[:, :, :l2, :]  # (BN,B1,l2,2)
            seg2 = torch.cumsum(seg2_delta, dim=2) + seg1_end.unsqueeze(1).unsqueeze(2)
            seg2_end = seg2[:, :, -1, :]  # (BN,B1,2)
        else:
            seg2 = torch.zeros(bn, b1, 0, 2, device=root_q.device)
            seg2_end = seg1_end.unsqueeze(1).expand(-1, b1, -1)

        # Map final modes to their branch-1 parent.
        parent_idx = torch.arange(k, device=root_q.device, dtype=torch.long) // b2  # (K,)

        # Segment 3 (fully branched)
        if l3 > 0:
            seg3_delta = b2_d[:, :, :l3, :]  # (BN,K,l3,2)
            seg2_end_modes = seg2_end[:, parent_idx, :]  # (BN,K,2)
            seg3 = torch.cumsum(seg3_delta, dim=2) + seg2_end_modes.unsqueeze(2)  # (BN,K,l3,2)
        else:
            seg3 = torch.zeros(bn, k, 0, 2, device=root_q.device)

        # Assemble full trajectory for each final mode.
        parts = []
        if l1 > 0:
            seg1_modes = seg1.unsqueeze(1).expand(-1, k, -1, -1)  # (BN,K,l1,2)
            parts.append(seg1_modes)
        if l2 > 0:
            seg2_modes = seg2[:, parent_idx, :, :]  # (BN,K,l2,2)
            parts.append(seg2_modes)
        if l3 > 0:
            parts.append(seg3)  # (BN,K,l3,2)

        traj_modes_bn = torch.cat(parts, dim=2)  # (BN,K,T,2)
        mode_logits_bn = self.mode_score_head(b2_q).squeeze(-1)  # (BN,K)

        return traj_modes_bn, mode_logits_bn, (p1, p2)

    def _reshape_bn_to_bnk(self, x_bn, batch_size, num_agents):
        """
        x_bn: (BN, ...)
        -> (B, N, ...)
        """
        return x_bn.view(batch_size, num_agents, *x_bn.shape[1:])

    @staticmethod
    def _gather_mode(traj_modes, mode_idx):
        """
        traj_modes: (B,N,K,T,2), mode_idx: (B,N)
        return: (B,N,T,2)
        """
        b, n, _, t, c = traj_modes.shape
        idx = mode_idx.view(b, n, 1, 1, 1).expand(b, n, 1, t, c)
        return traj_modes.gather(2, idx).squeeze(2)

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
        idx = torch.multinomial(flat, 1).view(mode_logits.shape[0], mode_logits.shape[1])
        return idx

    def forward(self, trajectories, kg_data, future_traj=None, mode="train", vehicle_masks=None):
        """
        Interface-compatible forward.
        In train mode:
        - outputs["diffusion_loss"] is repurposed as tree-head training loss
        - outputs["traj_pred_train"] contains best-of-K trajectory for rule losses
        """
        outputs = self._build_backbone(trajectories, kg_data, vehicle_masks=vehicle_masks)
        b, n, t_hist, _ = trajectories.shape
        del t_hist

        root_q, b1_q, b2_q = self._branch_queries(
            outputs["fused_conditions"],
            outputs["intent_priors"],
            outputs["interaction_features"],
        )

        if mode == "train" and future_traj is not None:
            t = int(future_traj.shape[2])
            traj_bn, logits_bn, (p1, p2) = self._decode_tree_trajectories(root_q, b1_q, b2_q, num_points=t)
            traj_modes = self._reshape_bn_to_bnk(traj_bn, b, n)           # (B,N,K,T,2)
            mode_logits = self._reshape_bn_to_bnk(logits_bn, b, n)        # (B,N,K)

            outputs["traj_modes"] = traj_modes
            outputs["mode_logits"] = mode_logits
            outputs["branch_points"] = (p1, p2)

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
                goal_pred = best_traj[:, :, -1, :]
                goal_gt = future_traj[:, :, -1, :]
                goal_loss = F.smooth_l1_loss(goal_pred[valid], goal_gt[valid], reduction="mean")

                mode_loss = F.cross_entropy(
                    mode_logits.view(b * n, self.num_modes)[valid_flat],
                    best_idx.view(-1)[valid_flat],
                    reduction="mean",
                )
            else:
                z = torch.tensor(0.0, device=future_traj.device, dtype=future_traj.dtype)
                traj_loss = z
                goal_loss = z
                mode_loss = z

            tree_loss = traj_loss + self.lambda_goal * goal_loss + self.lambda_mode_cls * mode_loss

            # Keep compatibility with train_digir_full.py / base compute_losses().
            outputs["diffusion_loss"] = tree_loss
            outputs["loss_tree_traj"] = traj_loss
            outputs["loss_tree_goal"] = goal_loss
            outputs["loss_tree_mode"] = mode_loss

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
        Interface-compatible generation:
        - num_samples <= 1: (B,N,T,2)
        - num_samples > 1:  (S,B,N,T,2)
        """
        del sampling, step  # unused

        outputs = self._build_backbone(trajectories, kg_data, vehicle_masks=vehicle_masks)
        b, n, _, _ = trajectories.shape

        root_q, b1_q, b2_q = self._branch_queries(
            outputs["fused_conditions"],
            outputs["intent_priors"],
            outputs["interaction_features"],
        )

        traj_bn, logits_bn, _ = self._decode_tree_trajectories(
            root_q, b1_q, b2_q, num_points=int(num_points)
        )
        traj_modes = self._reshape_bn_to_bnk(traj_bn, b, n)       # (B,N,K,T,2)
        mode_logits = self._reshape_bn_to_bnk(logits_bn, b, n)    # (B,N,K)

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


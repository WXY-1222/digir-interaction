"""
DIGIR: Dual-Granularity Intent Rollout
Main Model - Integrates all components for trajectory prediction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoders.trajectory_encoder import TrajectorySetEncoder
from models.encoders.graph_encoder import KnowledgeGraphEncoder
from models.dual_granularity.cross_attention import LocalContextExtractor, IntentPriorQuery
from models.dual_granularity.scene_transformer import SceneIntentPooler
from models.dual_granularity.v2v_interaction import V2VInteraction
from models.dual_granularity.gated_fusion import GatedFusion
from models.diffusion.conditional_diffusion import ConditionalDiffusion, ConditionalDenoisingNet


class DIGIR(nn.Module):
    """
    Dual-Granularity Intent Rollout Model
    Implements the complete architecture from Section 4 of the paper
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.get('d_model', 128)
        self.num_intent_classes = config.get('num_intent_classes', 4)
        self.prediction_horizon = config.get('prediction_horizon', 12)
        self.hist_len = config.get('hist_len', 8)
        self.ablate_cross_attn = bool(config.get('ablate_cross_attn', False))
        self.ablate_gate = str(config.get('ablate_gate', 'none'))
        self.gate_fixed_ratio = config.get('gate_fixed_ratio', None)

        # ============ Encoders (Section 4.2.1) ============
        # Trajectory Encoder: encodes historical trajectories
        # Support 4D input: [x, y, heading, speed]
        self.traj_encoder = TrajectorySetEncoder(
            input_dim=4,  # Changed from 2 to support heading and speed
            hidden_dim=self.d_model,
            num_layers=config.get('traj_enc_layers', 4),
            num_heads=config.get('num_heads', 4),
            dropout=config.get('dropout', 0.1)
        )

        # Graph Encoder: encodes intersection Knowledge Graph
        self.graph_encoder = KnowledgeGraphEncoder(
            num_facility_types=config.get('num_facility_types', 10),
            facility_dim=32,
            hidden_dim=self.d_model,
            num_layers=config.get('graph_enc_layers', 3)
        )

        # ============ Scene-Level Intent Construction (Section 4.2) ============
        # Local Context Extraction: cross-attention between vehicle and graph
        self.local_context_extractor = LocalContextExtractor(
            d_model=self.d_model,
            num_heads=config.get('num_heads', 4),
            dropout=config.get('dropout', 0.1)
        )

        # Scene Transformer: global intent pooling
        self.scene_transformer = SceneIntentPooler(
            d_model=self.d_model,
            num_heads=config.get('num_heads', 4),
            num_transformer_layers=config.get('scene_tf_layers', 2),
            dropout=config.get('dropout', 0.1)
        )

        # ============ Cross-Granularity Mapping (Section 4.3) ============
        # Intent Prior Query: maps global intent to vehicle-specific priors
        self.intent_query = IntentPriorQuery(
            d_model=self.d_model,
            d_prior=config.get('d_prior', 128),
            num_heads=config.get('num_heads', 4),
            dropout=config.get('dropout', 0.1)
        )

        # Intent Classification Head (Section 4.3.2)
        self.intent_classifier = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(self.d_model // 2, self.num_intent_classes)
        )

        # ============ Agent-Level Interaction (Section 4.4) ============
        # V2V Interaction: micro-dynamic modeling
        self.v2v_interaction = V2VInteraction(
            d_model=self.d_model,
            num_heads=config.get('num_heads', 4),
            num_layers=config.get('v2v_layers', 2),
            dropout=config.get('dropout', 0.1)
        )

        # Gated Fusion: combines strategic intent and reactive interaction
        self.gated_fusion = GatedFusion(
            d_model=self.d_model,
            use_elementwise_gate=config.get('elementwise_gate', True)
        )

        # ============ Conditional Diffusion (Section 4.4.3) ============
        # Denoising network
        denoise_net = ConditionalDenoisingNet(
            point_dim=2,
            context_dim=self.d_model,
            tf_layer=config.get('diffusion_tf_layers', 4),
            residual=False
        )

        # Diffusion model
        self.diffusion = ConditionalDiffusion(
            net=denoise_net,
            num_steps=config.get('diffusion_steps', 100),
            beta_1=float(config.get('beta_1', 1e-4)),
            beta_T=float(config.get('beta_T', 5e-2))
        )

        # ============ Auxiliary Components ============
        # Trajectory encoder for posterior intent (used in L_cross)
        self.posterior_traj_encoder = nn.Sequential(
            nn.Linear(self.prediction_horizon * 2, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model)
        )

        self.posterior_intent_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.d_model // 2, self.num_intent_classes)
        )

    def encode_scene(self, trajectories, kg_data, vehicle_masks=None):
        """
        Scene-level semantic intent construction (Section 4.2)

        Args:
            trajectories: (batch_size, N, hist_len, 4) - historical trajectories [x, y, heading, speed]
            kg_data: dict with keys:
                - facility_types: (batch_size, M)
                - positions: (batch_size, M, 2)
                - edge_index: graph connectivity
                - edge_types: edge type indices

        Returns:
            scene_intent: (batch_size, d_model) - z_scene
            local_contexts: (batch_size, N, d_model) - k_a^t
            motion_summaries: (batch_size, N, d_model) - s_a^t
        """
        batch_size, N, _, _ = trajectories.shape

        # 1. Encode trajectories (Eq. 4.2)
        motion_summaries = self.traj_encoder(trajectories)  # (B, N, d)
        if vehicle_masks is not None:
            motion_summaries = motion_summaries * vehicle_masks.unsqueeze(-1).float()

        # 2-3. Encode Knowledge Graph + Local Context Extraction via Cross-Attention (Eq. 4.3-4.5)
        if self.ablate_cross_attn:
            # Ablation: remove map-conditioned cross-attention signal.
            # Keep tensor shape unchanged so downstream modules are untouched.
            local_contexts = motion_summaries
        else:
            graph_embeddings = self.graph_encoder(
                kg_data['facility_types'],
                kg_data['positions'],
                kg_data['edge_index'],
                kg_data.get('edge_types')
            )  # (B, M, d)
            local_contexts = self.local_context_extractor(
                motion_summaries, graph_embeddings, vehicle_mask=vehicle_masks
            )  # (B, N, d)

        # 4. Global Intent Pooling via Scene Transformer (Eq. 4.6-4.8)
        scene_intent, local_contexts = self.scene_transformer(
            local_contexts,
            vehicle_mask=vehicle_masks,
        )  # (B, d), (B, N, d)

        return scene_intent, local_contexts, motion_summaries

    def cross_granularity_mapping(self, scene_intent, local_contexts, vehicle_masks=None):
        """
        Cross-granularity prior mapping (Section 4.3)

        Args:
            scene_intent: (batch_size, d_model) - z_scene
            local_contexts: (batch_size, N, d_model) - k_a^t

        Returns:
            intent_priors: (batch_size, N, d_model) - z_a^t
            intent_logits: (batch_size, N, num_intent_classes) - predicted intents
        """
        # Expand scene intent for all vehicles
        scene_expanded = scene_intent.unsqueeze(1)  # (B, 1, d)

        # Intent Prior Querying (Eq. 4.11, 4.12)
        intent_priors = self.intent_query(
            local_contexts,
            scene_expanded,
            vehicle_mask=vehicle_masks,
        )  # (B, N, d)

        # Intent Classification (Eq. 4.13)
        intent_logits = self.intent_classifier(intent_priors)  # (B, N, C)

        return intent_priors, intent_logits

    def agent_level_modeling(self, motion_summaries, intent_priors, vehicle_masks=None):
        """
        Agent-level interaction and fusion (Section 4.4)

        Args:
            motion_summaries: (batch_size, N, d_model) - s_a^t
            intent_priors: (batch_size, N, d_model) - z_a^t

        Returns:
            fused_conditions: (batch_size, N, d_model) - X_a^t
            interaction_features: (batch_size, N, d_model) - u_a^t
            gate_weights: (batch_size, N, d) or (B, N, 1) - g_a^t
        """
        # 1. V2V Micro-Dynamic Interaction (Eq. 4.14, 4.15)
        interaction_features = self.v2v_interaction(
            motion_summaries,
            vehicle_mask=vehicle_masks,
        )  # (B, N, d)

        # 2. Gated Fusion (Eq. 4.16, 4.17) + optional ablations
        if self.gate_fixed_ratio is not None:
            r = float(max(0.0, min(1.0, float(self.gate_fixed_ratio))))
            gate_weights = torch.full_like(interaction_features, r)
            fused_conditions = r * interaction_features + (1.0 - r) * intent_priors
        elif self.ablate_gate == 'fixed_half':
            gate_weights = torch.full_like(interaction_features, 0.5)
            fused_conditions = 0.5 * interaction_features + 0.5 * intent_priors
        elif self.ablate_gate == 'force_intent':
            gate_weights = torch.zeros_like(interaction_features)
            fused_conditions = intent_priors
        elif self.ablate_gate == 'force_interaction':
            gate_weights = torch.ones_like(interaction_features)
            fused_conditions = interaction_features
        else:
            fused_conditions, gate_weights = self.gated_fusion(
                interaction_features, intent_priors
            )  # (B, N, d) / gate: (B,N,d) or (B,N,1)

        if vehicle_masks is not None:
            vm = vehicle_masks.unsqueeze(-1).float()
            fused_conditions = fused_conditions * vm
            interaction_features = interaction_features * vm
            gate_weights = gate_weights * vm

        return fused_conditions, interaction_features, gate_weights

    def forward(self, trajectories, kg_data, future_traj=None, mode='train', vehicle_masks=None):
        """
        Forward pass

        Args:
            trajectories: (batch_size, N, hist_len, 4) - [x, y, heading, speed]
            kg_data: dict with KG information
            future_traj: (batch_size, N, T, 2) - ground truth future [x, y] (for training)
            mode: 'train' or 'eval'
            vehicle_masks: (batch_size, N) optional — masks diffusion loss to valid agents only

        Returns:
            outputs: dict containing predictions and intermediate representations
        """
        batch_size, N, _, _ = trajectories.shape

        # ============ Scene-Level Intent Construction ============
        scene_intent, local_contexts, motion_summaries = self.encode_scene(
            trajectories,
            kg_data,
            vehicle_masks=vehicle_masks,
        )

        # ============ Cross-Granularity Mapping ============
        intent_priors, intent_logits = self.cross_granularity_mapping(
            scene_intent,
            local_contexts,
            vehicle_masks=vehicle_masks,
        )

        # ============ Agent-Level Modeling ============
        fused_conditions, interaction_features, gate_weights = self.agent_level_modeling(
            motion_summaries,
            intent_priors,
            vehicle_masks=vehicle_masks,
        )

        outputs = {
            'scene_intent': scene_intent,
            'local_contexts': local_contexts,
            'motion_summaries': motion_summaries,
            'intent_priors': intent_priors,
            'intent_logits': intent_logits,
            'interaction_features': interaction_features,
            'fused_conditions': fused_conditions,
            'gate_weights': gate_weights
        }

        # ============ Diffusion-based Generation ============
        if mode == 'train' and future_traj is not None:
            # Training: compute diffusion loss
            # Reshape for processing all agents
            future_flat = future_traj.view(batch_size * N, self.prediction_horizon, 2)
            condition_flat = fused_conditions.view(batch_size * N, self.d_model)
            mask_flat = None
            if vehicle_masks is not None:
                mask_flat = vehicle_masks.reshape(-1).float()

            diffusion_loss = self.diffusion.get_loss(future_flat, condition_flat, mask=mask_flat)
            outputs['diffusion_loss'] = diffusion_loss

            # Additionally sample one trajectory per agent for rule-based losses (L_col, L_map)
            # Only do this when rule loss is enabled to avoid extra compute for baseline.
            lambda_rule = self.config.get('lambda_rule', 0.0)
            if lambda_rule and float(lambda_rule) > 0.0:
                sample_step = max(1, int(self.config.get('sample_step', 10)))
                with torch.no_grad():
                    samples = self.diffusion.sample(
                        num_points=self.prediction_horizon,
                        context=condition_flat,
                        num_samples=1,
                        sampling="ddim",
                        step=sample_step,
                    )  # (1, B*N, T, 2)
                    samples = samples[0].view(batch_size, N, self.prediction_horizon, 2)
                outputs['traj_pred_train'] = samples

        return outputs

    def generate(self, trajectories, kg_data, num_points=12, num_samples=20,
                 sampling="ddim", step=20, bestof=True, vehicle_masks=None):
        """
        Generate future trajectory predictions

        Args:
            trajectories: (batch_size, N, hist_len, 4) - [x, y, heading, speed]
            kg_data: dict with KG information
            num_points: number of future timesteps
            num_samples: number of trajectory samples
            sampling: "ddpm" or "ddim"
            step: diffusion sampling stride
            bestof: whether to select best sample

        Returns:
            trajectories: (num_samples, batch_size, N, num_points, 2)
        """
        batch_size, N, _, _ = trajectories.shape

        # Encode scene
        scene_intent, local_contexts, motion_summaries = self.encode_scene(
            trajectories,
            kg_data,
            vehicle_masks=vehicle_masks,
        )

        # Cross-granularity mapping
        intent_priors, intent_logits = self.cross_granularity_mapping(
            scene_intent,
            local_contexts,
            vehicle_masks=vehicle_masks,
        )

        # Agent-level modeling
        fused_conditions, _, _ = self.agent_level_modeling(
            motion_summaries,
            intent_priors,
            vehicle_masks=vehicle_masks,
        )

        # Generate trajectories with diffusion
        condition_flat = fused_conditions.view(batch_size * N, self.d_model)

        samples = self.diffusion.sample(
            num_points=num_points,
            context=condition_flat,
            num_samples=num_samples,
            sampling=sampling,
            step=step
        )  # (num_samples, B*N, T, 2)

        # Reshape back
        samples = samples.view(num_samples, batch_size, N, num_points, 2)

        if bestof and num_samples > 1:
            # Select best samples based on intent alignment
            samples = self._select_best_samples(samples, intent_logits)

        return samples

    def _select_best_samples(self, samples, intent_logits):
        """
        Select best trajectory samples based on scoring function
        (Simplified version of Algorithm 1)
        """
        # Placeholder: select first sample for now
        # Full implementation would use Eq. 4.25
        return samples[0] if samples.size(0) > 1 else samples

    def compute_losses(self, outputs, future_traj, intent_labels, vehicle_masks=None):
        """
        Compute dual-granularity training objectives (Section 4.5)

        Args:
            outputs: dict from forward pass
            future_traj: (batch_size, N, T, 2) - ground truth future
            intent_labels: (batch_size, N) - intent pseudo-labels
            vehicle_masks: (batch_size, N) - valid vehicle mask

        Returns:
            losses: dict of individual losses
            total_loss: weighted sum of all losses
        """
        batch_size, N, T, _ = future_traj.shape
        device = future_traj.device

        # 1. Low-Granularity Reconstruction + Rule Loss (Eq. 4.18, 4.19, 4.20)
        loss_diff = outputs['diffusion_loss']

        # Predicted trajectory used for rule losses; fall back to GT if missing
        Y_pred = outputs.get('traj_pred_train', None)
        if Y_pred is None:
            Y_pred = future_traj

        # Training script sets ref_point = per-agent last (x,y) in global meters so Y_pred + ref_point is global
        # when Y_pred is local/displacement (diffusion target).
        ref_point = outputs.get('ref_point', None)  # (B, N, 1, 2)
        if ref_point is not None:
            Y_pred_global = Y_pred + ref_point
        else:
            Y_pred_global = Y_pred

        if vehicle_masks is not None:
            valid_mask = vehicle_masks.bool()
        else:
            valid_mask = torch.ones(batch_size, N, dtype=torch.bool, device=device)

        # 1.1 Collision penalty L_col (soft penalty on distances below threshold)
        col_loss = torch.tensor(0.0, device=device)
        for b in range(batch_size):
            idx = torch.where(valid_mask[b])[0]
            n_valid = int(idx.numel())
            if n_valid < 2:
                continue
            pos = Y_pred_global[b, idx]  # (n, T, 2) in global coordinates
            diff = pos.unsqueeze(1) - pos.unsqueeze(2)  # (n, n, T, 2)
            dist = torch.norm(diff, dim=-1)  # (n, n, T)
            thresh = 2.0
            pen = torch.relu(thresh - dist)
            # Only penalize unique agent pairs (i<j) on agent-agent axes.
            # Avoid torch.triu on 3D tensors, which would act on the last two dims.
            tri = torch.triu_indices(n_valid, n_valid, offset=1, device=device)
            if tri.numel() > 0:
                col_loss = col_loss + pen[tri[0], tri[1], :].mean()
        if batch_size > 0:
            col_loss = col_loss / batch_size

        # 1.2 Off-road penalty L_map: distance to nearest road segment (edge_index) beyond margin
        kg_pos = outputs.get('kg_positions', None)
        kg_edge_index = outputs.get('kg_edge_index', None)
        map_loss = torch.tensor(0.0, device=device)
        if kg_pos is not None:
            if kg_pos.dim() == 2:
                kg_pos = kg_pos.unsqueeze(0).expand(batch_size, -1, -1)
            margin = self.config.get('map_margin', 3.0)
            for b in range(batch_size):
                idx = torch.where(valid_mask[b])[0]
                if idx.numel() == 0:
                    continue
                pts = Y_pred_global[b, idx].reshape(-1, 2)  # (n*T, 2)
                nodes = kg_pos[b, :, :2]             # (M, 2)
                if kg_edge_index is not None and kg_edge_index.numel() > 0:
                    ei = kg_edge_index
                    if ei.dim() == 3:
                        # (B,2,E) -> take b-th
                        ei = ei[b]
                    a = nodes[ei[0].long()]  # (E,2)
                    bnd = nodes[ei[1].long()]  # (E,2)
                    ab = bnd - a
                    ab2 = (ab * ab).sum(dim=1).clamp_min(1e-8)  # (E,)
                    p = pts[:, None, :]  # (P,1,2)
                    a_ = a[None, :, :]   # (1,E,2)
                    ab_ = ab[None, :, :] # (1,E,2)
                    ap = p - a_
                    t = (ap * ab_).sum(dim=2) / ab2[None, :]
                    t = t.clamp(0.0, 1.0)
                    proj = a_ + t[:, :, None] * ab_
                    d = torch.norm(p - proj, dim=2)  # (P,E)
                    min_d = d.min(dim=1)[0]          # (P,)
                else:
                    dists = torch.cdist(pts, nodes)      # (n*T, M)
                    min_d = dists.min(dim=1)[0]         # (n*T,)
                pen = torch.relu(min_d - margin)    # only penalize beyond margin
                map_loss = map_loss + pen.mean()
            if batch_size > 0:
                map_loss = map_loss / batch_size

        lambda_rule = float(self.config.get('lambda_rule', 0.0))
        if lambda_rule <= 0.0:
            # Baseline: no rule loss
            loss_fine = loss_diff
            col_loss = torch.tensor(0.0, device=device)
            map_loss = torch.tensor(0.0, device=device)
            loss_rule = torch.tensor(0.0, device=device)
        else:
            loss_rule = col_loss + map_loss
            loss_fine = loss_diff + lambda_rule * loss_rule

        # 2. High-Granularity Intent Loss (Eq. 4.21)
        intent_logits = outputs['intent_logits']  # (B, N, C)
        intent_logits_flat = intent_logits.view(batch_size * N, -1)
        intent_labels_flat = intent_labels.view(-1)
        # Robust cross-entropy:
        # - If a batch has all labels as -1 (ignored), PyTorch can still yield unstable numerics.
        # - Filter valid labels explicitly.
        valid_mask = intent_labels_flat != -1
        if valid_mask.any():
            loss_coarse = F.cross_entropy(
                intent_logits_flat[valid_mask],
                intent_labels_flat[valid_mask],
            )
        else:
            loss_coarse = torch.tensor(0.0, device=intent_logits_flat.device, dtype=intent_logits_flat.dtype)

        # 3. Cross-Granularity Alignment Loss (Eq. 4.22, 4.23)
        # Encode generated trajectory to posterior intent
        with torch.no_grad():
            # This should be done with actual generated trajectory
            # For training, we use ground truth as approximation
            future_flat = future_traj.view(batch_size * N, -1)  # (B*N, T*2)
            posterior_feat = self.posterior_traj_encoder(future_flat)
            posterior_logits = self.posterior_intent_head(posterior_feat)
            posterior_probs = F.softmax(posterior_logits, dim=-1)

        # KL divergence (stable):
        # Use log_softmax for the prior and clamp/renormalize both distributions to avoid log(0).
        eps = 1e-8
        prior_log_probs = F.log_softmax(intent_logits_flat, dim=-1)
        posterior_probs = posterior_probs.clamp(min=eps)
        posterior_probs = posterior_probs / posterior_probs.sum(dim=-1, keepdim=True)

        valid_cross = intent_labels_flat != -1
        if vehicle_masks is not None:
            valid_cross = valid_cross & vehicle_masks.reshape(-1).bool()
        if valid_cross.any():
            loss_cross = F.kl_div(
                prior_log_probs[valid_cross],
                posterior_probs[valid_cross],
                reduction='batchmean',
            )
        else:
            loss_cross = torch.tensor(0.0, device=intent_logits_flat.device, dtype=intent_logits_flat.dtype)

        # Total loss (Eq. 4.24)
        lambda_1 = self.config.get('lambda_fine', 1.0)
        lambda_2 = self.config.get('lambda_coarse', 0.5)
        lambda_3 = self.config.get('lambda_cross', 0.1)

        total_loss = lambda_1 * loss_fine + lambda_2 * loss_coarse + lambda_3 * loss_cross

        losses = {
            'loss_diff': float(loss_diff.detach().item()),
            'loss_col': float(col_loss.detach().item()),
            'loss_map': float(map_loss.detach().item()),
            'loss_rule': float(loss_rule.detach().item()),
            'lambda_rule': float(lambda_rule),
            'loss_fine': float(loss_fine.detach().item()),
            'loss_coarse': float(loss_coarse.detach().item()),
            'loss_cross': float(loss_cross.detach().item()),
            'total_loss': total_loss,
        }

        return losses, total_loss

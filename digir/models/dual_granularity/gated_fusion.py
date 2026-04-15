"""
DIGIR: Dual-Granularity Intent Rollout
Gated Fusion Module - Dual-granularity information fusion
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFusion(nn.Module):
    """
    Dual-granularity gated fusion mechanism (Section 4.4.2)
    Dynamically balances strategic intent and reactive interaction
    """
    def __init__(self, d_model, use_elementwise_gate=True):
        super().__init__()
        self.d_model = d_model
        self.use_elementwise_gate = use_elementwise_gate

        # MLP for computing gating weight
        gate_input_dim = d_model * 2
        if use_elementwise_gate:
            # Element-wise gating (per-dimension)
            self.gate_mlp = nn.Sequential(
                nn.Linear(gate_input_dim, d_model),
                nn.Sigmoid()
            )
        else:
            # Scalar gating (single weight for all dimensions)
            self.gate_mlp = nn.Sequential(
                nn.Linear(gate_input_dim, 1),
                nn.Sigmoid()
            )

    def forward(self, interaction_features, intent_priors):
        """
        Args:
            interaction_features: (batch_size, N, d_model) - u_a^t (micro-dynamic)
            intent_priors: (batch_size, N, d_model) - z_a^t (macro-intent)
        Returns:
            fused_features: (batch_size, N, d_model) - X_a^t
        """
        # Concatenate features
        combined = torch.cat([interaction_features, intent_priors], dim=-1)

        # Compute gating weight
        g = self.gate_mlp(combined)  # (B, N, d) or (B, N, 1)

        # Element-wise convex combination
        fused = g * interaction_features + (1 - g) * intent_priors

        return fused, g  # Return gate for interpretability


class MultiScaleGatedFusion(nn.Module):
    """
    Extension: Multi-scale gated fusion with hierarchical gates
    Allows finer control over fusion at different semantic levels
    """
    def __init__(self, d_model, num_scales=3):
        super().__init__()
        self.d_model = d_model
        self.num_scales = num_scales

        # Scale-specific projections
        self.scale_projections = nn.ModuleList([
            nn.Linear(d_model, d_model // num_scales)
            for _ in range(num_scales)
        ])

        # Gate for each scale
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model * 2 // num_scales, 1),
                nn.Sigmoid()
            )
            for _ in range(num_scales)
        ])

        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, interaction_features, intent_priors):
        """
        Args:
            interaction_features: (batch_size, N, d_model)
            intent_priors: (batch_size, N, d_model)
        Returns:
            fused_features: (batch_size, N, d_model)
        """
        batch_size, N, _ = interaction_features.shape

        # Project to scale-specific subspaces
        interaction_scales = [proj(interaction_features) for proj in self.scale_projections]
        intent_scales = [proj(intent_priors) for proj in self.scale_projections]

        # Fuse at each scale
        fused_scales = []
        for i in range(self.num_scales):
            combined = torch.cat([interaction_scales[i], intent_scales[i]], dim=-1)
            g = self.gates[i](combined)  # (B, N, 1)
            fused = g * interaction_scales[i] + (1 - g) * intent_scales[i]
            fused_scales.append(fused)

        # Concatenate and project
        fused_concat = torch.cat(fused_scales, dim=-1)
        output = self.output_proj(fused_concat)

        return output


class TemporalGatedFusion(nn.Module):
    """
    Temporal extension: Gate weights that evolve over prediction horizon
    """
    def __init__(self, d_model, max_horizon=12):
        super().__init__()
        self.d_model = d_model
        self.max_horizon = max_horizon

        # Learnable temporal gate bias
        self.temporal_bias = nn.Parameter(torch.zeros(max_horizon, d_model))

        self.gate_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

    def forward(self, interaction_features, intent_priors, timestep=None):
        """
        Args:
            interaction_features: (batch_size, N, d_model) or (batch_size, N, T, d_model)
            intent_priors: (batch_size, N, d_model) or (batch_size, N, T, d_model)
            timestep: int or None - if provided, use temporal bias at this step
        """
        combined = torch.cat([interaction_features, intent_priors], dim=-1)
        g_base = self.gate_mlp(combined)

        if timestep is not None and timestep < self.max_horizon:
            # Add temporal bias
            bias = self.temporal_bias[timestep:timestep+1]
            if len(g_base.shape) == 4:  # (B, N, T, d)
                bias = bias.unsqueeze(0).unsqueeze(0)
            elif len(g_base.shape) == 3:  # (B, N, d)
                bias = bias.unsqueeze(0)
            g = torch.sigmoid(g_base + bias)
        else:
            g = g_base

        fused = g * interaction_features + (1 - g) * intent_priors
        return fused, g

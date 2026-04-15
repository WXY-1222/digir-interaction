"""
DIGIR: Dual-Granularity Intent Rollout
V2V Interaction Module - Micro-dynamic interaction modeling
"""
import torch
import torch.nn as nn


class V2VInteraction(nn.Module):
    """
    Vehicle-to-Vehicle interaction modeling via self-attention (Section 4.4.1)
    Captures reactive, microscopic driving instincts
    """
    def __init__(self, d_model, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Multi-head self-attention layers
        self.attention_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm1_layers = nn.ModuleList()
        self.norm2_layers = nn.ModuleList()

        for _ in range(num_layers):
            self.attention_layers.append(
                nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
            )
            self.ffn_layers.append(nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model)
            ))
            self.norm1_layers.append(nn.LayerNorm(d_model))
            self.norm2_layers.append(nn.LayerNorm(d_model))

        self.dropout = nn.Dropout(dropout)

    def forward(self, kinematic_features):
        """
        Args:
            kinematic_features: (batch_size, N, d_model) - S^t = [s_1^t, ..., s_N^t]
        Returns:
            interaction_features: (batch_size, N, d_model) - u_a^t
        """
        x = kinematic_features

        # Apply multiple attention layers
        for i in range(len(self.attention_layers)):
            # Self-attention
            attn_out, _ = self.attention_layers[i](x, x, x)
            x = self.norm1_layers[i](x + self.dropout(attn_out))

            # FFN
            ffn_out = self.ffn_layers[i](x)
            x = self.norm2_layers[i](x + self.dropout(ffn_out))

        return x  # u_a^t


class InteractionGraph(nn.Module):
    """
    Fully connected interaction graph for V2V communication
    Optionally with distance-based edge weighting
    """
    def __init__(self, d_model, use_distance_weighting=True):
        super().__init__()
        self.use_distance_weighting = use_distance_weighting

        if use_distance_weighting:
            # MLP for computing edge weights based on relative positions
            self.edge_weight_mlp = nn.Sequential(
                nn.Linear(4, 64),  # [dx, dy, dvx, dvy]
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

    def forward(self, vehicle_states, positions=None):
        """
        Args:
            vehicle_states: (batch_size, N, d_model)
            positions: (batch_size, N, 2) - optional, for distance weighting
        Returns:
            adjacency: (batch_size, N, N) - adjacency matrix (fully connected or weighted)
        """
        batch_size, N, _ = vehicle_states.shape

        if not self.use_distance_weighting or positions is None:
            # Fully connected (uniform weights)
            return torch.ones(batch_size, N, N, device=vehicle_states.device)

        # Compute distance-based edge weights
        # Expand positions for pairwise differences
        pos_i = positions.unsqueeze(2)  # (B, N, 1, 2)
        pos_j = positions.unsqueeze(1)  # (B, 1, N, 2)

        relative_pos = pos_i - pos_j  # (B, N, N, 2)
        distances = torch.norm(relative_pos, dim=-1, keepdim=True)  # (B, N, N, 1)

        # Compute edge weights
        edge_features = torch.cat([relative_pos, distances], dim=-1)  # (B, N, N, 3)
        edge_weights = self.edge_weight_mlp(edge_features).squeeze(-1)  # (B, N, N)

        return edge_weights

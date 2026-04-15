"""
DIGIR: Dual-Granularity Intent Rollout
Trajectory Encoder - Transformer-based historical trajectory encoding
"""
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1), :]


class TrajectoryEncoder(nn.Module):
    """
    Transformer-based trajectory encoder (Section 4.2.1)
    Encodes historical trajectory into motion summary vector s_a^t
    """
    def __init__(self, input_dim=2, hidden_dim=128, num_layers=4, num_heads=4, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input embedding
        self.input_embed = nn.Linear(input_dim, hidden_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output normalization
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, trajectory):
        """
        Args:
            trajectory: (batch_size, hist_len, 2) - historical trajectory coordinates
        Returns:
            s_a: (batch_size, hidden_dim) - motion summary vector
        """
        # Embed input
        x = self.input_embed(trajectory)  # (B, H, d)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer(x)  # (B, H, d)

        # Pool by taking the last token (or mean pooling)
        s_a = x[:, -1, :]  # (B, d) - take last timestep

        s_a = self.norm(s_a)
        return s_a


class TrajectorySetEncoder(nn.Module):
    """
    Encodes trajectories for all N vehicles in the scene
    """
    def __init__(self, input_dim=2, hidden_dim=128, num_layers=4, num_heads=4, dropout=0.1):
        super().__init__()
        self.encoder = TrajectoryEncoder(input_dim, hidden_dim, num_layers, num_heads, dropout)

    def forward(self, trajectories):
        """
        Args:
            trajectories: (batch_size, N, hist_len, feature_dim) - trajectories of N vehicles
        Returns:
            S: (batch_size, N, hidden_dim) - motion summaries for all vehicles
        """
        batch_size, N, hist_len, feature_dim = trajectories.shape

        # Reshape to process all trajectories
        traj_flat = trajectories.view(batch_size * N, hist_len, feature_dim)

        # Encode
        s_flat = self.encoder(traj_flat)  # (B*N, d)

        # Reshape back
        S = s_flat.view(batch_size, N, -1)  # (B, N, d)
        return S

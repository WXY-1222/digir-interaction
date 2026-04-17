"""
DIGIR: Dual-Granularity Intent Rollout
Scene Transformer - Global intent pooling mechanism
"""
import torch
import torch.nn as nn


class SceneTransformer(nn.Module):
    """
    Scene Transformer for global intent pooling (Section 4.2.3)
    Aggregates all vehicle local contexts into a global scene intent z_scene
    """
    def __init__(self, d_model, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Learnable scene token (analogous to [CLS] token)
        self.scene_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Layer normalization
        self.norm = nn.LayerNorm(d_model)

    def forward(self, local_contexts, vehicle_mask=None):
        """
        Args:
            local_contexts: (batch_size, N, d_model) - k_1^t, ..., k_N^t
        Returns:
            scene_intent: (batch_size, d_model) - z_scene
            updated_contexts: (batch_size, N, d_model) - updated vehicle representations
        """
        batch_size, N, d = local_contexts.shape

        # Expand scene token for batch
        scene_token = self.scene_token.expand(batch_size, -1, -1)  # (B, 1, d)

        # Prepend scene token to local contexts
        sequence = torch.cat([scene_token, local_contexts], dim=1)  # (B, N+1, d)

        # Apply transformer with optional key padding mask.
        src_key_padding_mask = None
        if vehicle_mask is not None:
            pad_mask = (~vehicle_mask.bool())
            cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=local_contexts.device)
            src_key_padding_mask = torch.cat([cls_mask, pad_mask], dim=1)  # (B, N+1)
        transformed = self.transformer(sequence, src_key_padding_mask=src_key_padding_mask)  # (B, N+1, d)

        # Extract scene intent from first position
        scene_intent = transformed[:, 0, :]  # (B, d)

        # Extract updated local contexts (optional, for downstream use)
        updated_contexts = transformed[:, 1:, :]  # (B, N, d)
        if vehicle_mask is not None:
            updated_contexts = updated_contexts * vehicle_mask.unsqueeze(-1).float()

        scene_intent = self.norm(scene_intent)

        return scene_intent, updated_contexts


class SceneIntentPooler(nn.Module):
    """
    Complete scene-level intent construction module (Section 4.2)
    Combines all steps: encoding, cross-attention, and global pooling
    """
    def __init__(self, d_model, num_heads=4, num_transformer_layers=2, dropout=0.1):
        super().__init__()
        self.scene_transformer = SceneTransformer(
            d_model=d_model,
            num_layers=num_transformer_layers,
            num_heads=num_heads,
            dropout=dropout
        )

    def forward(self, local_contexts, vehicle_mask=None):
        """
        Args:
            local_contexts: (batch_size, N, d_model) - from cross-attention
        Returns:
            scene_intent: (batch_size, d_model) - z_scene^t
            vehicle_features: (batch_size, N, d_model) - updated features
        """
        scene_intent, vehicle_features = self.scene_transformer(local_contexts, vehicle_mask=vehicle_mask)
        return scene_intent, vehicle_features

"""
DIGIR: Dual-Granularity Intent Rollout
Cross-Attention Module - For local context extraction and cross-granularity mapping
"""
import torch
import torch.nn as nn
import math


class CrossAttention(nn.Module):
    """
    Cross-Attention mechanism (Section 4.2.2 and 4.3.1)
    Used for:
    1. Local Context Extraction: s_a^t -> k_a^t (vehicle to graph)
    2. Intent Prior Querying: z_scene -> z_a (scene to vehicle)
    """
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Linear projections for Q, K, V
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        # Output projection
        self.W_O = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch_size, seq_len_q, d_model)
            key: (batch_size, seq_len_k, d_model)
            value: (batch_size, seq_len_v, d_model)
            mask: optional attention mask
        Returns:
            output: (batch_size, seq_len_q, d_model)
            attention_weights: (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        batch_size = query.size(0)

        # Linear projections and reshape for multi-head attention
        # (B, seq_len, d_model) -> (B, num_heads, seq_len, d_k)
        Q = self.W_Q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, num_heads, seq_q, seq_k)

        if mask is not None:
            # Supported mask formats:
            # - (B, seq_k): key-valid mask, 1/True = keep, 0/False = mask out
            # - (B, seq_q, seq_k): pair-wise mask
            # - (B, 1, seq_q, seq_k) or (B, num_heads, seq_q, seq_k): pre-expanded
            if mask.dim() == 2:
                m = mask[:, None, None, :]
            elif mask.dim() == 3:
                m = mask[:, None, :, :]
            else:
                m = mask
            scores = scores.masked_fill(m == 0, -1e9)

        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)  # (B, num_heads, seq_q, d_k)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Final linear projection
        output = self.W_O(context)

        return output, attention_weights


class LocalContextExtractor(nn.Module):
    """
    Local context extraction via cross-attention (Section 4.2.2)
    Extracts vehicle-centric local semantic state k_a^t
    """
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        self.cross_attn = CrossAttention(d_model, num_heads, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, vehicle_state, graph_embeddings, vehicle_mask=None):
        """
        Args:
            vehicle_state: (batch_size, N, d_model) - motion summary s_a^t
            graph_embeddings: (batch_size, M, d_model) - H_kg
        Returns:
            local_context: (batch_size, N, d_model) - k_a^t
        """
        batch_size, N, d = vehicle_state.shape

        # Reshape for processing all vehicles
        Q = vehicle_state  # (B, N, d)
        K = V = graph_embeddings  # (B, M, d)

        # Cross-attention
        attn_out, _ = self.cross_attn(Q, K, V)  # (B, N, d)

        # Residual connection and layer norm
        local_context = self.norm(vehicle_state + self.dropout(attn_out))
        if vehicle_mask is not None:
            local_context = local_context * vehicle_mask.unsqueeze(-1).float()

        return local_context


class IntentPriorQuery(nn.Module):
    """
    Cross-granularity intent querying (Section 4.3.1)
    Maps global scene intent to vehicle-specific intent priors
    """
    def __init__(self, d_model, d_prior=None, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_prior = d_prior or d_model

        # Projection matrices for prior subspace
        self.W_Q_prior = nn.Linear(d_model, self.d_prior)
        self.W_K_prior = nn.Linear(d_model, self.d_prior)
        self.W_V_prior = nn.Linear(d_model, self.d_prior)

        self.cross_attn = CrossAttention(self.d_prior, num_heads, dropout)

        # FFN for semantic richness
        self.ffn = nn.Sequential(
            nn.Linear(self.d_prior, self.d_prior * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_prior * 4, self.d_prior)
        )

        self.norm1 = nn.LayerNorm(self.d_prior)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Project back to d_model if needed
        self.output_proj = nn.Linear(self.d_prior, d_model) if self.d_prior != d_model else nn.Identity()

    def forward(self, local_context, scene_intent, vehicle_mask=None):
        """
        Args:
            local_context: (batch_size, N, d_model) - k_a^t
            scene_intent: (batch_size, 1, d_model) - z_scene
        Returns:
            intent_prior: (batch_size, N, d_model) - z_a^t
        """
        # Project to prior subspace
        Q = self.W_Q_prior(local_context)  # (B, N, d_p)
        K = self.W_K_prior(scene_intent)   # (B, 1, d_p)
        V = self.W_V_prior(scene_intent)   # (B, 1, d_p)

        # Cross-attention: query scene intent with vehicle context
        attn_out, _ = self.cross_attn(Q, K, V)  # (B, N, d_p)
        attn_out = self.norm1(Q + self.dropout(attn_out))

        # FFN
        ffn_out = self.ffn(attn_out)
        intent_prior = self.norm1(attn_out + self.dropout(ffn_out))

        # Project back and residual with local context
        intent_prior = self.output_proj(intent_prior)
        intent_prior = self.norm2(local_context + self.dropout(intent_prior))
        if vehicle_mask is not None:
            intent_prior = intent_prior * vehicle_mask.unsqueeze(-1).float()

        return intent_prior

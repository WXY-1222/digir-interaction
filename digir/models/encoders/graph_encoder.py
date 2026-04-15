"""
DIGIR: Dual-Granularity Intent Rollout
Graph Encoder - GNN for Knowledge Graph encoding
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphEncoder(nn.Module):
    """
    Graph Neural Network for encoding intersection Knowledge Graph (Section 4.2.1)
    Encodes static intersection structure into node embeddings H_kg
    """
    def __init__(self, node_dim, hidden_dim, num_layers=3, num_edge_types=4):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input projection
        self.node_embed = nn.Linear(node_dim, hidden_dim)

        # Message passing layers
        self.convs = nn.ModuleList()
        self.edge_mlps = nn.ModuleList()

        for _ in range(num_layers):
            # Node update MLP
            self.convs.append(nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ))
            # Edge type embedding
            self.edge_mlps.append(nn.Embedding(num_edge_types, hidden_dim))

        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

    def forward(self, node_features, edge_index, edge_types=None):
        """
        Args:
            node_features: (batch_size, M, node_dim) - M facility nodes
            edge_index: (2, num_edges) or (batch_size, 2, num_edges) - graph connectivity
            edge_types: (num_edges,) or (batch_size, num_edges) - edge type indices
        Returns:
            H_kg: (batch_size, M, hidden_dim) - encoded graph node embeddings
        """
        # Handle both batched and unbatched inputs
        if len(node_features.shape) == 2:
            node_features = node_features.unsqueeze(0)

        batch_size, M, _ = node_features.shape

        # Initial embedding
        x = self.node_embed(node_features)  # (B, M, d)

        # Message passing
        for i in range(self.num_layers):
            x = self._message_passing_layer(x, edge_index, edge_types, i)
            x = self.norms[i](x)

        return x  # (B, M, d)

    def _message_passing_layer(self, x, edge_index, edge_types, layer_idx):
        """Single message passing step"""
        batch_size, M, d = x.shape

        # Aggregate messages from neighbors
        # For simplicity, assuming edge_index is (batch_size, 2, num_edges)
        if len(edge_index.shape) == 2:
            # Same graph for all batches
            edge_index = edge_index.unsqueeze(0).expand(batch_size, -1, -1)

        num_edges = edge_index.shape[-1]

        # Gather source and target node features
        src_idx = edge_index[:, 0, :]  # (B, num_edges)
        tgt_idx = edge_index[:, 1, :]  # (B, num_edges)

        # Get messages (using simple aggregation)
        messages = []
        for b in range(batch_size):
            src_nodes = x[b, src_idx[b], :]  # (num_edges, d)
            tgt_nodes = x[b, tgt_idx[b], :]  # (num_edges, d)

            # Add edge type information if provided
            if edge_types is not None:
                if len(edge_types.shape) == 1:
                    edge_emb = self.edge_mlps[layer_idx](edge_types)
                else:
                    edge_emb = self.edge_mlps[layer_idx](edge_types[b])
                src_nodes = src_nodes + edge_emb

            # Aggregate by target
            msg = torch.zeros(M, d, device=x.device)
            msg.index_add_(0, tgt_idx[b], src_nodes)
            messages.append(msg)

        messages = torch.stack(messages)  # (B, M, d)

        # Update node features
        x_new = self.convs[layer_idx](torch.cat([x, messages], dim=-1))

        return x_new


class KnowledgeGraphEncoder(nn.Module):
    """
    Higher-level wrapper for Knowledge Graph encoding
    Includes semantic attributes encoding for different facility types
    """
    def __init__(self, num_facility_types=10, facility_dim=32, hidden_dim=128, num_layers=3):
        super().__init__()

        # Facility type embedding
        self.facility_embed = nn.Embedding(num_facility_types, facility_dim)

        # Spatial coordinate embedding (2D positions)
        self.spatial_embed = nn.Linear(2, facility_dim)

        # Combined node feature encoder
        self.graph_encoder = GraphEncoder(
            node_dim=facility_dim * 2,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )

    def forward(self, facility_types, positions, edge_index, edge_types=None):
        """
        Args:
            facility_types: (batch_size, M) - facility type indices
            positions: (batch_size, M, 2) - spatial coordinates
            edge_index: graph connectivity
            edge_types: edge type indices
        Returns:
            H_kg: (batch_size, M, hidden_dim)
        """
        # Embed facility semantics
        type_emb = self.facility_embed(facility_types)  # (B, M, facility_dim)

        # Embed spatial positions
        pos_emb = self.spatial_embed(positions)  # (B, M, facility_dim)

        # Combine features
        node_features = torch.cat([type_emb, pos_emb], dim=-1)  # (B, M, 2*facility_dim)

        # Encode graph
        H_kg = self.graph_encoder(node_features, edge_index, edge_types)

        return H_kg

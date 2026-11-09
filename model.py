import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax, scatter
import numpy as np

class JointTransformer(nn.Module):
    """
    Graph Transformer layer that jointly processes node and edge features.
    """
    def __init__(self, in_channels: int, out_channels: int, edge_dim: int, heads: int = 8, dropout: float = 0.1):
        """
        :param in_channels: Dimension of input node features.
        :param out_channels: Dimension of output node features.
        :param edge_dim: Dimension of input edge features.
        :param heads: Number of attention heads.
        :param dropout: Dropout rate.
        """
        super().__init__()
        self.heads = heads
        self.out_channels = out_channels
        self.head_dim = out_channels // heads
        
        # Linear projections for query, key, and value
        self.q_lin = nn.Linear(in_channels, out_channels)
        self.k_lin = nn.Linear(in_channels, out_channels)
        self.v_lin = nn.Linear(in_channels, out_channels)
        
        # Linear projection for edge features
        self.edge_lin = nn.Linear(edge_dim, out_channels)
        
        # Output linear layer
        self.out_lin = nn.Linear(out_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the JointTransformer layer.
        
        :param x: Node features, shape [num_nodes, in_channels].
        :param edge_index: Graph connectivity, shape [2, num_edges].
        :param edge_attr: Edge features, shape [num_edges, edge_dim].
        :return: Output node features, shape [num_nodes, out_channels].
        """
        num_nodes = x.size(0)
        source, target = edge_index[0], edge_index[1]

        # 1. Project node features into query, key, value spaces
        q = self.q_lin(x).view(-1, self.heads, self.head_dim)
        k = self.k_lin(x).view(-1, self.heads, self.head_dim)
        v = self.v_lin(x).view(-1, self.heads, self.head_dim)

        # 2. Project edge features
        edge_features = self.edge_lin(edge_attr).view(-1, self.heads, self.head_dim)

        # 3. Select query, key, value for each edge
        q_i = q[target]  # Query of target nodes
        k_j = k[source]  # Key of source nodes
        v_j = v[source]  # Value of source nodes

        # 4. Compute attention scores
        # Edge features are added to the key vectors to incorporate edge information
        k_j_with_edge = k_j + edge_features
        
        # Scaled dot-product attention
        attn_scores = (q_i * k_j_with_edge).sum(dim=-1) / np.sqrt(self.head_dim)
        
        # 5. Normalize attention scores using softmax
        attn_weights = softmax(attn_scores, target, num_nodes=num_nodes)
        attn_weights = self.dropout(attn_weights)

        # 6. Aggregate information from neighbors (message passing)
        messages = v_j * attn_weights.unsqueeze(-1)
        
        # Use PyG's scatter (uses torch_scatter when available)
        out = scatter(messages, target, dim=0, dim_size=num_nodes, reduce='sum')
        
        # 7. Final output projection
        out = out.view(-1, self.out_channels)
        out = self.out_lin(out)
        
        return out

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv

from model import JointTransformer


class GNNEncoder(nn.Module):
    """GNN encoder with multiple architecture options."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        gnn_type: str = "gcn",
        dropout: float = 0.2,
        heads: int = 8,
        input_edge_dim=1,
        device: str = "cuda",
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.device = device
        self.gnn_type = gnn_type
        self.input_edge_dim = input_edge_dim

        for _ in range(num_layers):
            if gnn_type == "gcn":
                conv = GCNConv(hidden_dim, hidden_dim)
            elif gnn_type == "gat":
                if hidden_dim % heads != 0:
                    raise ValueError(
                        f"hidden_dim={hidden_dim} must be divisible by heads={heads} for gnn_type='gat'"
                    )
                conv = GATv2Conv(
                    hidden_dim,
                    hidden_dim // heads,
                    heads=heads,
                    dropout=dropout,
                    edge_dim=input_edge_dim,
                )
            elif gnn_type == "sage":
                conv = SAGEConv(hidden_dim, hidden_dim)
            elif gnn_type == "our":
                conv = JointTransformer(hidden_dim, hidden_dim, input_edge_dim)
            else:
                raise ValueError(f"Unsupported GNN type: {gnn_type}")
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.gate = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            if self.gnn_type == "our":
                if edge_attr is None:
                    raise ValueError("edge_attr must be provided for gnn_type 'our'")
                edge_attr = edge_attr.reshape(-1, self.input_edge_dim)
                out = F.relu(norm(conv(x, edge_index, edge_attr)))
            elif self.gnn_type == "gat":
                if edge_attr is not None:
                    edge_attr = edge_attr.reshape(-1, self.input_edge_dim)
                    out = F.relu(norm(conv(x, edge_index, edge_attr=edge_attr)))
                else:
                    out = F.relu(norm(conv(x, edge_index)))
            else:
                out = F.relu(norm(conv(x, edge_index)))
            gate_input = torch.cat([x, out], dim=-1)
            gate = torch.sigmoid(self.gate(gate_input))
            x = gate * out + (1 - gate) * x
        return x


class GraphLinkPredictor(nn.Module):
    """Contrastive learning model for link prediction on subgraphs."""

    def __init__(
        self,
        target_task: str,
        embedding_dim: int,
        hidden_dim: int,
        temperature: float = 0.1,
        dropout: float = 0.2,
        num_gnn_layers: int = 2,
        gnn_type: str = "gat",
        edge_dim: int = 1,
        pair_feature_dim: int = 0,
        em_use_interactions: bool = False,
        em_decoder_width_mult: float = 1.0,
        device: str = "cuda",
    ):
        super().__init__()
        self.target_task = target_task
        self.gnn_type = gnn_type
        self.pair_feature_dim = int(pair_feature_dim)
        self._decoder_pair_feature_dim = self.pair_feature_dim
        self.use_em_interactions = str(target_task) == "entity_matching" and bool(em_use_interactions)
        self.em_decoder_width_mult = float(em_decoder_width_mult)
        self.gnn_encoder = GNNEncoder(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers,
            gnn_type=gnn_type,
            dropout=dropout,
            input_edge_dim=edge_dim,
            device=device,
        )
        self.device = device

        if self.use_em_interactions:
            # Ditto-style pair interaction block: concat(u, v, |u-v|, u*v)
            edge_input_dim = hidden_dim * 4 + self._decoder_pair_feature_dim
        else:
            edge_input_dim = hidden_dim * 2 + self._decoder_pair_feature_dim

        decoder_hidden_dim = hidden_dim
        if str(target_task) == "entity_matching":
            width_mult = max(0.1, float(self.em_decoder_width_mult))
            decoder_hidden_dim = max(8, int(round(hidden_dim * width_mult)))

        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_input_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden_dim, decoder_hidden_dim),
        )

        self.projection_head = nn.Sequential(
            nn.Linear(decoder_hidden_dim, max(2, decoder_hidden_dim // 2)),
            nn.ReLU(),
            nn.Linear(max(2, decoder_hidden_dim // 2), max(2, decoder_hidden_dim // 4)),
        )

        self.link_predictor = nn.Linear(decoder_hidden_dim, 1)

    def forward(self, data, *, return_replay_tensors: bool = False):
        # Perform message passing on the sampled subgraph
        node_features = self.gnn_encoder(data.x, data.edge_index, data.edge_attr)

        # Extract embeddings for the source and destination nodes of the target links
        edge_label_index = data.edge_label_index
        src_nodes, dst_nodes = edge_label_index[0], edge_label_index[1]
        node1_emb = node_features[src_nodes]
        node2_emb = node_features[dst_nodes]
        if self.use_em_interactions:
            edge_embeddings = torch.cat(
                [
                    node1_emb,
                    node2_emb,
                    torch.abs(node1_emb - node2_emb),
                    node1_emb * node2_emb,
                ],
                dim=-1,
            )
        else:
            edge_embeddings = torch.cat([node1_emb, node2_emb], dim=-1)

        if self._decoder_pair_feature_dim > 0:
            pair_feats = getattr(data, "edge_pair_features", None)
            if pair_feats is None:
                pair_feats = torch.zeros(
                    (edge_embeddings.size(0), self._decoder_pair_feature_dim),
                    dtype=edge_embeddings.dtype,
                    device=edge_embeddings.device,
                )
            else:
                pair_feats = pair_feats.to(device=edge_embeddings.device, dtype=edge_embeddings.dtype)
            edge_embeddings = torch.cat([edge_embeddings, pair_feats], dim=-1)

        decoder_input = edge_embeddings
        h = self.edge_encoder(decoder_input)
        projections = F.normalize(self.projection_head(h), dim=1)
        link_logits = self.link_predictor(h).squeeze(-1)
        outputs = {"projections": projections, "link_logits": link_logits}
        if bool(return_replay_tensors):
            # Keep a detached copy for offline replay analysis.
            outputs["decoder_input"] = decoder_input
            outputs["edge_hidden"] = h
        return outputs


class ContrastiveLoss(nn.Module):
    """Supervised contrastive loss."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, projections, labels):
        valid_mask = labels != -1  # Assuming -1 is not a valid label
        if not valid_mask.any():
            return torch.tensor(0.0, device=projections.device, requires_grad=True)

        projections = projections[valid_mask]
        labels = labels[valid_mask]

        if projections.shape[0] < 2:
            return torch.tensor(0.0, device=projections.device, requires_grad=True)

        similarity_matrix = torch.matmul(projections, projections.T) / self.temperature
        # Positive pairs are those with the same label (i.e., both are true links)
        mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(0).T).float()
        mask.fill_diagonal_(0)

        # Only consider positive pairs for the loss numerator
        pos_mask = (mask * (labels.unsqueeze(0) == 1)).float()
        if pos_mask.sum() == 0:  # No positive pairs in batch
            return torch.tensor(0.0, device=projections.device, requires_grad=True)

        exp_sim = torch.exp(similarity_matrix)
        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))

        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-8)

        return -mean_log_prob_pos[labels == 1].mean()

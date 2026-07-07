"""Heterogeneous GATv2 model for metabolic network analysis."""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, GATv2Conv, Linear
from torch_geometric.data import HeteroData


class MetaGNNModel(nn.Module):
    """GATv2-based heterogeneous graph neural network for reaction node classification.
    
    Processes substrate_of and produces edges with optional shared_metabolite edges.
    Supports attention weight extraction and currency metabolite edge downscaling.
    """

    def __init__(
        self,
        num_reaction_features: int,
        num_metabolite_features: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.2,
    ):
        """Initialize the MetaGNN model.
        
        Args:
            num_reaction_features: Dimension of reaction node features.
            num_metabolite_features: Dimension of metabolite node features.
            hidden_dim: Hidden dimension for graph convolutions.
            num_layers: Number of heterogeneous convolution layers.
            num_heads: Number of attention heads in GATv2.
            dropout: Dropout probability.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout

        self.reaction_proj = Linear(num_reaction_features, hidden_dim)
        self.metabolite_proj = Linear(num_metabolite_features, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    ("metabolite", "substrate_of", "reaction"): GATv2Conv(
                        hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout
                    ),
                    ("reaction", "produces", "metabolite"): GATv2Conv(
                        hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout
                    ),
                    ("reaction", "shared_metabolite", "reaction"): GATv2Conv(
                        hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout
                    ),
                },
                aggr="mean",
            )
            self.convs.append(conv)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.store_attention = False
        self.attention_weights = {}

    def forward(
        self, data: HeteroData, return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the heterogeneous graph.
        
        Args:
            data: Heterogeneous graph data with node and edge features.
            return_attention: Whether to return attention weight statistics.
            
        Returns:
            Dictionary with 'scores' (reaction predictions) and optionally 'attention_weights'.
        """
        x_dict = {
            "reaction": self.reaction_proj(data["reaction"].x),
            "metabolite": self.metabolite_proj(data["metabolite"].x),
        }
        
        edge_index_dict = data.edge_index_dict
        edge_attr_dict = data.edge_attr_dict if hasattr(data, "edge_attr_dict") else {}
        
        if return_attention:
            self.store_attention = True
            self.attention_weights = {}

        for layer_idx, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            
            for key in x_dict:
                x_dict[key] = self.relu(x_dict[key])
                x_dict[key] = self.dropout(x_dict[key])

        reaction_embeddings = x_dict["reaction"]
        scores = self.mlp_head(reaction_embeddings)

        output = {"scores": scores.squeeze(-1)}
        
        if return_attention and self.attention_weights:
            output["attention_weights"] = self.attention_weights

        return output

    def apply_currency_downscaling(
        self, edge_weights: torch.Tensor, currency_mask: torch.Tensor, scale_factor: float = 0.1
    ) -> torch.Tensor:
        """Apply downscaling to currency metabolite edges.
        
        Args:
            edge_weights: Original edge weights.
            currency_mask: Boolean mask indicating currency metabolite edges.
            scale_factor: Scaling factor for currency edges (default 0.1).
            
        Returns:
            Scaled edge weights.
        """
        scaled_weights = edge_weights.clone()
        scaled_weights[currency_mask] *= scale_factor
        return scaled_weights

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv
from .tgat import TGATConv

class APTModel(nn.Module):
    """
    GraphSAGE + T-GAT model for APT detection.
    This model can flexibly include multiple GraphSAGE layers and T-GAT layers.
    If use_tgat is False, the model uses GraphSAGE (and optional GAT) only.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_sage_layers=1, num_tgat_layers=1, use_tgat=True, edge_dim=None, heads=1, gamma=1.0):
        super(APTModel, self).__init__()
        self.use_tgat = use_tgat
        self.num_sage_layers = num_sage_layers
        self.num_tgat_layers = num_tgat_layers if use_tgat else 0

        # Initialize GraphSAGE layers
        self.sage_convs = nn.ModuleList()
        prev_channels = in_channels
        for i in range(num_sage_layers):
            # For GraphSAGE layers, output dimension is hidden_channels
            out_channels_i = hidden_channels
            # If not using T-GAT, and this is last layer, output directly to class count
            if not use_tgat and i == num_sage_layers - 1:
                out_channels_i = out_channels
            self.sage_convs.append(SAGEConv(prev_channels, out_channels_i, normalize=True))
            prev_channels = out_channels_i

        # Initialize T-GAT layers (including final classification layer if use_tgat)
        self.tgat_convs = nn.ModuleList()
        if use_tgat:
            tgat_in_channels = prev_channels
            # Add (num_tgat_layers - 1) hidden T-GAT layers if more than one T-GAT layer
            for j in range(num_tgat_layers - 1):
                # Hidden T-GAT layers use hidden_channels output (with heads concatenated if multiple heads)
                self.tgat_convs.append(TGATConv(tgat_in_channels, hidden_channels, heads=heads, concat=True, edge_dim=edge_dim, gamma=gamma))
                # After a TGATConv with concat=True, the output dimension is hidden_channels * heads
                tgat_in_channels = hidden_channels * heads
            # Final T-GAT layer for output (classification)
            self.final_tgat = TGATConv(tgat_in_channels, out_channels, heads=heads, concat=False, edge_dim=edge_dim, gamma=gamma)
        else:
            self.final_tgat = None

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # Use edge attributes if available
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None

        # Apply GraphSAGE layers
        for i, conv in enumerate(self.sage_convs):
            if not self.use_tgat and i == len(self.sage_convs) - 1:
                # Last GraphSAGE layer in a model without T-GAT: output logits
                x = conv(x, edge_index)
            else:
                x = F.relu(conv(x, edge_index))

        # Apply T-GAT layers if enabled
        if self.use_tgat:
            for j, conv in enumerate(self.tgat_convs):
                x = F.relu(conv(x, edge_index, edge_attr))
            out = self.final_tgat(x, edge_index, edge_attr)
        else:
            out = x

        return F.log_softmax(out, dim=-1)

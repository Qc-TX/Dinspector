import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class TGATConv(nn.Module):
    """
    Threat Graph Attention Network (T-GAT) layer.
    This layer extends the graph attention mechanism to incorporate edge features.
    Parameter `gamma` controls the contribution of edge features in attention computation.
    """
    # tips:This part of the code will be open sourced after the article is published
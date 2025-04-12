import os
import torch
from torch_geometric.data import Data

def parse_edge_list(file_path, id_map=None, node_type_map=None, edge_type_map=None):
    """
    Parse an edge list file and return graph data in PyG Data format along with mapping info.
    Each line in file is expected in format: src_id<TAB>src_type<TAB>dst_id<TAB>dst_type<TAB>edge_type
    """


def load_dataset(name, scene=None):
    """
    Load the specified dataset and return train/test data along with feature and class counts.
    """

# tips:This part of the code will be open sourced after the article is published
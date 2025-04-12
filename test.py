import os
import argparse
import torch
import torch.nn.functional as F
from datasets import loader
from models.apt_model import APTModel

# Testing script for APT detection model
parser = argparse.ArgumentParser(description="Test the APT detection model and generate alert report.")
parser.add_argument('--dataset', type=str, required=True, choices=['darpatc', 'streamspot', 'unicorn', 'darpa'], help='Dataset name to test on')
parser.add_argument('--scene', type=str, default=None, help='Scene name (if dataset is DARPA)')
parser.add_argument('--model_path', type=str, default=None, help='Path to trained model checkpoint (.pth). If not specified, use default in checkpoints/')
parser.add_argument('--gpu', action='store_true', help='Use GPU if available for inference')
args = parser.parse_args()

device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load dataset (train+test) to build graph and mappings
train_data_list, test_data_list, feature_dim, num_classes, edge_dim, id_map, node_type_map = loader.load_dataset(args.dataset, scene=args.scene)
data = (test_data_list[0] if len(test_data_list) > 0 else train_data_list[0]).to(device)

# Initialize model and load weights
model = APTModel(in_channels=feature_dim, hidden_channels=128, out_channels=num_classes,
                 num_sage_layers=1, num_tgat_layers=1, use_tgat=True, edge_dim=edge_dim, heads=8, gamma=1.0).to(device)
model_file = args.model_path if args.model_path else os.path.join('checkpoints', f"{args.dataset}{('_' + args.scene) if args.scene else ''}_model.pth")
if not os.path.exists(model_file):
    raise FileNotFoundError(f"Model checkpoint not found: {model_file}")
model.load_state_dict(torch.load(model_file, map_location=device))
model.eval()

# Run inference
with torch.no_grad():
    out = model(data)
    probs = torch.exp(out)  # convert log-softmax to probability
preds = torch.argmax(probs, dim=1)

# Define threshold for anomaly detection
threshold = 1.5
thre_map = {
    'darpatc': {'cadets': 1.5, 'trace': 1.0, 'theia': 1.5, 'fivedirections': 1.0},
    'darpa': {'cadets': 1.5, 'trace': 1.0, 'theia': 1.5, 'fivedirections': 1.0},
    'streamspot': 2.0,
    'unicorn': 2.0
}
ds_key = 'darpatc' if args.dataset.lower() in ['darpa', 'darpatc'] else args.dataset.lower()
if ds_key in ['darpatc', 'darpa'] and args.scene:
    threshold = thre_map['darpatc'].get(args.scene, 1.5)
elif ds_key in thre_map and isinstance(thre_map[ds_key], float):
    threshold = thre_map[ds_key]

# Identify anomalies based on confidence (ratio of top1 to top2 prob)
top2_vals, top2_idx = torch.topk(probs, 2, dim=1)
top1 = top2_vals[:, 0]
top2 = top2_vals[:, 1]
ratios = top1 / (top2 + 1e-9)
anomalies = (ratios < threshold).nonzero(as_tuple=True)[0].tolist()

# Prepare mappings for report
id_map_inv = {v: k for k, v in id_map.items()}
node_type_map_inv = {v: k for k, v in node_type_map.items()}

# Output alert report
if not anomalies:
    print("No anomalies detected.")
else:
    print(f"Detected {len(anomalies)} anomaly(s) in graph:")
    for node_idx in anomalies:
        orig_id = id_map_inv.get(node_idx, str(node_idx))
        pred_label_idx = preds[node_idx].item()
        pred_label = node_type_map_inv.get(pred_label_idx, str(pred_label_idx))
        print(f" - Node {orig_id} (predicted type: {pred_label}) flagged as ANOMALOUS (low confidence)")

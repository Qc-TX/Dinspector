import os
import argparse
import torch
import torch.nn.functional as F
from datasets import loader
from models.apt_model import APTModel

# Training script for APT detection model
parser = argparse.ArgumentParser(description="Train the APT detection model.")
parser.add_argument('--dataset', type=str, required=True, choices=['darpatc', 'streamspot', 'unicorn'], help='Dataset name')
parser.add_argument('--scene', type=str, default=None, help='Scene name (required for DARPA dataset)')
parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=0, help='Batch size (0 for full batch)')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension for GraphSAGE layers')
parser.add_argument('--sage_layers', type=int, default=1, help='Number of GraphSAGE layers')
parser.add_argument('--tgat_layers', type=int, default=1, help='Number of T-GAT layers (if use_tgat is enabled)')
parser.add_argument('--no_tgat', action='store_true', help='Disable using T-GAT (use GraphSAGE-only model)')
parser.add_argument('--gamma', type=float, default=1.0, help='Gamma parameter for T-GAT layer')
parser.add_argument('--heads', type=int, default=8, help='Number of attention heads for T-GAT')
parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
parser.add_argument('--config', type=str, default=None, help='Path to YAML config file for default parameters')
args = parser.parse_args()

# If a config file is provided, override defaults with it
if args.config:
    try:
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    except Exception as e:
        print(f"Warning: could not load config file: {e}")

# Select device
device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load dataset
train_data_list, test_data_list, feature_dim, num_classes, edge_dim, id_map, node_type_map = loader.load_dataset(args.dataset, scene=args.scene)
data = train_data_list[0].to(device)

# Initialize model
model = APTModel(in_channels=feature_dim, hidden_channels=args.hidden_dim, out_channels=num_classes,
                 num_sage_layers=args.sage_layers, num_tgat_layers=args.tgat_layers,
                 use_tgat=(not args.no_tgat), edge_dim=edge_dim, heads=args.heads, gamma=args.gamma).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# Training loop
model.train()
for epoch in range(1, args.epochs + 1):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask].to(device))
    loss.backward()
    optimizer.step()
    pred = out[data.train_mask].max(dim=1)[1]
    acc = pred.eq(data.y[data.train_mask].to(device)).sum().item() / data.train_mask.sum().item()
    print(f"Epoch {epoch:03d}: Loss = {loss.item():.4f}, Train Accuracy = {acc:.4f}")

# Save model checkpoint
os.makedirs('checkpoints', exist_ok=True)
model_name = f"{args.dataset}{('_' + args.scene) if args.scene else ''}_model.pth"
model_path = os.path.join('checkpoints', model_name)
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

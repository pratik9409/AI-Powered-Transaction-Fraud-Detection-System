import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import os

class FraudGNN(nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(FraudGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, 1)
        
    def forward(self, x, edge_index):
        # Node embeddings
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        
        # Graph-level classification
        x = torch.mean(x, dim=0)  # Global mean pooling
        x = self.classifier(x)
        return torch.sigmoid(x)

def load_gnn_model(model_path='trained_models/gnn_model.pt', device='cpu'):
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Initialize model
    model = FraudGNN(num_node_features=32, hidden_channels=64)
    
    try:
        # Try to load pretrained weights
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded GNN model from {model_path}")
    except FileNotFoundError:
        # If no model exists, initialize with random weights and save
        print(f"No model found at {model_path}, creating new model")
        torch.save(model.state_dict(), model_path)
        print(f"New model saved to {model_path}")
    
    model.to(device)
    model.eval()
    return model
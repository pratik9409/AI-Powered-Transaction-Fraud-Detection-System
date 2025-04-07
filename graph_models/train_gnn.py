import torch
from torch_geometric.data import Data
from gnn_model import FraudGNN
import os

def train_and_save_gnn_model():
    # Create sample data for demonstration
    num_nodes = 100
    num_features = 32
    x = torch.randn((num_nodes, num_features))
    edge_index = torch.randint(0, num_nodes, (2, 200))
    y = torch.randint(0, 2, (1,)).float()
    
    # Initialize model
    model = FraudGNN(num_node_features=num_features, hidden_channels=64)
    
    # Simple training loop (for demonstration)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    for epoch in range(10):
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
    
    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'trained_models/gnn_model.pt')
    print("GNN model saved to models/gnn_model.pt")

if __name__ == '__main__':
    train_and_save_gnn_model()
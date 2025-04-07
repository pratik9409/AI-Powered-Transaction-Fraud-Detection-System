import torch
from torch_geometric.data import Data
from collections import defaultdict

class TransactionGraphBuilder:
    def __init__(self):
        self.node_index = defaultdict(int)
        self.current_id = 0
        self.edges = []
        self.node_features = []
        self.node_types = []
        
    def get_node_id(self, node_key, node_type):
        if node_key not in self.node_index:
            self.node_index[node_key] = self.current_id
            self.current_id += 1
            # Simple feature representation
            self.node_features.append([1.0 if i == node_type else 0.0 for i in range(3)])
            self.node_types.append(node_type)
        return self.node_index[node_key]
    
    def add_transaction(self, transaction):
        # Account node (type 0)
        acc_id = self.get_node_id(transaction['AccountID'], 0)
        # Merchant node (type 1)
        merchant_id = self.get_node_id(transaction['MerchantID'], 1)
        # Device node (type 2)
        device_id = self.get_node_id(transaction['DeviceID'], 2)
        
        # Add edges
        self.edges.append((acc_id, merchant_id))
        self.edges.append((acc_id, device_id))
        
        # Convert to PyG format
        edge_index = torch.tensor(list(zip(*self.edges)), dtype=torch.long)
        x = torch.tensor(self.node_features, dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index)
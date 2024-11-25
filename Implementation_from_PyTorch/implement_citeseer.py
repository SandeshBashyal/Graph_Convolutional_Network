import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import add_self_loops, degree


dataset = Planetoid(root = 'Citeseer', name = 'Citeseer')
data = dataset[0]

# Step 1: Precompute normalized adjacency matrix
def normalize_adjacency(edge_index, num_nodes):
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    row, col = edge_index
    deg = degree(row, num_nodes=num_nodes, dtype=torch.float32)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    return edge_index, norm

edge_index, norm = normalize_adjacency(data.edge_index, data.num_nodes)

# Step 2: Define GCN model
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.W1 = torch.nn.Parameter(torch.empty(input_dim, hidden_dim))
        self.W2 = torch.nn.Parameter(torch.empty(hidden_dim, output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.W1)
        torch.nn.init.xavier_uniform_(self.W2)

    def forward(self, x, edge_index, norm):
        # Layer 1
        x = torch.mm(x, self.W1)
        row, col = edge_index
        x = torch.sparse.mm(torch.sparse_coo_tensor(edge_index, norm, (x.size(0), x.size(0))), x)
        x = F.relu(x)

        # Layer 2
        x = torch.mm(x, self.W2)
        x = torch.sparse.mm(torch.sparse_coo_tensor(edge_index, norm, (x.size(0), x.size(0))), x)
        return F.log_softmax(x, dim=1)

# Initialize model, optimizer, and data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(input_dim=dataset.num_node_features, hidden_dim=64, output_dim=dataset.num_classes).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

# Training loop
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, edge_index, norm)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Testing loop
def test():
    model.eval()
    logits = model(data.x, edge_index, norm)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

# Training and evaluation
best_val_acc = 0
for epoch in range(200):
    loss = train()
    train_acc, val_acc, test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_test_acc = test_acc
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')

print(f'Best Validation Accuracy: {best_val_acc:.4f}, Test Accuracy at Best Val: {best_test_acc:.4f}')
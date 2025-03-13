import pandas as pd
import numpy as np
import torch 
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, DBLP
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = GCNConv(128,16)
        self.conv2 = GCNConv(16, 4)
    def forward(self,data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

# dataset = Planetoid(root='data_Cora/', name = 'Cora')
dataset = DBLP(root='data_DBLP')
device = torch.device('cuda')
model = GCN().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

model.train()
for epoch in range(500):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask],data.y[data.train_mask])
    loss.backward()
    optimizer.step()
print(data.y[data.train_mask].size())
model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')
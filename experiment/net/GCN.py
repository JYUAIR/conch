import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


class GCN(nn.Module):
    def __init__(self, emb_size: int, out_size: int, dropout: float):
        super(GCN, self).__init__()
        self.emb_size = emb_size
        self.dropout = dropout
        self.out_size = out_size
        self.conv1 = GCNConv(self.emb_size, 16)
        self.conv2 = GCNConv(16, self.out_size)

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        out = self.conv1(x, edge_index)
        out = F.relu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.conv2(out, edge_index)

        return F.log_softmax(out, dim=1)


if __name__ == '__main__':
    from torch_geometric.datasets import Planetoid

    dataset = Planetoid(root='~/Downloads', name='Cora')
    device = torch.device('cpu')
    model = GCN(dataset.num_node_features, dataset.num_classes, 0.5).to(device)
    data = dataset.data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        print(f'epoch: {epoch + 1}; loss: {loss}')
        loss.backward()
        optimizer.step()

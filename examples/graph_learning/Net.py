'''
q_function.py

define the Q function Q(s, a). Q(s, a): maps the graph of s to value of each action.
'''

import torch
import torch.nn.functional as F
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
from math import ceil

'''
class GNN:

define a GNN module

parameters:
    in_channels: number of feature channels for each input node
    hidden_channels: number of feature channels for each hidden node
    batch_normalization: if add a batch normalization after each conv layer
'''


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, batch_normalization=False, add_loop=False, lin=True):
        super(GNN, self).__init__()

        self.add_loop = add_loop
        self.batch_normalization = batch_normalization

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels, normalize)

        if self.batch_normalization:
            self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
            self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
            self.bn3 = torch.nn.BatchNorm1d(out_channels)

        if lin is True:
            self.lin = torch.nn.Linear(2 * hidden_channels + out_channels,
                                       out_channels)
        else:
            self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_features = x.size()

        x = x.view(-1, num_features)
        x = getattr(self, 'bn{}'.format(i))(x)
        x = x.view(batch_size, num_nodes, num_features)
        return x

    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()

        if self.add_loop:
            # Replicate feature removed from PyTorch Geometric
            N = adj.size()[1]
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[:, idx, idx] = 1

        x0 = x
        if self.batch_normalization:
            x1 = self.bn(1, F.relu(self.conv1(x0, adj, mask)))
            x2 = self.bn(2, F.relu(self.conv2(x1, adj, mask)))
            x3 = self.bn(3, F.relu(self.conv3(x2, adj, mask)))
        else:
            x1 = F.relu(self.conv1(x0, adj, mask))
            x2 = F.relu(self.conv2(x1, adj, mask))
            x3 = F.relu(self.conv3(x2, adj, mask))

        x = torch.cat([x1, x2, x3], dim=-1)

        if self.lin is not None:
            x = F.relu(self.lin(x))

        return x


'''
class Net

define a graph network structure for Q function

parameters:
    max_nodes: maximum number of nodes in the graph
    num_channels: number of feature channels for each input graph node
    num_outputs: size of the action space
'''


class Net(torch.nn.Module):
    def __init__(self, max_nodes, num_channels, num_outputs, layer_size=64, batch_normalization=False):
        super(Net, self).__init__()

        num_nodes = ceil(0.25 * max_nodes)
        self.gnn1_pool = GNN(num_channels, layer_size, num_nodes, batch_normalization=batch_normalization, add_loop=True)
        self.gnn1_embed = GNN(num_channels, layer_size, layer_size, batch_normalization=batch_normalization, add_loop=True, lin=False)

        num_nodes = ceil(0.25 * num_nodes)
        self.gnn2_pool = GNN(3 * layer_size, layer_size, num_nodes, batch_normalization=batch_normalization)
        self.gnn2_embed = GNN(3 * layer_size, layer_size, layer_size, batch_normalization=batch_normalization, lin=False)

        self.gnn3_embed = GNN(3 * layer_size, layer_size, layer_size, batch_normalization=batch_normalization, lin=False)

        self.lin1 = torch.nn.Linear(3 * layer_size, layer_size)
        self.lin2 = torch.nn.Linear(layer_size, num_outputs)

    def forward(self, x, adj, mask=None):
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)

        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)

        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        x = self.gnn3_embed(x, adj)

        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))

        x = self.lin2(x)

        return x, l1 + l2, e1 + e2

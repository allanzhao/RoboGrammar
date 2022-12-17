import os.path as osp
from math import ceil

import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.data import DenseDataLoader
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
import IPython 
import sys
torch.set_printoptions(threshold=sys.maxsize)
from examples.graph_learning import parse_log_file
import numpy as np
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import Data
import pickle

load_data = True
variational = True


max_nodes = 17

def estimate_vars(all_link_features, all_link_adj, all_rewards):
  key_dict = {}
  for feat, adj, rew in zip(all_link_features, all_link_adj, all_rewards):
    key = (feat.tostring(), adj.tostring())
    try:
      key_dict[key].append(rew)
    except:
      key_dict[key] = [rew]
    

  std_dict = {}
  for key in key_dict:
    std_dict[key] = np.std(key_dict[key])
  #TODO:
  #1. Compute std of each key
  #2. Re-loop over every key feature in order (maybe store in a loop on the first forward pass) and then return the stds in the same order
  #3. In the evaluation section, return difference of output variance and the sample variance as a proxy for the true variance
  return std_dict
  

class MyFilter(object):
    def __call__(self, data):
        return data.num_nodes <= max_nodes


path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
                'PROTEINS_dense')
dataset = TUDataset(path, name='PROTEINS', transform=T.ToDense(max_nodes),
                    pre_filter=MyFilter())

print('path = ', path)

num_channels = 31
if not load_data:
  all_link_features, all_link_adj, all_rewards = parse_log_file.main('flat_jan21.csv', 'data/designs/grammar_jan21.dot')
  std_dict = estimate_vars(all_link_features, all_link_adj, all_rewards)
  
  if variational:
    all_rewards = [(reward,) for reward in all_rewards]
  else:
    all_rewards = [(reward,) for reward in all_rewards]

  #xperimental postprocessing
  #step 1: make symmetric
  all_link_adj_symmetric = [link_adj + np.transpose(link_adj) for link_adj in all_link_adj]

  #step 2: Add blank rows, pad with 0s, and fill out mask:
  #max length:
  max_nodes = max([feat.shape[0] for feat in all_link_features])

  def pad(array, shape):
      """
      array: Array to be padded
      reference: Reference array with the desired shape
      offsets: list of offsets (number of elements must be equal to the dimension of the array)
      """
      # Create an array of zeros with the reference shape
      result = np.zeros(shape)
      if len(shape) == 1:
        result[:array.shape[0], :] = array # ERROR: why result is 2d
      elif len(shape) == 2:
        result[:array.shape[0], :array.shape[1]] = array
      else:
        raise Exception('only 1 and 2d supported for now')
      return result
      
  all_link_adj_symmetric_pad = [pad(adj, (max_nodes, max_nodes)) for adj in all_link_adj_symmetric]
  all_features_pad = [pad(feat, (max_nodes, feat.shape[1])) for feat in all_link_features]

  def create_mask(feat, max_nodes):
    return np.array([True if i < feat.shape[0] else False for i in range(max_nodes)])

  all_masks = [create_mask(feat, max_nodes) for feat in all_link_features]
  #num_channels = all_features_pad[0].shape[1]
  
  #step 3: Create dataset object
  data = [Data(adj=torch.from_numpy(adj).float(),
               mask=torch.from_numpy(mask),
               x=torch.from_numpy(x[:, :num_channels]).float(), 
               y=torch.from_numpy(np.array([y])).float(),
               std=torch.from_numpy(np.array([std_dict[std]])).float() ) for adj, mask, x, y, std in zip(all_link_adj_symmetric_pad, all_masks, all_features_pad, all_rewards, std_dict)]
  import random
  random.shuffle(data)
                                
  dataset = dataset.shuffle()
  n = (len(dataset) + 9) // 10
  test_dataset = data[:n]
  val_dataset = data[n:2 * n]
  train_dataset = data[2 * n:]
  
  with open('test_loader', 'wb') as test_file, open('val_loader', 'wb') as val_file, open('train_loader', 'wb') as train_file: 
    pickle.dump(test_dataset, test_file)
    pickle.dump(train_dataset, train_file)
    pickle.dump(val_dataset, val_file)
  
else:
  with open('test_loader', 'rb') as test_file, open('val_loader', 'rb') as val_file, open('train_loader', 'rb') as train_file: 
    test_dataset = pickle.load(test_file)
    train_dataset = pickle.load(train_file)
    val_dataset = pickle.load(val_file)
    
test_loader = DenseDataLoader(test_dataset, batch_size=20)
val_loader = DenseDataLoader(val_dataset, batch_size=20)
train_loader = DenseDataLoader(train_dataset, batch_size=20)




class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, add_loop=False, lin=True):
        super(GNN, self).__init__()

        self.add_loop = add_loop

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels, normalize)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        if lin is True:
            self.lin = torch.nn.Linear(2 * hidden_channels + out_channels,
                                       out_channels)
        else:
            self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, 'bn{}'.format(i))(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()

        x0 = x
        #IPython.embed()
        x1 = self.bn(1, F.relu(self.conv1(x0, adj, mask, self.add_loop)))
        x2 = self.bn(2, F.relu(self.conv2(x1, adj, mask, self.add_loop)))
        x3 = self.bn(3, F.relu(self.conv3(x2, adj, mask, self.add_loop)))

        x = torch.cat([x1, x2, x3], dim=-1)
        
        if self.lin is not None:
            x = F.relu(self.lin(x))

        return x


class Net(torch.nn.Module):
    def __init__(self, variational=False):
        super(Net, self).__init__()
        
        self.variational = variational
        
        

        num_nodes = ceil(0.25 * max_nodes)
        self.gnn1_pool = GNN(num_channels, 64, num_nodes, add_loop=True)
        self.gnn1_embed = GNN(num_channels, 64, 64, add_loop=True, lin=False)
        
        num_nodes = ceil(0.25 * num_nodes)
        self.gnn2_pool = GNN(3 * 64, 64, num_nodes)
        self.gnn2_embed = GNN(3 * 64, 64, 64, lin=False)

        self.gnn3_embed = GNN(3 * 64, 64, 64, lin=False)

        self.lin1 = torch.nn.Linear(3 * 64, 64)
        
        out_channels = 2 if self.variational else 1
        
        self.lin2 = torch.nn.Linear(64, out_channels)
        

    def forward(self, x, adj, mask=None):
        
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)

        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)
 
        #print('time for gnn2')

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)

        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        x = self.gnn3_embed(x, adj)

        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))

        x = self.lin2(x)
        
        return x, l1 + l2, e1 + e2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(variational).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


def train(epoch):
    model.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output, _, _ = model(data.x, data.adj, data.mask)        
        if not model.variational:
          loss = F.mse_loss(output[:, 0], data.y.view(-1))
        else:
          means = output[:, 0]
          stds = torch.exp(output[:, 1])
          dist = torch.distributions.Normal(means, stds)

          regularizer = 0.5 * torch.mean(torch.exp(stds) + means**2 - 1.0 - stds)
          loss = torch.mean(-dist.log_prob(data.y.view(-1))) + 0.001 * regularizer  
        loss.backward()
        loss_all += loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)


@torch.no_grad()
def test(loader, size):
    model.eval()
    error = 0.
    for data in loader:
        data = data.to(device)
        pred = model(data.x, data.adj, data.mask)[0]
        error += F.mse_loss(pred[:, 0], data.y.view(-1))
        pred_std = pred[:, 1]
        true_std = data.std.view(-1)
        #print('predicted std is ', torch.exp(pred_std))
        #print('true std is ', true_std)
        #print('prediction is ', pred[:, 0])
        #print('truth is ', data.y.view(-1))
        #correct += pred.eq(data.y.view(-1)).sum().item()
    return error / size


best_val_acc = test_acc = 10000.0
for epoch in range(1, 151):
    train_loss = train(epoch)
    train_acc = test(train_loader, len(train_loader))
    val_acc = test(val_loader, len(val_loader))
    if val_acc < best_val_acc:
        test_acc = test(test_loader, len(test_loader))
        best_val_acc = val_acc
    print('Epoch: {:03d}, Train Loss: {:.7f}, '
          'Val Acc: {:.7f}, Test Acc: {:.7f}, '
          'Train Acc: {:.7f}'.format(epoch, train_loss,
                                                     val_acc, test_acc,
                                                                      train_acc))

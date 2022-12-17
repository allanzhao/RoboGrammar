'''
train_robot_value_prediction.py

use Graph Neural Network to learn a value function for robot designs.

Learned GNN: 
input: a graph of the robot design with all nodes being terminal nodes.
output: the predicted reward for the design.

Argument:
--dataset-name: the name of the dataset
--use-cuse: if use gpu to train, [default is False]
'''

import sys
import os

project_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
current_dir = os.path.dirname(os.path.abspath(__file__))

# import python packages
from math import ceil
import time
import numpy as np
import pickle
import argparse
import random
from copy import deepcopy

# import third-party packages
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.data import DenseDataLoader
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import Data
torch.set_printoptions(threshold=sys.maxsize)

import IPython 

# import our packages
import parse_log_file
from common import *

# parse argument
parser = argparse.ArgumentParser()
parser.add_argument('--dataset-name', type = str, default = 'flat_mar23')
parser.add_argument('--use-cuda', default = False, action = 'store_true')
parser.add_argument('--save-interval', type = int, default = 10)
parser.add_argument('--load-path', type = str, default = None)
parser.add_argument('--test', default = False, action = 'store_true')

args = parser.parse_args()
dataset_name = args.dataset_name
save_dir = os.path.join(current_dir, 'trained_models', 'value_function', dataset_name, get_time_stamp())

# define if construct dataset from raw data
load_data = False

# dataset path 
dataset_dir = os.path.join(current_dir, 'data', dataset_name)
testset_path = os.path.join(dataset_dir, 'test_loader')
valset_path = os.path.join(dataset_dir, 'val_loader')
trainset_path = os.path.join(dataset_dir, 'train_loader')
testset_path = os.path.join(dataset_dir, 'test_loader')

# load pre-prosessed dataset. build if not exists
if os.path.isfile(testset_path) and os.path.isfile(valset_path) and os.path.isfile(trainset_path):
    with open(testset_path, 'rb') as test_file, open(valset_path, 'rb') as val_file, open(trainset_path, 'rb') as train_file: 
        test_dataset = pickle.load(test_file)
        train_dataset = pickle.load(train_file)
        val_dataset = pickle.load(val_file)
    max_nodes = 19
    num_features = 32 # 31 for previous configuration and 32 for current one (including torque)
else:
    os.makedirs(dataset_dir, exist_ok = True)

    raw_dataset_path = os.path.join(current_dir, 'data', args.dataset_name + '.csv')

    all_link_features, all_link_adj, all_rewards \
        = parse_log_file.main(raw_dataset_path, os.path.join(project_dir, 'data/designs/grammar_jan21.dot'))

    # exprimental postprocessing
    # step 1: make symmetric
    all_link_adj_symmetric = [link_adj + np.transpose(link_adj) for link_adj in all_link_adj]

    # step 2: Add blank rows, pad with 0s, and fill out mask:
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
    num_features = all_features_pad[0].shape[1]

    print('max-nodes = ', max_nodes, ', num-features = ', num_features)
    #step 3: Create dataset object
    data = [Data(adj=torch.from_numpy(adj).float(),
                mask=torch.from_numpy(mask),
                x=torch.from_numpy(x[:, :num_features]).float(), 
                y=torch.from_numpy(np.array([y])).float() ) for adj, mask, x, y in zip(all_link_adj_symmetric_pad, all_masks, all_features_pad, all_rewards)]
    random.shuffle(data)

    n_test = (len(data) + 2) // 3
    random.shuffle(data)
    known_dataset = data[:-n_test]
    unknown_dataset = data[-n_test:]
    # random.shuffle(known_dataset)
    n_val = (len(known_dataset) + 9) // 10
    train_dataset = known_dataset[:-n_val]
    val_dataset = known_dataset[-n_val:]
    test_dataset = unknown_dataset
    
    with open(testset_path, 'wb') as test_file, open(valset_path, 'wb') as val_file, open(trainset_path, 'wb') as train_file: 
        pickle.dump(test_dataset, test_file)
        pickle.dump(train_dataset, train_file)
        pickle.dump(val_dataset, val_file)

# print dataset info
print_info('dataset loaded:')
print_info('size of training set: {}'.format(len(train_dataset)))
print_info('size of validation set: {}'.format(len(val_dataset)))
print_info('size of testing set: {}'.format(len(test_dataset)))

y_min, y_max = 10000000.0, -10000000.0
for data in train_dataset:
    y_min = min(y_min, data.y[0])
    y_max = max(y_max, data.y[0])
print('training set: y min = ', y_min, ', y max = ', y_max)
y_min, y_max = 10000000.0, -10000000.0
for data in test_dataset:
    y_min = min(y_min, data.y[0])
    y_max = max(y_max, data.y[0])
print('testing set: y min = ', y_min, ', yy max = ', y_max)

# constrct batch
test_loader = DenseDataLoader(test_dataset, batch_size=20)
val_loader = DenseDataLoader(val_dataset, batch_size=20)
# random.shuffle(train_dataset)
train_loader = DenseDataLoader(train_dataset, batch_size=20)

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

        x0 = x
        #IPython.embed()
        if self.batch_normalization:
            x1 = self.bn(1, F.relu(self.conv1(x0, adj, mask, self.add_loop)))
            x2 = self.bn(2, F.relu(self.conv2(x1, adj, mask, self.add_loop)))
            x3 = self.bn(3, F.relu(self.conv3(x2, adj, mask, self.add_loop)))
        else:
            x1 = F.relu(self.conv1(x0, adj, mask, self.add_loop))
            x2 = F.relu(self.conv2(x1, adj, mask, self.add_loop))
            x3 = F.relu(self.conv3(x2, adj, mask, self.add_loop))

        x = torch.cat([x1, x2, x3], dim=-1)
        
        if self.lin is not None:
            x = F.relu(self.lin(x))

        return x


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        batch_normalization = False

        num_nodes = ceil(0.25 * max_nodes)
        self.gnn1_pool = GNN(num_features, 64, num_nodes, batch_normalization = batch_normalization, add_loop=True)
        self.gnn1_embed = GNN(num_features, 64, 64, batch_normalization = batch_normalization, add_loop=True, lin=False)
        
        num_nodes = ceil(0.25 * num_nodes)
        self.gnn2_pool = GNN(3 * 64, 64, num_nodes, batch_normalization = batch_normalization)
        self.gnn2_embed = GNN(3 * 64, 64, 64, batch_normalization = batch_normalization, lin=False)

        self.gnn3_embed = GNN(3 * 64, 64, 64, batch_normalization = batch_normalization, lin=False)

        self.lin1 = torch.nn.Linear(3 * 64, 64)
        self.lin2 = torch.nn.Linear(64, 1)
        

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


device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')
model = Net().to(device)
if args.load_path is not None and os.path.isfile(args.load_path):
    model.load_state_dict(torch.load(args.load_path))
    print_info('Successfully loaded the GNN model from {}'.format(args.load_path))
else:
    print_info('Train with random initial GNN model')

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


def train(epoch):
    model.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output, loss_link, loss_entropy = model(data.x, data.adj, data.mask)        
        loss = F.mse_loss(output[:, 0], data.y.view(-1))
        # loss = loss + loss_link + loss_entropy
        loss.backward()
        loss_all += loss.item()
        optimizer.step()
    return loss_all / len(train_loader)


@torch.no_grad()
def test(loader, size, debug = False):
    model.eval()
    error = 0.
    idx = 0
    for data in loader:
        data = data.to(device)
        pred = model(data.x, data.adj, data.mask)[0]
        error += F.mse_loss(pred[:, 0], data.y.view(-1))
        if idx == 0 and debug:
            print_info('predict = {}'.format(pred[:, 0]))
            print_info('y       = {}'.format(data.y.view(-1)))
        idx += 1
    return error / size


if not args.test:
    os.makedirs(save_dir, exist_ok = True)

    best_val_error = test_error = 10000.0
    best_model = None
    for epoch in range(1, 151):
        t_start = time.time()
        train_loss = train(epoch)
        train_error = test(train_loader, len(train_loader))
        val_error = test(val_loader, len(val_loader))
        t_end = time.time()
        if val_error < best_val_error:
            test_error = test(test_loader, len(test_loader), debug = True)
            best_val_error = val_error
            best_model = deepcopy(model)

        print('Epoch: {:03d}, Epoch Time: {:.1f}s, Train Loss: {:.7f}, Train Error: {:.7f}, Val Error: {:.7f}, Test Error: {:.7f}'\
            .format(epoch, t_end - t_start, train_loss, train_error, val_error, test_error))
        
        # save model with fixed interval
        if args.save_interval > 0 and epoch % args.save_interval == 0:
            save_path = os.path.join(save_dir, 'model_state_dict_{}.pt'.format(epoch))
            torch.save(model.state_dict(), save_path)

    # save the final model
    save_path = os.path.join(save_dir, 'model_state_dict_final.pt')
    torch.save(model.state_dict(), save_path)
else:
    train_error = test(train_loader, len(train_loader))
    val_error = test(val_loader, len(val_loader))
    test_error = test(test_loader, len(test_loader), debug = True)
    print('Train Error: {:.7f}, Val Error: {:.7f}, Test Error: {:.7f}'\
            .format(train_error, val_error, test_error))
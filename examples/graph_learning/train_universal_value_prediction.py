'''
train_universal_value_prediction.py

use Graph Neural Network to learn a univeral value function for all robot designs.

Learned GNN: 
input: a graph of the robot design including partial designs and terminal designs
output: the predicted reward for the design (maximal reward in the searching subtree).

Arguments:
--dataset-name: the name of the dataset
--use-cuse: if use gpu to train, [default is False]
--load-path: the path to load a pre-trained model
--test: if only test the loaded model
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
import torch_geometric.transforms as T
from torch_geometric.data import DenseDataLoader
from torch_geometric.data.data import Data

import IPython 

# import our packages
import parse_log_file
from common import *
from Net import Net
from load_log_file import *

def train(model, train_loader):
    global optimizer, device

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
def test(model, loader, debug = False):
    global device

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
            idx = 1
    return error / len(loader)

if __name__ == '__main__':
    # parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type = str, default = 'flat_mar23')
    parser.add_argument('--use-cuda', default = False, action = 'store_true')
    parser.add_argument('--save-interval', type = int, default = 10)
    parser.add_argument('--load-path', type = str, default = None)
    parser.add_argument('--test', default = False, action = 'store_true')
    parser.add_argument('--max-nodes', default = 50, type = int)

    args = parser.parse_args()
    dataset_name = args.dataset_name

    # dataset path 
    dataset_dir = os.path.join(current_dir, 'data', 'universal_design_dataset', dataset_name)
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
    else:
        os.makedirs(dataset_dir, exist_ok = True)

        raw_dataset_path = os.path.join(current_dir, 'data', args.dataset_name + '.csv')

        all_features, all_link_adj, all_masks, all_rewards \
            = load_partial_design_data(raw_dataset_path, os.path.join(project_dir, 'data/designs/grammar_jan21.dot'))

        # Create dataset object
        data = [Data(adj=torch.from_numpy(adj).float(),
                    mask=torch.from_numpy(mask),
                    x=torch.from_numpy(x).float(), 
                    y=torch.from_numpy(np.array([y])).float() ) for adj, mask, x, y in zip(all_link_adj, all_masks, all_features, all_rewards)]
        random.shuffle(data)

        n_val = (len(data) + 9) // 10
        n_test = (len(data) + 9) // 10
        train_dataset = data[:-n_test - n_val]
        val_dataset = data[-n_test - n_val:-n_test]
        test_dataset = data[-n_test:]
        
        with open(testset_path, 'wb') as test_file, open(valset_path, 'wb') as val_file, open(trainset_path, 'wb') as train_file: 
            pickle.dump(test_dataset, test_file)
            pickle.dump(train_dataset, train_file)
            pickle.dump(val_dataset, val_file)

    # max_nodes = test_dataset[0].x.shape[0] # suppose to be 19
    max_nodes = args.max_nodes
    num_features = test_dataset[0].x.shape[1] # suppose to be 39

    print_info(f'Dataset: train: size {len(train_dataset)}, test: size {len(test_dataset)}, val: size {len(val_dataset)}')
    print_info(f'max_nodes = {max_nodes}, num_features = {num_features}')

    # constrct batch
    test_loader = DenseDataLoader(test_dataset, batch_size=20)
    val_loader = DenseDataLoader(val_dataset, batch_size=20)
    train_loader = DenseDataLoader(train_dataset, batch_size=20)

    global device
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')
    model = Net(max_nodes = max_nodes, num_channels = num_features, num_outputs = 1).to(device)
    if args.load_path is not None and os.path.isfile(args.load_path):
        model.load_state_dict(torch.load(args.load_path))
        print_info('Successfully loaded the GNN model from {}'.format(args.load_path))
    else:
        print_info('Train with random initial GNN model')

    if not args.test:
        save_dir = os.path.join(current_dir, 'trained_models', 'universal_value_function', dataset_name, get_time_stamp())

        os.makedirs(save_dir, exist_ok = True)

        global optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        best_val_error = test_error = 10000.0
        best_model = None
        for epoch in range(1, 101):
            t_start = time.time()
            train_loss = train(model, train_loader)
            train_error = test(model, train_loader)
            val_error = test(model, val_loader)
            test_error = test(model, test_loader)
            t_end = time.time()
            if val_error < best_val_error:
                best_val_error = val_error
                best_model = deepcopy(model)
                print('Best: Epoch: {:03d}, Epoch Time: {:.1f}s, Train Loss: {:.7f}, Train Error: {:.7f}, Val Error: {:.7f}, Test Error: {:.7f}'\
                    .format(epoch, t_end - t_start, train_loss, train_error, val_error, test_error))
            else:
                print('Epoch: {:03d}, Epoch Time: {:.1f}s, Train Loss: {:.7f}, Train Error: {:.7f}, Val Error: {:.7f}, Test Error: {:.7f}'\
                    .format(epoch, t_end - t_start, train_loss, train_error, val_error, test_error))
            
            # save model with fixed interval
            if args.save_interval > 0 and epoch % args.save_interval == 0:
                save_path = os.path.join(save_dir, 'model_state_dict_{}.pt'.format(epoch))
                torch.save(model.state_dict(), save_path)

        # save the final model
        save_path = os.path.join(save_dir, 'model_state_dict_final.pt')
        torch.save(model.state_dict(), save_path)

        # save the best model
        save_path = os.path.join(save_dir, 'model_state_dict_best.pt')
        torch.save(best_model.state_dict(), save_path)
    else:
        train_error = test(model, train_loader)
        val_error = test(model, val_loader)
        test_error = test(model, test_loader)
        print('Train Error: {:.7f}, Val Error: {:.7f}, Test Error: {:.7f}'\
                .format(train_error, val_error, test_error))
'''
heuristic_search_algo_2.py

Implement the second algorithm in the pdf: overleaf.com/project/5ea0b6ae56ec33000137a41c

Learn a heuristic function V(s) for each universal design to predict the best reward under that node in the searching tree.

1. Search strategy is epsilon greedy.
2. Train V to fit bellman update equation.
'''

import sys
import os
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'graph_learning'))

# import python packages
import argparse
import time
import random
import pickle
import csv
import numpy as np
from copy import deepcopy
import torch
from torch import optim
import torch.nn.functional as F
from torch_geometric.data.data import Data

# import our own packages
from arguments import get_parser
from utils import solve_argv_conflict
from common import *
from design_search import make_initial_graph, build_normalized_robot, get_applicable_matches, has_nonterminals
import tasks
import pyrobotdesign as rd
from RobotGrammarEnv import RobotGrammarEnv
from Net import Net
from Preprocessor import Preprocessor

# predict without grad
def predict(V, state):
    global preprocessor
    adj_matrix_np, features_np, masks_np = preprocessor.preprocess(state)
    with torch.no_grad():
        features = torch.tensor(features_np).unsqueeze(0)
        adj_matrix = torch.tensor(adj_matrix_np).unsqueeze(0)
        masks = torch.tensor(masks_np).unsqueeze(0)
        output, _, _ = V(features, adj_matrix, masks)
        return output.item()

def predict_batch(V, states):
    global preprocessor
    adj_matrix_np, features_np, masks_np = [], [], []
    for state in states:
        adj_matrix, features, masks = preprocessor.preprocess(state)
        adj_matrix_np.append(adj_matrix)
        features_np.append(features)
        masks_np.append(masks)
    with torch.no_grad():
        adj_matrix = torch.tensor(adj_matrix_np)
        features = torch.tensor(features_np)
        masks = torch.tensor(masks_np)
        output, _, _ = V(features, adj_matrix, masks)
    return output[:, 0].detach().numpy()

def select_action(env, V, state, eps):
    available_actions = env.get_available_actions(state)
    if len(available_actions) == 0:
        return None
    sample = random.random()
    if sample > eps:
        next_states = []
        for action in available_actions:
            next_states.append(env.transite(state, action))
        values = predict_batch(V, next_states)
        best_action = available_actions[np.argmax(values)]
    else:
        best_action = available_actions[random.randrange(len(available_actions))]
    
    return best_action

def compute_Vhat(robot_graph, env, V):
    if has_nonterminals(robot_graph):
        available_actions = env.get_available_actions(robot_graph)
        if len(available_actions) == 0:
            return -5.0
        next_states = []
        for action in available_actions:
            next_states.append(env.transite(robot_graph, action))
        values = predict_batch(V, next_states)
        return np.max(values)
    else:
        return env.get_reward(robot_graph)[1]

def search_algo_2(args):
    # iniailize random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # initialize/load
    # TODO: use 80 to fit the input of trained MPC GNN, use args.depth * 3 later for real mpc
    max_nodes = 80
    task_class = getattr(tasks, args.task)
    task = task_class()
    graphs = rd.load_graphs(args.grammar_file)
    rules = [rd.create_rule_from_graph(g) for g in graphs]

    # state preprocessor
    # Find all possible link labels, so they can be one-hot encoded
    all_labels = set()
    for rule in rules:
        for node in rule.lhs.nodes:
            all_labels.add(node.attrs.require_label)
    all_labels = sorted(list(all_labels))
    global preprocessor
    preprocessor = Preprocessor(max_nodes = max_nodes, all_labels = all_labels)

    # initialize the env
    env = RobotGrammarEnv(task, rules, enable_reward_oracle = True, preprocessor = preprocessor)

    # initialize Value function
    device = 'cpu'
    state = env.reset()
    sample_adj_matrix, sample_features, sample_masks = preprocessor.preprocess(state)
    num_features = sample_features.shape[1]
    V = Net(max_nodes = max_nodes, num_channels = num_features, num_outputs = 1).to(device)

    # load pretrained V function
    if args.load_V_path is not None:
        V.load_state_dict(torch.load(args.load_V_path))
        print_info('Loaded pretrained V function from {}'.format(args.load_V_path))

    if not args.test:
        # initialize save folders and files
        fp_log = open(os.path.join(args.save_dir, 'log.txt'), 'w')
        fp_log.close()
        design_csv_path = os.path.join(args.save_dir, 'designs.csv')
        fp_csv = open(design_csv_path, 'w')
        fieldnames = ['rule_seq', 'reward']
        writer = csv.DictWriter(fp_csv, fieldnames=fieldnames)
        writer.writeheader()
        fp_csv.close()

        # initialize the optimizer
        global optimizer
        optimizer = torch.optim.Adam(V.parameters(), lr = args.lr)

        # initialize best design
        best_design, best_reward = None, -np.inf
        
        # initialize the seen states pool
        states_pool = []
        
        # initialize visited states
        state_set = set()

        # TODO: load previously explored designs
        
        # explored designs
        designs = []
        design_rewards = []
        
        # reward history
        epoch_rew_his = []

        for epoch in range(args.num_iterations):
            t_start = time.time()

            V.eval()

            t0 = time.time()

            # use e-greedy to sample a design within maximum #steps.
            if args.eps_schedule == 'linear-decay':
                # linear schedule
                eps = args.eps_start + epoch / args.num_iterations * (args.eps_end - args.eps_start)
            elif args.eps_schedule == 'exp-decay':
                # exp schedule
                eps = args.eps_end + (args.eps_start - args.eps_end) * np.exp(-1.0 * epoch / args.num_iterations / args.eps_decay)

            done = False
            while not done:
                state = env.reset()
                rule_seq = []
                state_seq = [state]
                total_reward = 0.
                for _ in range(args.depth):
                    action = select_action(env, V, state, eps)
                    if action is None:
                        break
                    rule_seq.append(action)
                    next_state, reward, done = env.step(action)
                    total_reward += reward
                    state_seq.append(next_state)
                    state = next_state
                    if done:
                        break
            
            # save the design and the reward in the list
            designs.append(rule_seq)
            design_rewards.append(total_reward)

            # update best design
            if total_reward > best_reward:
                best_design, best_reward = rule_seq, total_reward
            
            # update state pool
            for ancestor in state_seq:
                state_hash_key = hash(ancestor)
                if not (state_hash_key in state_set):
                    state_set.add(state_hash_key)
                    states_pool.append(ancestor)

            t1 = time.time()

            # optimize
            V.train()
            total_loss = 0.0
            for _ in range(args.depth):
                minibatch = random.sample(states_pool, min(len(states_pool), args.batch_size))

                train_adj_matrix, train_features, train_masks, train_reward = [], [], [], []
                for robot_graph in minibatch:
                    V_hat = compute_Vhat(robot_graph, env, V)
                    adj_matrix, features, masks = preprocessor.preprocess(robot_graph)
                    train_adj_matrix.append(adj_matrix)
                    train_features.append(features)
                    train_masks.append(masks)
                    train_reward.append(V_hat)
                
                train_adj_matrix_torch = torch.tensor(train_adj_matrix)
                train_features_torch = torch.tensor(train_features)
                train_masks_torch = torch.tensor(train_masks)
                train_reward_torch = torch.tensor(train_reward)
                
                optimizer.zero_grad()
                output, loss_link, loss_entropy = V(train_features_torch, train_adj_matrix_torch, train_masks_torch)
                loss = F.mse_loss(output[:, 0], train_reward_torch)
                loss.backward()
                total_loss += loss.item()
                optimizer.step()

            t2 = time.time()

            # logging
            if (epoch + 1) % args.log_interval == 0 or epoch + 1 == args.num_iterations:
                iter_save_dir = os.path.join(args.save_dir, '{}'.format(epoch + 1))
                os.makedirs(os.path.join(iter_save_dir), exist_ok = True)
                # save model
                save_path = os.path.join(iter_save_dir, 'V_model.pt')
                torch.save(V.state_dict(), save_path)
                # save explored designs and their rewards
                fp_csv = open(design_csv_path, 'a')
                fieldnames = ['rule_seq', 'reward']
                writer = csv.DictWriter(fp_csv, fieldnames=fieldnames)
                for i in range(epoch - args.log_interval + 1, epoch + 1):
                    writer.writerow({'rule_seq': str(designs[i]), 'reward': design_rewards[i]})
                fp_csv.close()

            epoch_rew_his.append(total_reward)

            t_end = time.time()
            avg_loss = total_loss / args.depth
            len_his = min(len(epoch_rew_his), 30)
            avg_reward = np.sum(epoch_rew_his[-len_his:]) / len_his
            print('Epoch {}: Time = {:.2f}, T_sample = {:.2f}, T_opt = {:.2f}, eps = {:.3f}, training loss = {:.4f}, reward = {:.4f}, last 30 epoch reward = {:.4f}, best reward = {:.4f}'.format(epoch, t_end - t_start, t1 - t0, t2 - t1, eps, avg_loss, total_reward, avg_reward, best_reward))
            fp_log = open(os.path.join(args.save_dir, 'log.txt'), 'a')
            fp_log.write('eps = {:.4f}, loss = {:.4f}, reward = {:.4f}, avg_reward = {:.4f}\n'.format(eps, avg_loss, total_reward, avg_reward))
            fp_log.close()

        save_path = os.path.join(args.save_dir, 'model_state_dict_final.pt')
        torch.save(V.state_dict(), save_path)
    else:
        import IPython
        IPython.embed()

        # test
        V.eval()
        print('Start testing')
        test_epoch = 30
        y0 = []
        y1 = []
        x = []
        for ii in range(10):
            eps = 1.0 - 0.1 * ii

            print('------------------------------------------')
            print('eps = ', eps)

            reward_sum = 0.
            best_reward = -np.inf
            for epoch in range(test_epoch):
                t0 = time.time()

                # use e-greedy to sample a design within maximum #steps.
                done = False
                while not done:
                    state = env.reset() 
                    rule_seq = []
                    state_seq = [state]
                    total_reward = 0.
                    for _ in range(args.depth):
                        action = select_action(env, V, state, eps)
                        if action is None:
                            break
                        rule_seq.append(action)
                        next_state, reward, done = env.step(action)
                        total_reward += reward
                        state_seq.append(next_state)
                        state = next_state
                        if done:
                            break

                reward_sum += total_reward
                best_reward = max(best_reward, total_reward)
                print(f'design {epoch}: reward = {total_reward}, time = {time.time() - t0}')

            print('test avg reward = ', reward_sum / test_epoch)
            print('best reward found = ', best_reward)
            x.append(eps)
            y0.append(reward_sum / test_epoch)
            y1.append(best_reward)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize = (10, 5))
        ax[0].plot(x, y0)
        ax[1].plot(x, y1)
        plt.show()

if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    args_list = ['--task', 'FlatTerrainTask',
                 '--grammar-file', '../../data/designs/grammar_jan21.dot',
                 '--num-iterations', '10000',
                 '--mpc-num-processes', '8',
                 '--lr', '1e-4',
                 '--eps-start', '1.0',
                 '--eps-end', '0.2',
                 '--eps-decay', '0.3',
                 '--batch-size', '32',
                 '--depth', '25',
                 '--save-dir', './trained_models/FlatTerrainTask/algo2/',
                 '--render-interval', '80',
                 '--log-interval', '100']
                #  '--load-V-path', './trained_models/universal_value_function/test_model.pt']
    
    solve_argv_conflict(args_list)
    parser = get_parser()
    args = parser.parse_args(args_list + sys.argv[1:])

    if not args.test:
        args.save_dir = os.path.join(args.save_dir, get_time_stamp())
        try:
            os.makedirs(args.save_dir, exist_ok = True)
        except OSError:
            pass
        
        fp = open(os.path.join(args.save_dir, 'args.txt'), 'w')
        fp.write(str(args_list + sys.argv[1:]))
        fp.close()

    search_algo_2(args)

'''
heuristic_search_algo_1.py

Implement the first algorithm in the pdf: overleaf.com/project/5ea0b6ae56ec33000137a41c

Learn a heuristic function V(s) for each universal design to predict the best reward under that node in the searching tree.

1. Search strategy is epsilon greedy.
2. Update the target value for the ancestors of the explored design
3. train V to fit target value
'''

import sys
import os
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'graph_learning'))
sys.path.append(os.path.join(base_dir, 'design_search'))

# import python packages
import argparse
import time
import random
import pickle
import csv
import ast
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
from states_pool import StatesPool

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

def select_action_serial(env, V, state, eps):
    available_actions = env.get_available_actions(state)
    if len(available_actions) == 0:
        return None
    sample = random.random()
    if sample > eps:
        best_action, best_V = -1, -np.inf
        for action in available_actions:
            next_state = env.transite(state, action)
            next_V = predict(V, next_state)
            if next_V > best_V:
                best_action, best_V = action, next_V
    else:
        best_action = available_actions[random.randrange(len(available_actions))]
    
    return best_action

def select_action(env, V, state, eps):
    available_actions = env.get_available_actions(state)
    if len(available_actions) == 0:
        return None, None
    sample = random.random()
    step_type = ""
    if sample > eps:
        next_states = []
        for action in available_actions:
            next_states.append(env.transite(state, action))
        values = predict_batch(V, next_states)
        best_action = available_actions[np.argmax(values)]
        step_type = 'optimal'
    else:
        best_action = available_actions[random.randrange(len(available_actions))]
        step_type = 'random'
    
    return best_action, step_type

def update_Vhat(V_hat, state_seq, reward):
    for state in state_seq:
        state_hash_key = hash(state)
        if not (state_hash_key in V_hat): # TODO: check the complexity of this line
            V_hat[state_hash_key] = -np.inf
        V_hat[state_hash_key] = max(V_hat[state_hash_key], reward)

def update_states_pool(states_pool, state_seq):
    for state in state_seq[2:]: # the first two state in the sequence are always same, so omitted
        # state_hash_key = hash(state)
        # if not (state_hash_key in V_hat):
        #     states_pool.append(state)
        states_pool.push(state) # try to use state distribution

def apply_rules(state, actions, env):
    cur_state = state
    state_seq = [cur_state]
    for action in actions:
        cur_state = env.transite(cur_state, action)
        state_seq.append(cur_state)
    return cur_state, state_seq

def show_all_actions(env, V, V_hat, state):
    available_actions = env.get_available_actions(state)
    print('V(state) = ', predict(V, state), ', V_hat(state) = ', V_hat[hash(state)])
    for action in available_actions:
        next_state = env.transite(state, action)
        print('action = ', action, ', V = ', predict(V, next_state), ', V_hat = ', V_hat[hash(next_state)] if hash(next_state) in V_hat else -1.0)

def search_algo(args):
    # iniailize random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    
    # initialize/load
    # TODO: use 80 to fit the input of trained MPC GNN, use args.depth * 3 later for real mpc
    max_nodes = 80
    task_class = getattr(tasks, args.task)
    task = task_class()
    graphs = rd.load_graphs(args.grammar_file)
    rules = [rd.create_rule_from_graph(g) for g in graphs]

    # initialize preprocessor
    # Find all possible link labels, so they can be one-hot encoded
    all_labels = set()
    for rule in rules:
        for node in rule.lhs.nodes:
            all_labels.add(node.attrs.require_label)
    all_labels = sorted(list(all_labels))
    
    global preprocessor
    preprocessor = Preprocessor(max_nodes = max_nodes, all_labels = all_labels)

    # initialize the env
    env = RobotGrammarEnv(task, rules, seed = args.seed, mpc_num_processes = args.mpc_num_processes)

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
    
    # initialize target V_hat look up table
    V_hat = dict()

    # load pretrained V_hat
    if args.load_Vhat_path is not None:
        V_hat_fp = open(args.load_Vhat_path, 'rb')
        V_hat = pickle.load(V_hat_fp)
        V_hat_fp.close()
        print_info('Loaded pretrained Vhat from {}'.format(args.load_Vhat_path))

    # initialize the seen states pool
    states_pool = StatesPool(capacity = args.states_pool_capacity)

    # explored designs
    designs = []
    design_rewards = []

    if not args.test:
        # initialize save folders and files
        fp_log = open(os.path.join(args.save_dir, 'log.txt'), 'w')
        fp_log.close()
        fp_eval = open(os.path.join(args.save_dir, 'eval.txt'), 'w')
        fp_eval.close()
        design_csv_path = os.path.join(args.save_dir, 'designs.csv')
        fp_csv = open(design_csv_path, 'w')
        fieldnames = ['rule_seq', 'reward']
        writer = csv.DictWriter(fp_csv, fieldnames=fieldnames)
        writer.writeheader()
        fp_csv.close()

        # initialize the optimizer
        global optimizer
        optimizer = torch.optim.Adam(V.parameters(), lr = args.lr)

        # initialize best design rule sequence
        best_design, best_reward = None, -np.inf

        # reward history
        epoch_rew_his = []
        last_checkpoint = -1

        # recording time
        t_sample_sum = 0.

        # record the count for invalid samples
        no_action_samples, step_exceeded_samples = 0, 0

        for epoch in range(args.num_iterations):
            t_start = time.time()

            V.eval()

            # update eps and eps_sample
            if args.eps_schedule == 'linear-decay':
                eps = args.eps_start + epoch / args.num_iterations * (args.eps_end - args.eps_start)
            elif args.eps_schedule == 'exp-decay':
                eps = args.eps_end + (args.eps_start - args.eps_end) * np.exp(-1.0 * epoch / args.num_iterations / args.eps_decay)

            if args.eps_sample_schedule == 'linear-decay':
                eps_sample = args.eps_sample_start + epoch / args.num_iterations * (args.eps_sample_end - args.eps_sample_start)
            elif args.eps_sample_schedule == 'exp-decay':
                eps_sample = args.eps_sample_end + (args.eps_sample_start - args.eps_sample_end) * np.exp(-1.0 * epoch / args.num_iterations / args.eps_sample_decay)

            t_sample, t_update, t_mpc, t_opt = 0, 0, 0, 0

            selected_design, selected_reward = None, -np.inf
            selected_state_seq, selected_rule_seq = None, None

            p = random.random()
            if p < eps_sample:
                num_samples = 1
            else:
                num_samples = args.num_samples
        
            # use e-greedy to sample a design within maximum #steps.
            for _ in range(num_samples):
                valid = False
                while not valid:
                    t0 = time.time()

                    state = env.reset()
                    rule_seq = []
                    state_seq = [state]
                    no_action_flag = False
                    for _ in range(args.depth):
                        action, step_type = select_action(env, V, state, eps)
                        if action is None:
                            no_action_flag = True
                            break
                        rule_seq.append(action)
                        next_state = env.transite(state, action)
                        state_seq.append(next_state)
                        state = next_state
                    
                    valid = env.is_valid(state)

                    t_sample += time.time() - t0

                    t0 = time.time()

                    if not valid:
                        # update the invalid sample's count
                        if no_action_flag:
                            no_action_samples += 1
                        else:
                            step_exceeded_samples += 1
                        # update the Vhat for invalid designs
                        update_Vhat(V_hat, state_seq, -2.0)
                        # update states pool
                        update_states_pool(states_pool, state_seq)
                    else:
                        if (hash(state) in V_hat) and (V_hat[hash(state)] > -2.0 + 1e-3):
                            update_Vhat(V_hat, state_seq, -2.0)
                            update_states_pool(states_pool, state_seq)
                            valid = False

                    t_update += time.time() - t0

                predicted_value = predict(V, state)
                if predicted_value > selected_reward:
                    selected_design, selected_reward = state, predicted_value
                    selected_rule_seq, selected_state_seq = rule_seq, state_seq

            t0 = time.time()

            repeated = False
            if (hash(selected_design) in V_hat) and (V_hat[hash(selected_design)] > -2.0 + 1e-3):
                repeated = True

            ctrl_seq, reward = env.get_reward(selected_design)

            t_mpc += time.time() - t0

            # save the design and the reward in the list
            designs.append(selected_rule_seq)
            design_rewards.append(reward)

            # update best design
            if reward > best_reward:
                best_design, best_reward = selected_rule_seq, reward
                print_info('new best: reward = {:.4f}, predicted reward = {:.4f}, num_samples = {}'.format(reward, selected_reward, num_samples))

            t0 = time.time()

            # update V_hat for the valid design
            update_Vhat(V_hat, selected_state_seq, reward)

            # update states pool for the valid design
            update_states_pool(states_pool, selected_state_seq)

            t_update += time.time() - t0

            t0 = time.time()

            # optimize
            V.train()
            total_loss = 0.0
            for _ in range(args.opt_iter):
                minibatch = states_pool.sample(min(len(states_pool), args.batch_size))
                
                train_adj_matrix, train_features, train_masks, train_reward = [], [], [], []
                for robot_graph in minibatch:
                    hash_key = hash(robot_graph)
                    target_reward = V_hat[hash_key]
                    adj_matrix, features, masks = preprocessor.preprocess(robot_graph)
                    train_adj_matrix.append(adj_matrix)
                    train_features.append(features)
                    train_masks.append(masks)
                    train_reward.append(target_reward)
                
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

            t_opt += time.time() - t0

            t_end = time.time()

            t_sample_sum += t_sample

            # logging
            if (epoch + 1) % args.log_interval == 0 or epoch + 1 == args.num_iterations:
                iter_save_dir = os.path.join(args.save_dir, '{}'.format(epoch + 1))
                os.makedirs(os.path.join(iter_save_dir), exist_ok = True)
                # save model
                save_path = os.path.join(iter_save_dir, 'V_model.pt')
                torch.save(V.state_dict(), save_path)
                # save V_hat
                save_path = os.path.join(iter_save_dir, 'V_hat')
                fp = open(save_path, 'wb')
                pickle.dump(V_hat, fp)
                fp.close()

            # save explored design and its reward
            fp_csv = open(design_csv_path, 'a')
            fieldnames = ['rule_seq', 'reward']
            writer = csv.DictWriter(fp_csv, fieldnames=fieldnames)
            for i in range(last_checkpoint + 1, len(designs)):
                writer.writerow({'rule_seq': str(designs[i]), 'reward': design_rewards[i]})
            last_checkpoint = len(designs) - 1
            fp_csv.close()

            epoch_rew_his.append(reward)

            avg_loss = total_loss / args.opt_iter
            len_his = min(len(epoch_rew_his), 30)
            avg_reward = np.sum(epoch_rew_his[-len_his:]) / len_his
            if repeated:
                print('Epoch {}: T_sample = {:.2f}, T_update = {:.2f}, T_mpc = {:.2f}, T_opt = {:.2f}, eps = {:.3f}, eps_sample = {:.3f}, #samples = {}, training loss = {:.4f}, predicted_reward = {:.4f}, reward = {:.4f}, last 30 epoch reward = {:.4f}, best reward = {:.4f}'.format(\
                    epoch, t_sample, t_update, t_mpc, t_opt, eps, eps_sample, num_samples, \
                    avg_loss, selected_reward, reward, avg_reward, best_reward))
            else:
                print_warning('Epoch {}: T_sample = {:.2f}, T_update = {:.2f}, T_mpc = {:.2f}, T_opt = {:.2f}, eps = {:.3f}, eps_sample = {:.3f}, #samples = {}, training loss = {:.4f}, predicted_reward = {:.4f}, reward = {:.4f}, last 30 epoch reward = {:.4f}, best reward = {:.4f}'.format(\
                    epoch, t_sample, t_update, t_mpc, t_opt, eps, eps_sample, num_samples, \
                    avg_loss, selected_reward, reward, avg_reward, best_reward))

            fp_log = open(os.path.join(args.save_dir, 'log.txt'), 'a')
            fp_log.write('eps = {:.4f}, eps_sample = {:.4f}, num_samples = {}, T_sample = {:4f}, T_update = {:4f}, T_mpc = {:.4f}, T_opt = {:.4f}, loss = {:.4f}, predicted_reward = {:.4f}, reward = {:.4f}, avg_reward = {:.4f}\n'.format(\
                eps, eps_sample, num_samples, t_sample, t_update, t_mpc, t_opt, avg_loss, selected_reward, reward, avg_reward))
            fp_log.close()

            if (epoch + 1) % args.log_interval == 0:
                print_info('Avg sampling time for last {} epoch: {:.4f} second'.format(args.log_interval, t_sample_sum / args.log_interval))
                t_sample_sum = 0.
                invalid_cnt, valid_cnt = 0, 0
                for state in states_pool.pool:
                    if np.isclose(V_hat[hash(state)], -2.0):
                        invalid_cnt += 1
                    else:
                        valid_cnt += 1
                print_info('states_pool size = {}, #valid = {}, #invalid = {}, #valid / #invalid = {}'.format(len(states_pool), valid_cnt, invalid_cnt, valid_cnt / invalid_cnt if invalid_cnt > 0 else 10000.0))
                print_info('Invalid samples: #no_action_samples = {}, #step_exceeded_samples = {}, #no_action / #step_exceeded = {}'.format(no_action_samples, step_exceeded_samples, no_action_samples / step_exceeded_samples))

            # # evaluation
            # if (args.eval_interval > 0) and ((epoch + 1) % args.eval_interval == 0 or epoch + 1 == args.num_iterations):
            #     print_info('-------- Doing evaluation --------')
            #     print_info('#states = {}'.format(len(states_pool)))
            #     loss_total = 0.
            #     for state in states_pool.pool:
            #         value = predict(V, state)
            #         loss_total += (V_hat[hash(state)] - value) ** 2
            #     print_info('Loss = {:.3f}'.format(loss_total / len(states_pool)))
            #     fp_eval = open(os.path.join(args.save_dir, 'eval.txt'), 'a')
            #     fp_eval.write('epoch = {}, loss = {:.3f}\n'.format(epoch + 1, loss_total / len(states_pool)))
            #     fp_eval.close()

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
        for ii in range(0, 11):
            eps = 1.0 - 0.1 * ii

            print('------------------------------------------')
            print('eps = ', eps)

            reward_sum = 0.
            best_reward = -np.inf
            for epoch in range(test_epoch):
                t0 = time.time()

                # use e-greedy to sample a design within maximum #steps.
                vaild = False
                while not valid:
                    state = env.reset()
                    rule_seq = []
                    state_seq = [state]
                    for _ in range(args.depth):
                        action, step_type = select_action(env, V, state, eps)
                        if action is None:
                            break
                        rule_seq.append(action)
                        next_state = env.transite(state, action)
                        state_seq.append(next_state)
                        if not has_nonterminals(next_state):
                            valid = True
                            break
                        state = next_state
                        
                _, reward = env.get_reward(state)
                reward_sum += reward
                best_reward = max(best_reward, reward)
                print(f'design {epoch}: reward = {reward}, time = {time.time() - t0}')

            print('test avg reward = ', reward_sum / test_epoch)
            print('best reward found = ', best_reward)
            x.append(eps)
            y0.append(reward_sum / test_epoch)
            y1.append(best_reward)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize = (10, 5))
        ax[0].plot(x, y0)
        ax[0].set_title('Avg Reward')
        ax[0].set_xlabel('eps')
        ax[0].set_ylabel('reward')

        ax[1].plot(x, y1)
        ax[0].set_title('Best Reward')
        ax[0].set_xlabel('eps')
        ax[0].set_ylabel('reward')
        
        plt.show()

if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    args_list = ['--task', 'FlatTerrainTask',
                 '--grammar-file', '../../data/designs/grammar_apr30.dot',
                 '--num-iterations', '10000',
                 '--mpc-num-processes', '4',
                 '--lr', '1e-4',
                 '--eps-start', '1.0',
                 '--eps-end', '0.2',
                 '--eps-decay', '0.3',
                 '--eps-schedule', 'exp-decay',
                 '--eps-sample-start', '1.0',
                 '--eps-sample-end', '0.2',
                 '--eps-sample-decay', '0.3',
                 '--eps-sample-schedule', 'exp-decay',
                 '--num-samples', '1', 
                 '--opt-iter', '25', 
                 '--batch-size', '32',
                 '--states-pool-capacity', '50000',
                 '--depth', '25',
                 '--save-dir', './trained_models/FlatTerrainTask/mpc/',
                 '--render-interval', '80',
                 '--log-interval', '200',
                 '--eval-interval', '1000']
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

    search_algo(args)

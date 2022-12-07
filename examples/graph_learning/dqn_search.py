import sys
import os

base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'graph_learning'))

import argparse
from collections import deque, namedtuple
import random
import torch
from torch import optim
import torch.nn.functional as F
import numpy as np
import time

from RobotGrammarEnv import RobotGrammarEnv
from Net import Net
from arguments import get_parser
import tasks
import pyrobotdesign as rd
from utils import solve_argv_conflict
from common import *
from design_search import make_initial_graph, build_normalized_robot, get_applicable_matches, has_nonterminals
from Preprocessor import Preprocessor

from results import get_robot_image, make_robot_from_rule_sequence
import matplotlib.pyplot as plt

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):
    def __init__(self, capacity=1000000):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        self.memory.append(Transition(*args))
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, min(len(self.memory), batch_size))

    def __len__(self):
        return len(self.memory)


# predict without grad
def predict(Q, state):
    global preprocessor
    adj_matrix_np, features_np, masks_np = preprocessor.preprocess(state)
    with torch.no_grad():
        features = torch.tensor(features_np).unsqueeze(0)
        adj_matrix = torch.tensor(adj_matrix_np).unsqueeze(0)
        masks = torch.tensor(masks_np).unsqueeze(0)
        output, _, _ = Q(features, adj_matrix, masks)

        return output


def select_action_sergey(env, Q, state, eps):
    available_actions = env.get_available_actions(state)
    if len(available_actions) == 0:
        return None
    sample = random.random()
    if sample > eps:
        evals = predict(Q, state)
        best_action = available_actions[0]
        for action in available_actions:
            if evals[0][action] > evals[0][best_action]:  # squeeze
                best_action = action
    else:
        best_action = available_actions[random.randrange(len(available_actions))]

    return best_action

def select_action(env, Q, state, eps):
    available_actions = env.get_available_actions(state)
    if len(available_actions) == 0:
        return None
    sample = random.random()
    if sample > eps:
        evals = predict(Q, state)
        best_action = available_actions[0]
        for action in available_actions:
            if evals[0][action] > evals[0][best_action]:  # squeeze
                best_action = action
    else:
        best_action = available_actions[random.randrange(len(available_actions))]

    return best_action


def optimize(Q, Q_target, memory, batch_size, depth):
    # Mettici il codice di sergey

    minibatch = memory.sample(batch_size)

    minibatch_len = len(minibatch)

    features_batch = np.zeros((minibatch_len, depth*2, 41))
    adj_matrix_batch = np.zeros(
        (minibatch_len, depth*2, depth*2))  # preprocessor.preprocess(state) kicks out info useful for the dim
    masks_batch = np.zeros((minibatch_len, depth*2,))
    y_batch = np.zeros((minibatch_len, 20,))
    # = [], [], [], []

    for i, (state, action, next_state, reward, done) in enumerate(minibatch):
        y_target = predict(Q, state)
        if done:
            y_target[0][action] = reward
        else:
            y_next_state = predict(Q_target, next_state)
            y_target[0][action] = reward + np.max(y_next_state[0].numpy())

        adj_matrix_np, features_np, masks_np = preprocessor.preprocess(state)
        features_batch[i] = features_np
        adj_matrix_batch[i] = adj_matrix_np
        masks_batch[i] = masks_np
        y_batch[i] = y_target[0].numpy()

    features_batch = torch.tensor(features_batch)
    adj_matrix_batch = torch.tensor(adj_matrix_batch)
    masks_batch = torch.tensor(masks_batch)
    y_batch = torch.tensor(y_batch)

    global optimizer
    optimizer.zero_grad()
    output, loss_link, loss_entropy = Q(features_batch, adj_matrix_batch, masks_batch)
    loss = F.mse_loss(output, y_batch)
    loss.backward()
    optimizer.step()

    return loss.item()


def search(args):
    # initialize the env
    max_nodes = args.depth * 2
    task_class = getattr(tasks, args.task)
    task = task_class()
    graphs = rd.load_graphs(args.grammar_file)
    rules = [rd.create_rule_from_graph(g) for g in graphs]
    env = RobotGrammarEnv(task, rules, seed=args.seed, mpc_num_processes=args.mpc_num_processes)

    # state preprocessor
    # Find all possible link labels, so they can be one-hot encoded
    all_labels = set()
    for rule in rules:
        for node in rule.lhs.nodes:
            all_labels.add(node.attrs.require_label)
    all_labels = sorted(list(all_labels))
    global preprocessor
    preprocessor = Preprocessor(max_nodes=max_nodes, all_labels=all_labels)

    # initialize Q function
    device = 'cpu'
    state = env.reset()
    sample_adj_matrix, sample_features, sample_masks = preprocessor.preprocess(state)
    num_features = sample_features.shape[1]
    Q = Net(max_nodes=max_nodes, num_channels=num_features, num_outputs=len(rules)).to(device)
    Q_target = Net(max_nodes=max_nodes, num_channels=num_features, num_outputs=len(rules)).to(device)

    # initialize the optimizer
    global optimizer
    optimizer = torch.optim.Adam(Q.parameters(), lr=args.lr)

    # initialize DQN
    memory = ReplayMemory(capacity=1000000)
    scores = deque(maxlen=100)
    data = []

    best_rule_seq = []
    best_reward = 0

    for epoch in range(args.num_iterations):
        done = False
        eps = args.eps_start + epoch / args.num_iterations * (args.eps_end - args.eps_start)
        # eps = 1.0
        while not done:
            state = env.reset()
            total_reward = 0.
            rule_seq = []
            state_seq = []
            for i in range(args.depth):
                action = select_action(env, Q, state, eps)
                rule_seq.append(action)
                if action is None:
                    break
                next_state, reward, done = env.step(action)
                state_seq.append((state, action, next_state, reward, done))
                total_reward += reward
                state = next_state
                # print(rule_seq, total_reward, done)

                if done:
                    robot = make_robot_from_rule_sequence(rule_seq, rules)
                    v = get_robot_image(robot, task, render=False)
                    plt.imshow(v)
                    _str_ = str(rule_seq)[1:-1].replace(', ', '_') + '.png'  # title
                    plt.savefig('imgs/' + _str_, bbox_inches='tight')

                    if best_reward < total_reward:
                        best_reward = total_reward
                        best_rule_seq = rule_seq
                    break

        for i in range(len(state_seq)):
            memory.push(state_seq[i][0], state_seq[i][1], state_seq[i][2], state_seq[i][3], state_seq[i][4])
            data.append((state_seq[i][0], state_seq[i][1], total_reward))
        scores.append(total_reward)

        loss = 0.0
        for i in range(len(state_seq)):
            loss += optimize(Q, Q_target, memory, args.batch_size, args.depth)
        print('epoch ', epoch, ': reward = ', total_reward, ', eps = ', eps, ', Q loss = ', loss)
        print(best_reward, '@', best_rule_seq)
    # test
    cnt = 0
    for i in range(len(data)):
        if data[i][2] > 0.5:
            y_predict = predict(Q, data[i][0])
            print('target = ', data[i][2], ', predicted = ', y_predict[0][data[i][1]])
            cnt += 1
            if cnt == 5:
                break
    cnt = 0
    for i in range(len(data)):
        if data[i][2] < 0.5:
            y_predict = predict(Q, data[i][0])
            print('target = ', data[i][2], ', predicted = ', y_predict[0][data[i][1]])
            cnt += 1
            if cnt == 5:
                break


if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    args_list = ['--task', 'FlatTerrainTask',
                 # '--grammar-file', '../../data/designs/grammar_jan21.dot',
                 '--grammar-file', 'data/designs/grammar_apr30.dot',
                 '--num-iterations', '100',
                 '--mpc-num-processes', '8',
                 '--lr', '1e-3',
                 '--eps-start', '0.9',
                 '--eps-end', '0.5', # 0.05
                 '--batch-size', '32',
                 '--depth', '50',
                 '--save-dir', './trained_models/FlatTerrainTask/test/',
                 '--render-interval', '80']

    solve_argv_conflict(args_list)
    parser = get_parser()
    args = parser.parse_args(args_list + sys.argv[1:])

    # TODO: load cached mpc results
    # if args.log_file:

    args.save_dir = os.path.join(args.save_dir, get_time_stamp())
    try:
        os.makedirs(args.save_dir, exist_ok=True)
    except OSError:
        pass

    fp = open(os.path.join(args.save_dir, 'args.txt'), 'w')
    fp.write(str(args_list + sys.argv[1:]))
    fp.close()

    search(args)

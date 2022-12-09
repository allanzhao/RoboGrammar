import argparse
from collections import deque, namedtuple
import random
import torch
from torch import optim
# import torch.nn.functional as F
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
from tensorboardX import SummaryWriter
import datetime


class RobotRL:
    def __init__(self, rules, env_dummy, args_main):

        self.max_nodes = args_main.depth * 2
        self.rules = rules

        # state preprocessor
        # Find all possible link labels, so they can be one-hot encoded
        all_labels = set()
        for rule in self.rules:
            for node in rule.lhs.nodes:
                all_labels.add(node.attrs.require_label)
        self.all_labels = sorted(list(all_labels))

        self.preprocessor = Preprocessor(max_nodes=self.max_nodes, all_labels=self.all_labels)

        self.device = 'cpu'
        state = env_dummy.reset()
        _, sample_features, _ = self.preprocessor.preprocess(state)
        self.num_features = sample_features.shape[1]

        self.Q = Net(max_nodes=self.max_nodes,
                     num_channels=self.num_features,
                     num_outputs=len(self.rules),
                     layer_size=args_main.layer_size,
                     batch_normalization=args_main.batch_norm).to(self.device)

        self.Q_target = Net(max_nodes=self.max_nodes,
                            num_channels=self.num_features,
                            num_outputs=len(self.rules),
                            layer_size=args_main.layer_size,
                            batch_normalization=args_main.batch_norm).to(self.device)

        # initialize the optimizer
        self.optimizer = optim.Adam(self.Q.parameters(), lr=args_main.lr)

        self.hyperparams = {'num_iterations': args_main.num_iterations,
                            'eps_start': args_main.eps_start,
                            'eps_end': args_main.eps_end,
                            'eps_delta': args_main.eps_end - args_main.eps_start,
                            'lr': args_main.lr,
                            'depth': args_main.depth,
                            'batch_size': args_main.batch_size,
                            'freq_update': args_main.freq_update,
                            'batch_norm': args_main.batch_norm,
                            'layer_size': args_main.layer_size,
                            'store_cache': args_main.store_cache,
                            'seed': args_main.seed}

        self.summ_writer = SummaryWriter(f'runs/depth_{args_main.depth}_'
                                         f'num_{args_main.num_iterations}_'
                                         f'bs_{args_main.batch_size}_'
                                         f'lr_{args_main.lr}_'
                                         f'es_{args_main.eps_start}_'
                                         f'ee_{args_main.eps_end}_'
                                         f'frq_{args_main.freq_update}_'
                                         f'bn_{args_main.batch_norm}_'
                                         f'size_{args_main.layer_size}_'
                                         f's_{args_main.seed}_'
                                         f'{datetime.datetime.now():%Y%m%d_%H%M%S}', flush_secs=1, max_queue=1)

    def log_scalar(self, scalar, name, step_):
        self.summ_writer.add_scalar('{}'.format(name), scalar, step_)

    def select_action(self, env, state, eps):
        ''' e-greedy select action '''
        sample = random.random()
        if sample > eps:
            available_actions_mask, action_not_available = env.get_available_actions_mask(state)
            if action_not_available:
                return None
            else:

                evals, _, _ = self.Q(*self.preprocess_state(state))

                evals_masked = evals.squeeze().detach().numpy() + available_actions_mask  # -inf is not available
                return np.argmax(evals_masked)  # best action

        else:
            available_actions = env.get_available_actions(state)
            if len(available_actions) == 0:
                return None
            else:
                return np.random.choice(available_actions, size=1)[0]  # take one random action

    def preprocess_state(self, state):
        adj_matrix_np, features_np, masks_np = self.preprocessor.preprocess(state)
        features = torch.tensor(features_np).unsqueeze(0)
        adj_matrix = torch.tensor(adj_matrix_np).unsqueeze(0)
        masks = torch.tensor(masks_np).unsqueeze(0)
        return features, adj_matrix, masks


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


import pickle

class ReplayMemory(object):
    def __init__(self, capacity=1000000):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        self.memory.append(Transition(*args))
        self.position = (self.position + 1) % self.capacity
        # if len(self.memory) // 100 % 2 == 1: # every 200 from 100,300, ...
        #     file_to_store = open(f"replay_memory/size_{len(self.memory)}.pickle", "wb")
        #     pickle.dump(self.memory, file_to_store, pickle.HIGHEST_PROTOCOL)
        #     file_to_store.close()

    def sample(self, batch_size):
        return random.sample(self.memory, min(len(self.memory), batch_size))

    def __len__(self):
        return len(self.memory)

    # def store(self):


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


class RobotRL:
    def __init__(self, args_main):
        # initialize the env
        self.max_nodes = args_main.depth * 2
        task_class = getattr(tasks, args_main.task)
        self.task = task_class()
        graphs = rd.load_graphs(args_main.grammar_file)
        self.rules = [rd.create_rule_from_graph(g) for g in graphs]

        self.env = RobotGrammarEnv(self.task,
                                   self.rules,
                                   seed=args_main.seed,
                                   mpc_num_processes=args_main.mpc_num_processes)

        # state preprocessor
        # Find all possible link labels, so they can be one-hot encoded
        all_labels = set()
        for rule in self.rules:
            for node in rule.lhs.nodes:
                all_labels.add(node.attrs.require_label)
        self.all_labels = sorted(list(all_labels))

        self.preprocessor = Preprocessor(max_nodes=self.max_nodes, all_labels=self.all_labels)

        self.device = 'cpu'
        state = self.env.reset()
        _, sample_features, _ = self.preprocessor.preprocess(state)
        self.num_features = sample_features.shape[1]

        self.Q = Net(max_nodes=self.max_nodes,
                     num_channels=self.num_features,
                     num_outputs=len(self.rules)).to(self.device)

        self.Q_target = Net(max_nodes=self.max_nodes,
                            num_channels=self.num_features,
                            num_outputs=len(self.rules)).to(self.device)

        # initialize the optimizer
        self.optimizer = optim.Adam(self.Q.parameters(), lr=args_main.lr)

    # def select_action(self, state, eps):
    #     available_actions = self.env.get_available_actions(state)
    #     if len(available_actions) == 0:
    #         return None
    #     sample = random.random()
    #     if sample > eps:
    #         evals = self.predict_q_values_nograd(state)
    #         best_action = available_actions[0]
    #         for action in available_actions:
    #             if evals[0][action] > evals[0][best_action]:
    #                 best_action = action
    #     else:
    #         best_action = available_actions[random.randrange(len(available_actions))]
    #
    #     return best_action

    def select_action(self, state, eps):
        ''' e-greedy select action '''
        sample = random.random()
        if sample > eps:
            available_actions_mask, action_not_available = self.env.get_available_actions_mask(state)
            if action_not_available:
                return None
            else:

                adj_matrix, features, masks = self.preprocessor.preprocess(state)
                evals, _, _ = self.Q(torch.tensor(features).unsqueeze(0),
                                     torch.tensor(adj_matrix).unsqueeze(0),
                                     torch.tensor(masks).unsqueeze(0))

                evals_masked = evals.squeeze().detach().numpy() + available_actions_mask  # -inf is not available
                return np.argmax(evals_masked)  # best action

        else:
            available_actions = self.env.get_available_actions(state)
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

    # def store(self):


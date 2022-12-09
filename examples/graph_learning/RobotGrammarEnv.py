'''
RobotGrammarEnv.py

Implement the environment for robot grammar search problem.
'''
# import python packages
import os
import random
from copy import deepcopy
import numpy as np

# import our own packages
import pyrobotdesign as rd
from design_search import make_initial_graph, build_normalized_robot, get_applicable_matches, has_nonterminals, \
    simulate, presimulate
from common import *
from Net import Net
import torch

import pandas as pd
'''
class RobotGrammarEnv

Parameters:
    task: a task to be evaluated for the design
    rules: collection of grammar rules (actions)
    seed: random seed for the env
    mpc_num_processes: number of threads for mpc
    enable_reward_oracle: whether use the GNN oracle to compute the reward
    preprocessor: the preprocessor concerting a robot_graph into the GNN input, required if enable_reward_oracle is True
'''


class RobotGrammarEnv:
    def __init__(self, task, rules, seed=0, mpc_num_processes=8, enable_reward_oracle=False, preprocessor=None,
                 store_cache=True):

        self.task = task
        self.rules = rules
        self.seed = seed
        self.rng = random.Random(seed)
        self.mpc_num_processes = mpc_num_processes
        self.enable_reward_oracle = enable_reward_oracle
        if self.enable_reward_oracle:
            assert preprocessor is not None
            self.preprocessor = preprocessor
            self.load_reward_oracle()
        self.initial_state = make_initial_graph()

        self.result_cache = dict()

        files = os.listdir('data/')

        results = [file for file in files if 'result_cache_' in file]
        results.sort()
        self.df = pd.read_csv(f'data/{results[-1]}')
        self.cache_init_len = self.df.shape[0]
        self.cache_update = 20
        self.store_cache = store_cache

        print(f'Cache loaded: {results[-1]}  store_cache: {store_cache} size: {self.cache_init_len}')

        self.state = None
        self.rule_seq = []

    def reset(self):
        self.state = self.initial_state
        self.rule_seq = []
        return self.state

    def load_reward_oracle(self):
        device = 'cpu'
        self.model = Net(max_nodes=50, num_channels=39, num_outputs=1).to(device)
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'trained_models/terminal_value_function/model_state_dict_best.pt')
        self.model.load_state_dict(torch.load(model_path))
        print_info('Successfully loaded the GNN reward oracle from {}'.format(model_path))

    def reward_oracle_evaluate(self, robot_graph):
        adj_matrix_np, link_features_np, masks_np = self.preprocessor.preprocess(robot_graph)
        with torch.no_grad():
            adj_matrix = torch.tensor(adj_matrix_np).unsqueeze(0)
            link_features = torch.tensor(link_features_np).unsqueeze(0)
            masks = torch.tensor(masks_np).unsqueeze(0)
            y = self.model(link_features, adj_matrix, masks)
            reward = y[0].item()
        return reward

    def transite(self, state, action):
        applicable_matches = list(get_applicable_matches(self.rules[action], state))
        next_state = rd.apply_rule(self.rules[action], state, applicable_matches[0])
        return next_state

    def get_available_actions(self, state):
        actions_2 = np.array([idx for idx, rule in enumerate(self.rules) if list(get_applicable_matches(rule, state))])

        # actions_1 = []
        # for idx, rule in enumerate(self.rules):
        #     if list(get_applicable_matches(rule, state)):
        #         actions_1.append(idx)
        # assert actions_1 == actions_1

        return actions_2

    def get_available_actions_mask_batch(self, state_n):
        ''' 1 if the action is ok -inf otherwise '''
        actions_2_n = np.zeros((len(state_n), 20))  # 20 actions total
        action_not_available_n = np.zeros((len(state_n),))

        for i, state in enumerate(state_n):
            actions_2_n[i] = np.array([1 if list(get_applicable_matches(rule, state)) else -np.inf for rule in self.rules])
            action_not_available_n[i] = np.sum(~np.isinf(actions_2_n[i])) == 0

        return actions_2_n, action_not_available_n

    def get_available_actions_mask(self, state):
        ''' 1 if the action is ok -inf otherwise '''
        actions_2 = np.array([1 if list(get_applicable_matches(rule, state)) else -np.inf for rule in self.rules])
        action_not_available = np.sum(~np.isinf(actions_2)) == 0

        return actions_2, action_not_available

    '''
    return if the design is valid (has no self-collision)
    '''

    def is_valid(self, state):
        if has_nonterminals(state):
            return False

        robot = build_normalized_robot(state)
        _, has_self_collision = presimulate(robot)

        return not has_self_collision

    def get_reward(self, robot_graph):
        if self.enable_reward_oracle:
            return None, self.reward_oracle_evaluate(robot_graph)
        else:
            robot = build_normalized_robot(robot_graph)
            opt_seed = self.rng.getrandbits(32)

            # Cache hit could be fixed
            reward = self.df[self.df.rule_seq == str(self.rule_seq)]['result'].max()
            if not np.isnan(reward):
                print('[H]', end='\t')
                return None, reward

            # self.df[self.df.rule_seq == str(self.rule_seq)]['result'].values
            # self.df[self.df.rule_seq == str(self.rule_seq)]
            # if result_cache_key in self.result_cache:
            #     result = self.result_cache[result_cache_key]
            #     self.result_cache_hit_count += 1
            # else:
            input_sequence, reward = simulate(robot, self.task, opt_seed, self.mpc_num_processes, episode_count=1)

            if reward is None or (reward is not None and reward > self.task.result_bound):
                reward = -2.0
            else:
                # cache store only when is not -2
                self.df.loc[len(self.df.index)] = [0, self.rule_seq, opt_seed, reward]

            return input_sequence, reward

    # NOTE: the input should guarantee that the applied action is valid
    def step(self, action):
        next_state = self.transite(self.state, action)
        self.rule_seq.append(action)

        if has_nonterminals(next_state):
            reward, done = 0., False

        else:
            input_sequence, reward = self.get_reward(next_state)
            done = True
            # every time the design is complete check if I can flush the cache
            if (self.df.shape[0] - self.cache_init_len > self.cache_update) and self.store_cache:
                self.df.to_csv(f'data/result_cache_{self.df.shape[0]:05d}.csv', index=None)
                self.cache_init_len = self.df.shape[0]

        self.state = next_state

        return self.state, reward, done

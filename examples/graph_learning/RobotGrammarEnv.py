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
from design_search import make_initial_graph, build_normalized_robot, get_applicable_matches, has_nonterminals, simulate, presimulate
from common import *
from Net import Net
import torch

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
    def __init__(self, task, rules, seed = 0, mpc_num_processes = 8, enable_reward_oracle = False, preprocessor = None):
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
        self.state = None
        self.rule_seq = []
    
    def reset(self):
        self.state = self.initial_state
        self.rule_seq = []
        return self.state
    
    def load_reward_oracle(self):
        device = 'cpu'
        self.model = Net(max_nodes = 50, num_channels = 39, num_outputs = 1).to(device)
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trained_models/terminal_value_function/model_state_dict_best.pt')
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
        actions = []
        for idx, rule in enumerate(self.rules):
            if list(get_applicable_matches(rule, state)):
                actions.append(idx)
        return np.array(actions)

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
            self.last_opt_seed = opt_seed

            input_sequence, reward = simulate(robot, self.task, opt_seed, self.mpc_num_processes, episode_count = 1)

            if reward is None or (reward is not None and reward > self.task.result_bound):
                reward = -2.0

            # if reward is None:
            #     reward = -2.0

            return input_sequence, reward

    # NOTE: the input should guarantee that the applied action is valid
    def step(self, action):
        next_state = self.transite(self.state, action)
        self.rule_seq.append(action)
        if has_nonterminals(next_state):
            reward = 0.
            done = False
        else:
            input_sequence, reward = self.get_reward(next_state)
            done = True
        
        self.state = next_state
        
        return self.state, reward, done

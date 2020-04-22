'''
RobotGrammarEnv.py

Implement the environment for robot grammar search problem.

Parameters:
    task: a task to be evaluated for the design
    rules: collection of grammar rules (actions)
    seed: random seed for the env
    mpc_num_processes: number of threads for mpc
'''
# import python packages
import random
from copy import deepcopy
import numpy as np

# import our own packages
import pyrobotdesign as rd
from design_search import make_initial_graph, build_normalized_robot, get_applicable_matches, has_nonterminals, simulate
from common import *

'''
class Result

Store the mpc results for a design. include control sequence and reward

the state of the env is a robot graph.
'''
class Result:
    def __init__(self, input_sequence, reward):
        self.input_sequence = deepcopy(input_sequence)
        self.reward = reward

class RobotGrammarEnv:
    def __init__(self, task, rules, seed = 0, mpc_num_processes = 8):
        self.task = task
        self.rules = rules
        self.mpc_seed = seed
        self.mpc_num_processes = mpc_num_processes
        self.initial_state = make_initial_graph()
        self.result_cache = dict()
        self.state = None
        self.rule_seq = []
    
    def reset(self):
        self.state = self.initial_state
        self.rule_seq = []
        return self.state
    
    def get_available_actions(self):
        actions = []
        for idx, rule in enumerate(self.rules):
            if list(get_applicable_matches(rule, self.state)):
                actions.append(idx)
        return np.array(actions)

    def get_reward(self, robot_graph):
        print('computing reward by mpc: ', self.rule_seq)
        return [], 1.0
        
        robot = build_normalized_robot(robot_graph)
        opt_seed = self.mpc_seed

        robot_hash_key = hash(robot_graph)
        if robot_hash_key in self.result_cache:
            results = self.result_cache[robot_hash_key]
            input_sequence, reward = results.input_sequence, results.reward
        else:
            input_sequence, reward = simulate(robot, self.task, opt_seed, self.mpc_num_processes)
            self.result_cache[robot_hash_key] = Result(input_sequence, reward)

        return input_sequence, reward

    # NOTE: the input should guarantee that the applied action is valid
    def step(self, action):
        applicable_matches = list(get_applicable_matches(self.rules[action], self.state))
        next_state = rd.apply_rule(self.rules[action], self.state, applicable_matches[0])
        self.rule_seq.append(action)
        if has_nonterminals(next_state):
            reward = 0.
            done = False
        else:
            input_sequence, reward = self.get_reward(next_state)
            done = True
        
        self.state = next_state
        
        return self.state, reward, done

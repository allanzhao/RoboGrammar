'''
RobotGrammarEnv.py

Implement the environment for robot grammar search problem.

Parameters:
    task: a task to be evaluated for the design
    rules: collection of grammar rules (actions)
    max_nodes: maximum number of nodes in the design graph
    seed: random seed for the env
    mpc_num_processes: number of threads for mpc
'''
# import python packages
import random
from copy import deepcopy
import numpy as np

# import our own packages
from DesignGraph import DesignGraphState
import pyrobotdesign as rd
from design_search import make_initial_graph, build_normalized_robot, get_applicable_matches, has_nonterminals
from common import *

'''
class Result

Store the mpc results for a design. include control sequence and reward
'''
class Result:
    def __init__(self, input_sequence, reward):
        self.input_sequence = deepcopy(input_sequence)
        self.reward = reward

class RobotGrammarEnv:
    def __init__(self, task, rules, max_nodes = 19, seed = 0, mpc_num_processes = 8):
        self.task = task
        self.rules = rules
        self.max_nodes = max_nodes
        self.mpc_seed = seed
        self.mpc_num_processes = mpc_num_processes
        self.initial_robot_graph = make_initial_graph()
        self.result_cache = dict()
        self.current_robot_graph = None

    def get_state(self, robot_graph):
        robot = build_normalized_robot(robot_graph)
        return DesignGraphState(self.max_nodes, robot = robot)
    
    def reset(self):
        self.current_robot_graph = deepcopy(self.initial_robot_graph)
        return self.get_state(self.current_robot_graph)
    
    def get_available_actions(self):
        actions = []
        for idx, rule in enumerate(self.rules):
            if list(get_applicable_matches(rule, self.current_robot_graph)):
                actions.append(idx)
        return np.array(actions)

    def get_reward(self, robot_graph):
        robot = build_normalized_robot(robot_graph)
        opt_seed = self.mpc_seed

        robot_hash_key = hash(robot_graph)
        if robot_hash_key in self.result_cache:
            results = self.result_cache[robot_hash_key]
            input_sequence, reward = results.input_sequence, results.reward
        else:
            input_sequence, reward = simulate(robot, self.task, opt_seed, self.mpc_num_processes)
            self.result_cache[robot_hash_key] = Results(input_sequence, reward)

        return input_sequence, reward

    # NOTE: the input should guarantee that the applied action is valid
    def step(self, action):
        applicable_matches = list(get_applicable_matches(self.rules[action], self.current_robot_graph))
        next_robot_graph = rd.apply_rule(rules[action], self.current_robot_graph, applicable_matches[0])
        if has_nonterminals(next_robot_graph):
            reward = 0.
            done = False
        else:
            input_sequence, reward = get_reward(next_robot_graph)
            done = True
        
        self.current_robot_graph = next_robot_graph
        
        return get_state(self.current_robot_graph), reward, done

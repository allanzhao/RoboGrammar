import csv
import ast
import numpy as np
import argparse
from RobotGrammarEnv import RobotGrammarEnv
from design_search import make_initial_graph
import pyrobotdesign as rd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-path', type = str, required = True)
    parser.add_argument('--grammar-file', type = str, default = '../../data/designs/grammar_apr30.dot', help="Grammar file (.dot)")
    
    args = parser.parse_args()

    fp = open(args.log_path, newline = '')
    reader = csv.DictReader(fp)
    
    graphs = rd.load_graphs(args.grammar_file)
    rules = [rd.create_rule_from_graph(g) for g in graphs]

    # initialize the env
    env = RobotGrammarEnv(None, rules)

    design_cnt = dict()
    memory = dict()
    N = 0
    best_reward = []
    rewards = []
    best_design = None
    best_rule_seq = None
    for row in reader:
        N += 1
        design = row['rule_seq']
        reward = float(row['reward'])
        if design not in memory:
            memory[design] = 0
        memory[design] += 1
        rule_seq = ast.literal_eval(row['rule_seq'])
        state = make_initial_graph()
        for rule in rule_seq:
            state = env.transite(state, rule)
        if hash(state) not in design_cnt:
            design_cnt[hash(state)] = [0, reward]
        design_cnt[hash(state)][0] += 1
        design_cnt[hash(state)][1] = max(design_cnt[hash(state)][1], reward)
        if len(best_reward) == 0:
            best_reward = [reward]
        else:
            if reward > best_reward[-1]:
                best_design = state
                best_rule_seq = design
                print('best: {}, {}, hash = {}'.format(reward, design, hash(state)))

            best_reward.append(max(reward, best_reward[-1]))
        rewards.append(reward)

    fp.close()

    print('repeated rule sequence:')
    repeat = 0
    for design in memory.keys():
        repeat += memory[design] - 1
        if memory[design] > 10:
            print('rule_seq = ', design, ', cnt = ', memory[design])
    
    print('repeated design')
    repeat = 0
    for key in design_cnt.keys():
        repeat += design_cnt[key][0] - 1
        if design_cnt[key][0] > 10:
            print('reward = ', design_cnt[key][1], ', repeated: ', design_cnt[key][0])

    print('repeat = ', repeat, '/', N, ', ratio = ', repeat / N)

    print('best rule seq = {}, best reward = {}'.format(best_rule_seq, design_cnt[hash(best_design)]))
    
    iters = list(range(0, len(rewards)))
    fig, ax = plt.subplots()
    ax.scatter(iters, rewards, s = 5, c = 'tab:blue')
    ax.plot(iters, best_reward, c = 'tab:green')

    plt.show()


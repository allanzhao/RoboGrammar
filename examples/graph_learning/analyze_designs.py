import csv
import ast
import numpy as np
import argparse
from RobotGrammarEnv import RobotGrammarEnv
from design_search import make_initial_graph
import pyrobotdesign as rd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-path', type = str, required = True)
    parser.add_argument('--grammar-file', type = str, default = '../../data/designs/grammar_jan21.dot', help="Grammar file (.dot)")

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
    for row in reader:
        N += 1
        design = row['rule_seq']
        reward = row['reward']
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
        # if N == 2000:
        #     break

    fp.close()

    print('repeated rule sequence:')
    repeat = 0
    for design in memory.keys():
        repeat += memory[design] - 1
        if memory[design] > 5:
            print('rule_seq = ', design, ', cnt = ', memory[design])
    
    print('repeated design')
    repeat = 0
    for key in design_cnt.keys():
        repeat += design_cnt[key][0] - 1
        if design_cnt[key][0] > 5:
            print('reward = ', design_cnt[key][1], ', repeated: ', design_cnt[key][0])

    print('repeat = ', repeat, '/', N, ', ratio = ', repeat / N)

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
    parser.add_argument('--index', type = int, default = None, help = 'index of the designs to be shown at the end')

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
    rule_seqs = []
    opt_seeds = []
    best_design = None
    best_rule_seq = None
    best_designs = []
    for row in reader:
        N += 1
        design = row['rule_seq']
        reward = float(row['reward'])
        if 'opt_seed' in row:
            opt_seed = row['opt_seed']
        else:
            opt_seed = None
        if design not in memory:
            memory[design] = 0
        memory[design] += 1
        rule_seq = ast.literal_eval(row['rule_seq'])
        rule_seqs.append(rule_seq)
        state = make_initial_graph()
        for rule in rule_seq:
            state = env.transite(state, rule)
        if hash(state) not in design_cnt:
            design_cnt[hash(state)] = [0, reward, reward, 0]
        design_cnt[hash(state)][0] += 1
        design_cnt[hash(state)][1] = max(design_cnt[hash(state)][1], reward)
        design_cnt[hash(state)][3] += reward
        if len(best_reward) == 0:
            best_reward = [reward]
        else:
            if reward > best_reward[-1]:
                best_design = state
                best_designs.append(state)
                best_rule_seq = design
                if design_cnt[hash(state)][0] > 1:
                    new_design = False
                else:
                    new_design = True
                if new_design:
                    print('best: iter = {}, reward: {:.4f}, rule_seq: {}, hash = {}, opt_seed = {}, new'.format(N, reward, design, hash(state), opt_seed))
                else:
                    print('best: iter = {}, reward: {:.4f}, rule_seq: {}, hash = {}, opt_seed = {}, first time reward = {}'.format(N, reward, design, hash(state), opt_seed, design_cnt[hash(state)][2]))

            best_reward.append(max(reward, best_reward[-1]))
        rewards.append(reward)
        opt_seeds.append(opt_seed)

    fp.close()
    
    print('repeated design')
    repeat = 0
    for key in design_cnt.keys():
        repeat += design_cnt[key][0] - 1
        # if design_cnt[key][0] > 10:
        #     print('reward = {:.4f}'.format(design_cnt[key][1]), ', avg reward: {:.4f}'.format(design_cnt[key][3] / design_cnt[key][0]), ', hash = ', key, ', repeated: ', design_cnt[key][0])

    for state in best_designs:
        key = hash(state)
        print('reward = {:.4f}'.format(design_cnt[key][1]), ', avg reward: {:.4f}'.format(design_cnt[key][3] / design_cnt[key][0]), ', hash = ', key, ', repeated: ', design_cnt[key][0])
        
    print('repeat = ', repeat, '/', N, ', ratio = ', repeat / N)

    print('best rule seq = {}, best reward = {:.4f}, avg reward = {:.4f}'.format(best_rule_seq, design_cnt[hash(best_design)][1], design_cnt[hash(best_design)][3] / design_cnt[hash(best_design)][0]))
        
    if args.index is not None:
        print('Index {} degisn: rule_seq = {}, reward = {}, opt_seed = {}'.format(args.index, rule_seqs[args.index - 1], rewards[args.index - 1], opt_seeds[args.index - 1]))
        split_path = args.log_path.split('/')
        for folder in split_path:
            if folder[-4:] == 'Task':
                task_name = folder
            if folder[-5:-1] == 'Task':
                task_name = folder
        rule_seq_str = ''
        for i in range(len(rule_seqs[args.index - 1])):
            if i == 0:
                rule_seq_str = rule_seq_str + str(rule_seqs[args.index - 1][i])
            else:
                rule_seq_str = rule_seq_str + ', ' + str(rule_seqs[args.index - 1][i])
        print('python examples/design_search/viewer.py {} data/designs/grammar_apr30.dot -j16 {} -o -s {}'.format(task_name, rule_seq_str, opt_seeds[args.index - 1]))
    else:
        iters = list(range(0, len(rewards)))
        fig, ax = plt.subplots()
        ax.scatter(iters, rewards, s = 5, c = 'tab:blue')
        ax.plot(iters, best_reward, c = 'tab:green')

        plt.show()


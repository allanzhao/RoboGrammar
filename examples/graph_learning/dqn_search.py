import sys
import os

base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'graph_learning'))

from collections import deque, namedtuple
import random
import torch
import torch.nn.functional as F
import numpy as np

from arguments import get_parser
from utils import solve_argv_conflict
from common import *

from RobotDQN import RobotRL, ReplayMemory
from results import get_robot_image, make_robot_from_rule_sequence
import matplotlib.pyplot as plt

import pyrobotdesign as rd
import tasks
from RobotGrammarEnv import RobotGrammarEnv


def update_target_network(q_net, q_net_target):
    for target_param, param in zip(
            q_net_target.parameters(), q_net.parameters()
    ):
        target_param.data.copy_(param.data)


def optimize(robot, memory, batch_size, environment, ddqn=True):
    '''  dqn critic update  '''

    minibatch = memory.sample(batch_size)
    state_n, action_n, next_state_n, reward_n, done_n = list(map(list, zip(*minibatch)))

    # Q Network tp
    adj_matrix_tp_batch, features_tp_batch, masks_tp_batch = robot.preprocessor.preprocess_batch(state_n)

    qa_t_values, _, _ = robot.Q(features_tp_batch, adj_matrix_tp_batch, masks_tp_batch)
    q_t_values = torch.gather(qa_t_values, dim=1, index=torch.tensor(action_n).unsqueeze(1))

    # Target Network tp1
    adj_matrix_tp1_batch, features_tp1_batch, masks_tp1_batch = robot.preprocessor.preprocess_batch(next_state_n)
    qa_tp1_values, _, _ = robot.Q_target(features_tp1_batch, adj_matrix_tp1_batch, masks_tp1_batch)

    if ddqn:  # double q learning
        qnet_tp1_to_be_masked, _, _ = robot.Q(features_tp1_batch, adj_matrix_tp1_batch, masks_tp1_batch)

        actions_mask, action_not_available_n = environment.get_available_actions_mask_batch(next_state_n)
        next_actions = (qnet_tp1_to_be_masked + torch.tensor(actions_mask)).argmax(
            dim=1)  # if all -inf just procede randomly
        q_tp1 = torch.gather(qa_tp1_values, dim=1, index=next_actions.unsqueeze(1)).squeeze(1)

    else:
        q_tp1, _ = qa_tp1_values.max(dim=1)  # this works

    target = torch.tensor(reward_n) + q_tp1 * (torch.logical_not(torch.tensor(done_n)))
    target = target.unsqueeze(1).detach()

    robot.optimizer.zero_grad()
    loss = F.mse_loss(q_t_values, target)
    loss.backward()
    robot.optimizer.step()

    return loss.item()


def __save_design__(robot, rule_seq):
    _robot_ = make_robot_from_rule_sequence(rule_seq, robot.rules)
    v = get_robot_image(_robot_, robot.task, render=False)
    plt.imshow(v)
    _str_ = str(rule_seq)[1:-1].replace(', ', '_') + '.png'  # title
    plt.savefig('imgs/' + _str_, bbox_inches='tight')


# def __build_a_design__(robot, eps):
#     ''' build a design, need to pass a copy of the robot object'''
#     done = False
#     trial_before_design = 0
#     while not done:
#         state = robot.env.reset()
#         total_reward, rule_seq, state_seq = 0., [], []
#
#         for i in range(robot.hyperparams['depth']):
#             action = robot.select_action(state, eps)  # e-greedy dovremmo aspettare di riempire il replay buffer
#             if action is None:
#                 print(f'#{i} @ {rule_seq}')
#                 break
#
#             rule_seq.append(action)
#             next_state, reward, done = robot.env.step(action)
#             state_seq.append((state, action, next_state, reward, done))
#             total_reward += reward
#             state = next_state
#
#             if done:
#                 # __save_design__(robot, rule_seq)
#                 # if best_reward < total_reward:
#                 #     best_reward, best_rule_seq = total_reward, rule_seq
#                 break
#
#         trial_before_design += 1


def search(robot, environment):
    # initialize DQN
    memory = ReplayMemory(capacity=1000000)
    scores = deque(maxlen=100)
    data = []

    best_reward, best_rule_seq = 0.0, []

    for epoch in range(robot.hyperparams['num_iterations']):
        eps = robot.hyperparams['eps_start'] + epoch / robot.hyperparams['num_iterations'] * robot.hyperparams[
            'eps_delta']

        done = False
        trial_before_design = 0

        while not done:
        ################################################

            state = environment.reset()
            total_reward, rule_seq, state_seq = 0., [], []

            for i in range(robot.hyperparams['depth']):
                action = robot.select_action(environment, state, eps)  # e-greedy dovremmo aspettare di riempire il replay buffer
                if action is None:
                    print(f'#{i} @ {rule_seq}')
                    break

                rule_seq.append(action)
                next_state, reward, done = environment.step(action)
                state_seq.append((state, action, next_state, reward, done))
                total_reward += reward
                state = next_state

                if done:
                    # __save_design__(robot, rule_seq)
                    # if best_reward < total_reward:
                    #     best_reward, best_rule_seq = total_reward, rule_seq
                    break

        # if not done:  # penalize if it does not finish early
        #     total_reward = -1
        #     state_seq[-1] = (state_seq[-1][0], state_seq[-1][1], state_seq[-1][2], -1, state_seq[-1][4])  # last reward

            trial_before_design += 1

        ################################################
        for i in range(len(state_seq)):
            memory.push(*state_seq[i])
            data.append((state_seq[i][0], state_seq[i][1], total_reward))
        scores.append(total_reward)

        loss = 0.0
        if len(memory) > 64:
            for _ in range(10):  # 10 gradient descent
                loss += optimize(robot, memory, robot.hyperparams['batch_size'], environment)

        if epoch % robot.hyperparams['freq_update'] == 0:
            update_target_network(robot.Q, robot.Q_target)

        print(f'epoch {epoch} : reward = {total_reward:.2f}, eps = {eps:.2f}, Q loss = {loss:.2f}')
        print(f'memory {len(memory)} - {total_reward:.2f} @', rule_seq)

        robot.log_scalar(total_reward, name='total_reward', step_=epoch)
        robot.log_scalar(eps, name='eps', step_=epoch)
        robot.log_scalar(loss, name='q_loss', step_=epoch)
        robot.log_scalar(len(memory), name='memory', step_=epoch)
        robot.log_scalar(trial_before_design, name='trial_before_design', step_=epoch)

    # test
    cnt = 0
    for i in range(len(data)):
        if data[i][2] > 0.5:
            y_predict = robot.predict_q_values_nograd(data[i][0])
            print('target = ', data[i][2], ', predicted = ', y_predict[0][data[i][1]])
            cnt += 1
            if cnt == 5:
                break
    cnt = 0
    for i in range(len(data)):
        if data[i][2] < 0.5:
            y_predict = robot.predict_q_values_nograd(data[i][0])
            print('target = ', data[i][2], ', predicted = ', y_predict[0][data[i][1]])
            cnt += 1
            if cnt == 5:
                break


if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    args_list = ['--task', 'FlatTerrainTask',
                 # '--grammar-file', '../../data/designs/grammar_jan21.dot',
                 '--grammar-file', 'data/designs/grammar_apr30.dot',
                 '--num-iterations', '10000',
                 '--mpc-num-processes', '8',
                 '--lr', '1e-3',
                 '--eps-start', '0.2',
                 '--eps-end', '0.05',  # 0.05
                 '--batch-size', '64',
                 '--depth', '20',
                 '--seed', '0',
                 '--freq-update', '20',
                 '--batch-norm', 'False',
                 '--layer-size', '64',
                 '--save-dir', './trained_models/FlatTerrainTask/test/',
                 '--store-cache', 'True',
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

    task_class = getattr(tasks, args.task)
    task = task_class()

    graphs = rd.load_graphs(args.grammar_file)
    rules = [rd.create_rule_from_graph(g) for g in graphs]

    env = RobotGrammarEnv(task,
                          rules,
                          seed=args.seed,
                          store_cache=args.store_cache,
                          mpc_num_processes=args.mpc_num_processes)

    robot_obj = RobotRL(rules=rules, env_dummy=env, args_main=args)

    search(robot_obj, environment=env)

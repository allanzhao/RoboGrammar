'''
This code is used to evalaute the imitators trained with different number of trajectories
and plot the results in the same figure for easy comparison.
'''

import argparse
import os
import glob
import gym

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from baselines.gail import run_mujoco
from baselines.gail import mlp_policy
from baselines.common import set_global_seeds, tf_util as U
from baselines.common.misc_util import boolean_flag
from baselines.gail.dataset.mujoco_dset import Mujoco_Dset


plt.style.use('ggplot')
CONFIG = {
    'traj_limitation': [1, 5, 10, 50],
}


def load_dataset(expert_path):
    dataset = Mujoco_Dset(expert_path=expert_path)
    return dataset


def argsparser():
    parser = argparse.ArgumentParser('Do evaluation')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--policy_hidden_size', type=int, default=100)
    parser.add_argument('--env', type=str, choices=['Hopper', 'Walker2d', 'HalfCheetah',
                                                    'Humanoid', 'HumanoidStandup'])
    boolean_flag(parser, 'stochastic_policy', default=False, help='use stochastic/deterministic policy to evaluate')
    return parser.parse_args()


def evaluate_env(env_name, seed, policy_hidden_size, stochastic, reuse, prefix):

    def get_checkpoint_dir(checkpoint_list, limit, prefix):
        for checkpoint in checkpoint_list:
            if ('limitation_'+str(limit) in checkpoint) and (prefix in checkpoint):
                return checkpoint
        return None

    def policy_fn(name, ob_space, ac_space, reuse=False):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    reuse=reuse, hid_size=policy_hidden_size, num_hid_layers=2)

    data_path = os.path.join('data', 'deterministic.trpo.' + env_name + '.0.00.npz')
    dataset = load_dataset(data_path)
    checkpoint_list = glob.glob(os.path.join('checkpoint', '*' + env_name + ".*"))
    log = {
        'traj_limitation': [],
        'upper_bound': [],
        'avg_ret': [],
        'avg_len': [],
        'normalized_ret': []
    }
    for i, limit in enumerate(CONFIG['traj_limitation']):
        # Do one evaluation
        upper_bound = sum(dataset.rets[:limit])/limit
        checkpoint_dir = get_checkpoint_dir(checkpoint_list, limit, prefix=prefix)
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
        env = gym.make(env_name + '-v1')
        env.seed(seed)
        print('Trajectory limitation: {}, Load checkpoint: {}, '.format(limit, checkpoint_path))
        avg_len, avg_ret = run_mujoco.runner(env,
                                             policy_fn,
                                             checkpoint_path,
                                             timesteps_per_batch=1024,
                                             number_trajs=10,
                                             stochastic_policy=stochastic,
                                             reuse=((i != 0) or reuse))
        normalized_ret = avg_ret/upper_bound
        print('Upper bound: {}, evaluation returns: {}, normalized scores: {}'.format(
            upper_bound, avg_ret, normalized_ret))
        log['traj_limitation'].append(limit)
        log['upper_bound'].append(upper_bound)
        log['avg_ret'].append(avg_ret)
        log['avg_len'].append(avg_len)
        log['normalized_ret'].append(normalized_ret)
        env.close()
    return log


def plot(env_name, bc_log, gail_log, stochastic):
    upper_bound = bc_log['upper_bound']
    bc_avg_ret = bc_log['avg_ret']
    gail_avg_ret = gail_log['avg_ret']
    plt.plot(CONFIG['traj_limitation'], upper_bound)
    plt.plot(CONFIG['traj_limitation'], bc_avg_ret)
    plt.plot(CONFIG['traj_limitation'], gail_avg_ret)
    plt.xlabel('Number of expert trajectories')
    plt.ylabel('Accumulated reward')
    plt.title('{} unnormalized scores'.format(env_name))
    plt.legend(['expert', 'bc-imitator', 'gail-imitator'], loc='lower right')
    plt.grid(b=True, which='major', color='gray', linestyle='--')
    if stochastic:
        title_name = 'result/{}-unnormalized-stochastic-scores.png'.format(env_name)
    else:
        title_name = 'result/{}-unnormalized-deterministic-scores.png'.format(env_name)
    plt.savefig(title_name)
    plt.close()

    bc_normalized_ret = bc_log['normalized_ret']
    gail_normalized_ret = gail_log['normalized_ret']
    plt.plot(CONFIG['traj_limitation'], np.ones(len(CONFIG['traj_limitation'])))
    plt.plot(CONFIG['traj_limitation'], bc_normalized_ret)
    plt.plot(CONFIG['traj_limitation'], gail_normalized_ret)
    plt.xlabel('Number of expert trajectories')
    plt.ylabel('Normalized performance')
    plt.title('{} normalized scores'.format(env_name))
    plt.legend(['expert', 'bc-imitator', 'gail-imitator'], loc='lower right')
    plt.grid(b=True, which='major', color='gray', linestyle='--')
    if stochastic:
        title_name = 'result/{}-normalized-stochastic-scores.png'.format(env_name)
    else:
        title_name = 'result/{}-normalized-deterministic-scores.png'.format(env_name)
    plt.ylim(0, 1.6)
    plt.savefig(title_name)
    plt.close()


def main(args):
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(args.seed)
    print('Evaluating {}'.format(args.env))
    bc_log = evaluate_env(args.env, args.seed, args.policy_hidden_size,
                          args.stochastic_policy, False, 'BC')
    print('Evaluation for {}'.format(args.env))
    print(bc_log)
    gail_log = evaluate_env(args.env, args.seed, args.policy_hidden_size,
                            args.stochastic_policy, True, 'gail')
    print('Evaluation for {}'.format(args.env))
    print(gail_log)
    plot(args.env, bc_log, gail_log, args.stochastic_policy)


if __name__ == '__main__':
    args = argsparser()
    main(args)

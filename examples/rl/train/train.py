import sys
import os
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'rl'))

import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
gym.logger.set_level(40)

import environments
# from rl.train.evaluation import evaluate, render
from rl.train.arguments import get_parser
from rl.train.utils import solve_argv_conflict
from common.common import *
from rl.train.evaluation import render

# import a2c_ppo_acktr
from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.envs import make_vec_envs, make_env
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage

def train(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device('cpu')

    os.makedirs(args.save_dir, exist_ok = True)

    training_log_path = os.path.join(args.save_dir, 'logs.txt')
    fp_log = open(training_log_path, 'w')
    fp_log.close()

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes, 
                        args.gamma, None, device, False, args = args)

    render_env = gym.make(args.env_name, args = args)
    render_env.seed(args.seed)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)
    
    if args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    else:
        raise NotImplementedError

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)
    
    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)
    
    episode_rewards = deque(maxlen=10)
    episode_lens = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)
            
            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    episode_lens.append(info['episode']['l'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            model_save_dir = os.path.join(args.save_dir, 'models')
            os.makedirs(model_save_dir, exist_ok = True)
            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(model_save_dir, args.env_name + '_iter{}'.format(j) + ".pt"))

        # save logs of every episode
        fp_log = open(training_log_path, 'a')
        total_num_steps = (j + 1) * args.num_processes * args.num_steps
        len_mean, len_min, len_max = np.mean(episode_lens), np.min(episode_lens), np.max(episode_lens)
        reward_mean, reward_min, reward_max = np.mean(episode_rewards), np.min(episode_rewards), np.max(episode_rewards)
        fp_log.write('iterations: {}, mean(len): {:.1f}, min(len): {}, max(len): {}, mean(reward): {:.3f}, min(reward): {:.3f}, max(reward): {:.3f}, value_loss: {:.3f}, action_loss: {:.3f}\n'.format(
            total_num_steps, len_mean, len_min, len_max, reward_mean, reward_min, reward_max, value_loss, action_loss))
        fp_log.close()

        # logging to console
        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {}, time {} minutes \n Last {} training episodes: mean/median length {:1f}/{}, min/max length {}/{} mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        (end - start) / 60., 
                        len(episode_rewards), 
                        np.mean(episode_lens), np.median(episode_lens), 
                        np.min(episode_lens), np.max(episode_lens),
                        np.mean(episode_rewards), np.median(episode_rewards), 
                        np.min(episode_rewards), np.max(episode_rewards), 
                        dist_entropy, value_loss,
                        action_loss))

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(args, actor_critic, ob_rms, args.env_name, args.seed,
                     args.num_processes, device)

        if (args.render_interval is not None and args.render_interval > 0 and j % args.render_interval == 0):
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            render(render_env, actor_critic, ob_rms, deterministic = True)

    render_env.close()
    envs.close()

if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    args_list = ['--env-name', 'RobotLocomotion-v0',
                 '--task', 'FlatTerrainTask',
                 '--grammar-file', '../../data/designs/grammar_jan21.dot',
                 '--algo', 'ppo',
                 '--use-gae',
                 '--log-interval', '5',
                 '--num-steps', '1024',
                 '--num-processes', '8',
                 '--lr', '3e-4',
                 '--entropy-coef', '0',
                 '--value-loss-coef', '0.5',
                 '--ppo-epoch', '10',
                 '--num-mini-batch', '32',
                 '--gamma', '0.995',
                 '--gae-lambda', '0.95',
                 '--num-env-steps', '30000000',
                 '--use-linear-lr-decay',
                 '--use-proper-time-limits',
                 '--save-interval', '100',
                 '--seed', '2',
                 '--save-dir', './trained_models/RobotLocomotion-v0/test/',
                 '--render-interval', '80']
    
    solve_argv_conflict(args_list)
    parser = get_parser()
    args = parser.parse_args(args_list + sys.argv[1:])

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    args.save_dir = os.path.join(args.save_dir, get_time_stamp())
    try:
        os.makedirs(args.save_dir, exist_ok = True)
    except OSError:
        pass

    fp = open(os.path.join(args.save_dir, 'args.txt'), 'w')
    fp.write(str(args_list + sys.argv[1:]))
    fp.close()

    train(args)
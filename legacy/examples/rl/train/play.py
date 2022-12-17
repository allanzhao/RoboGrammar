import sys
import os
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'rl'))

import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
gym.logger.set_level(40)

import environments
from rl.train.evaluation import render, render_full
from rl.train.arguments import get_parser

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.envs import make_vec_envs, make_env
from a2c_ppo_acktr.model import Policy

parser = get_parser()
parser.add_argument('--model-path', type = str, required = True)

args = parser.parse_args()

if not os.path.isfile(args.model_path):
    print_error('Model file does not exist')

torch.manual_seed(0)
torch.set_num_threads(1)
device = torch.device('cpu')

render_env = gym.make(args.env_name, args = args)
render_env.seed(0)

envs = make_vec_envs(args.env_name, 0, 4, 0.995, None, device, False, args = args)

actor_critic = Policy(
    envs.observation_space.shape,
    envs.action_space,
    base_kwargs={'recurrent': False})
actor_critic.to(device)

ob_rms = utils.get_vec_normalize(envs).ob_rms

actor_critic, ob_rms = torch.load(args.model_path)

actor_critic.eval()

envs.close()

render_full(render_env, actor_critic, ob_rms, deterministic = True, repeat = True)



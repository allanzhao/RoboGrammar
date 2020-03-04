import sys
import os
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(base_dir)

import numpy as np
import matplotlib.pyplot as plt
import argparse
from common import *

def get_value(str_data):
    return float(str_data.split()[-1])

parser = argparse.ArgumentParser()
parser.add_argument('--log-dir', type = str, required = True)
parser.add_argument('--save-path', type = str, default = None)

args = parser.parse_args()

log_path = os.path.join(args.log_dir, 'logs.txt')

if not os.path.isfile(log_path):
    print_error('Log file does not exist')

fp = open(log_path, 'r')
data = fp.readlines()
fp.close()

iterations = []
len_mean, len_min, len_max = [], [], []
reward_mean, reward_min, reward_max = [], [], [] 
value_loss, action_loss = [], []
for i in range(len(data)):
    linedata = data[i].split(',')
    iterations.append(get_value(linedata[0]))
    len_mean.append(get_value(linedata[1]))
    len_min.append(get_value(linedata[2]))
    len_max.append(get_value(linedata[3]))
    reward_mean.append(get_value(linedata[4]))
    reward_min.append(get_value(linedata[5]))
    reward_max.append(get_value(linedata[6]))
    value_loss.append(get_value(linedata[7]))
    action_loss.append(get_value(linedata[8]))

fig, ax = plt.subplots(2, 2, figsize = (10, 10))

ax[0][0].set_title('episode length')
ax[0][0].set_xlabel('steps')
ax[0][0].set_ylabel('length')
ax[0][0].plot(iterations, len_mean, c = 'orange')

ax[0][1].set_title('episode reward')
ax[0][1].set_xlabel('steps')
ax[0][1].set_ylabel('reward')
ax[0][1].plot(iterations, reward_mean, c = 'orange')
ax[0][1].fill_between(iterations, reward_min, reward_max, color = 'orange', alpha = 0.1)

ax[1][0].set_title('value loss')
ax[1][0].set_xlabel('steps')
ax[1][0].set_ylabel('loss')
ax[1][0].plot(iterations, value_loss, c = 'red')

ax[1][1].set_title('action loss')
ax[1][1].set_xlabel('steps')
ax[1][1].set_ylabel('loss')
ax[1][1].plot(iterations, action_loss, c = 'red')

if args.save_path is not None:
    plt.savefig(args.save_path)

plt.show()
import argparse
import deque
import random
import torch
from torch import optim
import torch.nn.functional as F
import numpy as np

from RobotGrammarEnv import RobotGrammarEnv
from Net import Net
from arguments import get_parser
import tasks
import pyrobotdesign as rd
from utils import solve_argv_conflict

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):
    def __init__(self, capacity = 1000000):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    
    def push(self, *args):
        self.memory.append(Transition(*args))
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

def predict(Q, state):
    with torch.no_grad():
        features = torch.tensor(state.features).unsqueeze(-1)
        adj_matrix = torch.tensor(state.adj_matrix).unsqueeze(-1)
        masks = torch.tensor(state.masks).unsqueeze(-1)

    return Q(features, adj_matrix, masks)

def select_action(env, Q, state, eps):
    available_actions = env.get_available_actions()
    if len(available_actions) == 0:
        return None
    sample = random.random()
    if sample > eps_threshold:
        evals, _, _ = predict(Q, state)
        best_action = available_actions[0]
        for action in available_actions:
            if evals[0][action] > evals[0][best_action]:
                best_action = action
    else:
        best_action = available_actions[random.randrange(len(available_actions))]
    
    return best_action

def optimize(Q, Q_target, memory, batch_size):
    minibatch = memory.sample(batch_size)

    features_batch, adj_matrix_batch, masks_batch, y_batch = []
    for state, action, next_state, reward, done in minibatch:
        y_target, _, _ = predict(Q, state)
        if done:
            y_target[0][action] = reward
        else:
            y_next_state, _, _ = predict(Q_target, next_state)
            y_target[0][action] = reward + \
                np.max(y_next_state0])
        features_batch.append(state.features)
        adj_matrix_batch.append(state.adj_matrix)
        masks_batch.append(state.masks)
        y_batch.append(y_target[0])
    
    features_batch = torch.tesor(features_batch)
    adj_matrix_batch = torch.tensor(adj_matrix_batch)
    masks_batch = torch.tensor(masks_batch)
    y_batch = torch.tensor(y_batch)

    optimizer.zero_grad()
    output, loss_link, loss_entropy = Q(features_batch, adj_matrix_batch, masks_batch)
    loss = F.mse_loss(output[:, 0], y_batch)
    loss.backward()
    optimizer.step()

def search(args):
    # initialize the env
    task_class = getattr(tasks, args.task)
    task = task_class()
    graphs = rd.load_graphs(args.grammar_file)
    rules = [rd.create_rule_from_graph(g) for g in graphs]
    env = RobotGrammarEnv(task, ruls, max_nodes = 19, seed = args.seed, mpc_num_processes = args.mpc_num_processes)

    # initialize Q function
    device = 'cpu'
    Q = Net().to(device)

    # initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

    # initialize DQN
    memory = ReplayMemory(capacity = 1000000)
    scores = deque(maxlen = 100)

    for epoch in range(args.num_iterations):
        state = env.reset()
        done = False
        eps = eps_start + epoch / args.num_iterations * (eps_end - eps_start)
        while not done:
            total_reward = 0.
            for i in range(args.depth):
                action = select_action(env, Q, state, eps)
                if action is None:
                    break
                next_state, reward, done = env.step(action)
                memory.push(state, action, next_state, reward, done)
                total_reward += reward
                state = next_state
                if done:
                    break
        score.append(total_reward)
        print('epoch ', epoch, ': reward = ', total_reward)

        optimize(Q, Q, memory, args.batch_size)

if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    args_list = ['--task', 'FlatTerrainTask',
                 '--grammar-file', '../../data/designs/grammar_jan21.dot',
                 '--num-iterations', '100000',
                 '--mpc-num-processes', '8',
                 '--lr', '1e-3',
                 '--eps-start', '0.9',
                 '--eps-end', '0.05',
                 '--batch-size', '64',
                 '--save-dir', './trained_models/FlatTerrainTask/test/',
                 '--render-interval', '80']
    
    solve_argv_conflict(args_list)
    parser = get_parser()
    args = parser.parse_args(args_list + sys.argv[1:])
    
    # TODO: load cached mpc results
    # if args.log_file:

    args.save_dir = os.path.join(args.save_dir, get_time_stamp())
    try:
        os.makedirs(args.save_dir, exist_ok = True)
    except OSError:
        pass
    
    fp = open(os.path.join(args.save_dir, 'args.txt'), 'w')
    fp.write(str(args_list + sys.argv[1:]))
    fp.close()

    search(args)


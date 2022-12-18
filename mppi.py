import multiprocessing as mp

import numpy as np

from env import SimEnvWrapper
from utils import get_make_sim_and_task_fn_without_args, stack_tensor_dict_list


def do_env_rollout(env, act_list, neural_input=None):
    """
        1) Construt env based on desired behavior and set to start_state.
        2) Generate rollouts using act_list.
           act_list is a list with each element having size (H,m).
           Length of act_list is the number of desired rollouts.
    """
    env.reset()
    env.real_env_step(False)
    
    paths = []
    
    H = act_list[0].shape[0]
    N = len(act_list)
    _neural_input = None
        
    for i in range(N):
        env.set_env_state()
        act = []
        rewards = []

        for k in range(H):
            act.append(act_list[i][k])
            
            if neural_input is not None:
                _neural_input = neural_input[k]
            _, r, _, _ = env.step(act[-1], _neural_input)
            
            rewards.append(r)

        path = dict(actions=np.array(act), rewards=np.array(rewards))
        paths.append(path)
    return paths


def generate_perturbed_actions(base_act, filter_coefs, neural_input=None, history=None, idx=0):
    """
    Generate perturbed actions around a base action sequence
    """
    
    # Half of the samples are based on repeating past motion ("history")
    
    from_hist = idx % 2 == 0
    if history is not None and from_hist:
        repeat_len = int((idx / 2) % (base_act.shape[0] / 2) + base_act.shape[0] / 2 + 1)
        
        tau = (np.arange(base_act.shape[0]) + 1) / base_act.shape[0]
        tau = tau.reshape((-1, 1))
        
        history = np.repeat(history[-repeat_len:], np.ceil(base_act.shape[0] / repeat_len), axis=0)[:base_act.shape[0]]
        base_act = tau * history + (1 - tau) * base_act
        sigma = 0.05
    else:
        sigma = 0.25
        
    betas = np.array(filter_coefs).reshape((-1, 1))
    eps = np.random.normal(loc=0, scale=sigma, size=(base_act.shape[0] + betas.shape[0], base_act.shape[1]))
    for i in range(0, eps.shape[0] - betas.shape[0]):
        eps[i] = np.sum(eps[i:i+betas.shape[0]] * betas, axis=0)
        
    if neural_input is not None:
        # compute relative changes in position
        relative_position_delta = base_act - np.concatenate((np.expand_dims(history[-1], axis=0), base_act[:-1]), axis=0)
        # multiply by neural input 
        # print(relative_position_delta.shape, base_act.shape, neural_input.shape)
        relative_position_delta *= neural_input[:base_act.shape[0], :base_act.shape[1]]
        # construct base_act back
        base_act = history[-1] + np.cumsum(relative_position_delta, axis=0)
    
    base_act += eps[:-betas.shape[0]]
    return base_act


def generate_paths(args):
    """
    first generate enough perturbed actions
    then do rollouts with generated actions
    set seed inside this function for multiprocessing
    """
    N, base_act, filter_coefs, base_seed, neural_input, history = args
    
    np.random.seed(base_seed)
    act_list = []

    make_sim_and_task_fn = get_make_sim_and_task_fn_without_args()
    
    env = SimEnvWrapper(make_sim_and_task_fn, load=True)
    env.set_seed(base_seed)
    
    for idx in range(N):
        act = generate_perturbed_actions(base_act, filter_coefs, neural_input=None, history=history, idx=idx)
        act_list.append(act)
        
    paths = do_env_rollout(env, act_list, neural_input=neural_input)
    return paths


def gather_paths_parallel(history, base_act, filter_coefs, neural_input, base_seed, paths_per_cpu, num_cpu=None):
    num_cpu = 1 if num_cpu is None else num_cpu
    num_cpu = mp.cpu_count() if num_cpu == 'max' else num_cpu
    assert isinstance(num_cpu, int)
    
    pool = mp.Pool(processes=num_cpu)
    input_lists = list()
    
    for i in range(num_cpu):
        cpu_seed = base_seed + i*paths_per_cpu
        input_lists.append((paths_per_cpu, base_act, filter_coefs, cpu_seed, neural_input, history))
        
    results = pool.map(generate_paths, input_lists)
    
    paths = []
    for result in results:
        for path in result:
            paths.append(path)
    
    return paths


class MPPI:
    def __init__(self, env, H, paths_per_cpu,
                 num_cpu=1,
                 kappa=1.0,
                 gamma=1.0,
                 mean=None, # originally, mean == 0 is used to populate the act_sequence.
                 filter_coefs=None,
                 default_act='repeat',
                 warmstart=True,
                 seed=123,
                 neuron_stream=None,
                 task_args=None):
        
        self.env, self.seed = env, seed
        self.n, self.m = env.observation_dim, env.action_dim
        self.H, self.paths_per_cpu, self.num_cpu = H, paths_per_cpu, num_cpu
        self.warmstart = warmstart

        self.mean, self.filter_coefs, self.kappa, self.gamma = mean, filter_coefs, kappa, gamma
        if mean is None:
            self.mean = np.zeros(self.m)
        if filter_coefs is None:
            self.filter_coefs = [np.ones(self.m), 1.0, 0.0, 0.0]
            
        self.default_act = default_act
        self.task_args = task_args

        self.sol_act = []

        self.env.reset(seed=seed)
        
        self.act_sequence = np.ones((self.H, self.m)) * self.mean
        self.init_act_sequence = self.act_sequence.copy()
        
        if neuron_stream is not None:
            self.neuron_stream = neuron_stream
            self.neuron_stream_full = self.neuron_stream.get_spike_frequencies_array(most_current=True)
            # TODO: check what is returned here.
            self.neuron_stream_full = self.neuron_stream_full.T @ self.neuron_stream_wrapper.weights
        else:
            self.neuron_stream, self.neuron_stream_full = None, None

    def update(self, paths):
        act = np.array([paths[i]["actions"] for i in range(len(paths))])
        R = self.score_trajectory(paths)        
        S = np.exp(self.kappa*(R-np.max(R)))

        # blend the action sequence
        weighted_seq = S*act.T
        act_sequence = np.sum(weighted_seq.T, axis=0)/(np.sum(S) + 1e-6)
        self.act_sequence = act_sequence

    def advance_time(self, act_sequence=None):
        act_sequence = self.act_sequence if act_sequence is None else act_sequence
        # accept first action and step
        action = act_sequence[0].copy()
        self.env.real_env_step(True)
        neural_input = self.neuron_stream_full[len(self.sol_act)] if self.neuron_stream_full is not None else None
        _, r, _, _ = self.env.step(action, neural_input=neural_input)
        self.sol_act.append(action)
        
        # get updated action sequence
        if self.warmstart:
            self.act_sequence[:-1] = act_sequence[1:]
            if self.default_act == 'repeat':
                self.act_sequence[-1] = self.act_sequence[-2]
            elif self.default_act == 'mean':
                self.act_sequence[-1] = self.mean.copy()
            else:
                raise NotImplementedError
        else:
            self.act_sequence = self.init_act_sequence.copy()

    def score_trajectory(self, paths, offset=0):
        scores = np.zeros(len(paths))
        for i in range(len(paths)):
            scores[i] = 0.0
            for t in range(paths[i]["rewards"].shape[0]):
                scores[i] += (self.gamma**t)*paths[i]["rewards"][t]
                
        if offset > 0:
            scores_argsort = np.argsort(scores)[::-1]
            scores_argsort += offset
            scores_argsort %= len(paths)
            scores = scores[scores_argsort]
                
        return scores

    def do_rollouts(self, seed):
        history = np.zeros((self.H, self.m))
        for i in range(min(self.H, len(self.sol_act))):
            history[-i-1] = self.sol_act[-i-1]
            
        if self.neuron_stream is not None:
            step = len(self.sol_act)
            neural_input = self.neuron_stream_full[step:step+self.H, :]
        else:
            neural_input = None
        
        paths = gather_paths_parallel(history=history,
                                      base_act=self.act_sequence,
                                      filter_coefs=self.filter_coefs,
                                      neural_input=neural_input,
                                      base_seed=seed,
                                      paths_per_cpu=self.paths_per_cpu,
                                      num_cpu=self.num_cpu)
        return paths

    def train_step(self, niter=1):
        t = len(self.sol_state) - 1
        for _ in range(niter):
            paths = self.do_rollouts(self.seed+t)
            self.update(paths)
        self.advance_time()



from abc import ABC, abstractmethod

from mjrl.utils.gym_env import GymEnv
import numpy as np
from time import sleep

class Env(ABC):
    @property
    @abstractmethod
    def initial_state(self):
        """Return the initial state, which should always be the same."""
        pass

    @abstractmethod
    def get_available_actions(self, state):
        """Return an iterable containing all actions that can be taken from the
        given state."""
        pass

    @abstractmethod
    def get_next_state(self, state, action):
        """Take the action from the given state and return the resulting state."""
        pass

    @abstractmethod
    def get_result(self, state):
        """Return the result of a playout ending in the given state (None if the
        result is unknown)."""
        pass

    @abstractmethod
    def get_key(self, state):
        """Return a key identifying the given state. The key may not be unique, as
        long as collisions are very unlikely."""
        pass


class EnvWrapper(GymEnv):
    
    def __init__(self, make_sim_and_task_fn, load=False):
        self.env, self.task = make_sim_and_task_fn()
        
        if load:
            self.env.restore_state_from_file("tmp.bullet")
        else:
            self.env.save_state_to_file("tmp.bullet")
        
        self._observation_dim = (None, )
        self._action_dim = self.env.get_robot_dof_count(0)
        self.seed = None
        self.real_step = False    
    
    def step(self, action, neural_input=None):
        r = 0
        
        if neural_input is not None:
            neural_input /= np.linalg.norm(neural_input, ord=1)
            neural_input /= neural_input.max()
            # neural_input = 0.05 + 0.95 * (neural_input - neural_input.min()) / (neural_input.max() - neural_input.min())
            for link, _neural_input in zip(self.env.get_robot(0).links, neural_input):
                link.joint_torque = _neural_input
        
        for k in range(self.task.interval):
                self.env.set_joint_targets(0, action.reshape(-1, 1))
                self.task.add_noise(self.env, (self.task.interval * self.seed + k) % (2 ** 32))
                self.env.step()
                
                r += self.task.get_objective_fn()(self.env, neural_input)
        
        if self.real_step:
            self.env.save_state_to_file("tmp.bullet")
        
        return None, r, None, None
    
    def reset(self, seed=None):
        if self.seed is None and seed is not None:
            self.seed = seed
    
    def set_seed(self, seed=None):
        self.seed = seed
    
    def get_env_state(self):
        return np.zeros(0)
    
    def get_obs(self):
        return None
    
    def set_env_state(self):
        self.env.restore_state()
    
    def real_env_step(self, boolean):
        self.real_step = boolean
        if not boolean:
            self.env.save_state()
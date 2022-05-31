from abc import ABC, abstractmethod

from mjrl.utils.gym_env import GymEnv
import numpy as np

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
  
  def __init__(self, make_sim_and_task_fn):
    self.env, self.task = make_sim_and_task_fn()
    self._observation_dim = (None, )
    self._action_dim = self.env.get_robot_dof_count(0)
    self.seed = None
  
  def step(self, action):
    r = 0
    for k in range(self.task.interval):
        self.env.set_joint_targets(0, action.reshape(-1, 1))
        self.task.add_noise(self.env, (self.task.interval * self.seed + k) % (2 ** 32))
        self.env.step()
        
        r += self.task.get_objective_fn()(self.env)
        
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
  
  def set_env_state(self, state):
    self.env.restore_state()
  
  def real_env_step(self, boolean):
    if not boolean:
      self.env.save_state()
from abc import ABC, abstractmethod
from collections import defaultdict
from math import log, sqrt
import random

class TreeNode(object):
  def __init__(self, state):
    self.state = state
    self.visit_count = 0
    self.result_sum = 0
    self.action_visit_counts = defaultdict(int)
    self.action_result_sums = defaultdict(float)
    self.blocked = False

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

class TreeSearch(object):
  def __init__(self, env, max_tries=100):
    self.env = env
    self.max_tries = max_tries
    self.nodes = dict() # Mapping from state keys to nodes
    self.nodes[env.get_key(env.initial_state)] = TreeNode(env.initial_state)

  def uct_score(self, node, action):
    action_visit_count = node.action_visit_counts[action]
    action_result_sum = node.action_result_sums[action]
    if action_visit_count > 0:
      return (action_result_sum / action_visit_count +
              sqrt(2.0 * log(node.visit_count) / action_visit_count))
    else:
      return float('inf')

  def select_action(self, state):
    available_actions = list()
    # Filter out actions leading to blocked nodes
    for action in self.env.get_available_actions(state):
      next_state = self.env.get_next_state(state, action)
      next_state_key = self.env.get_key(next_state)
      if (next_state_key not in self.nodes or
          not self.nodes[next_state_key].blocked):
        available_actions.append(action)

    if available_actions:
      try:
        # Follow tree policy
        node = self.nodes[self.env.get_key(state)]
        return max(available_actions,
                   key=lambda action: self.uct_score(node, action))
      except KeyError:
        # State was not visited yet, follow default policy
        return random.choice(available_actions)
    else:
      return None

  def update_node(self, node, action, result):
    node.visit_count += 1
    node.result_sum += result
    if action:
      node.action_visit_counts[action] += 1
      node.action_result_sums[action] += result

  def run_iteration(self):
    result = None
    while result is None:
      # Selection phase
      states = [self.env.initial_state]
      actions = []
      action = self.select_action(states[-1])
      while action is not None and self.env.get_key(states[-1]) in self.nodes:
        states.append(self.env.get_next_state(states[-1], action))
        actions.append(action)
        action = self.select_action(states[-1])

      # Expansion phase
      last_state_key = self.env.get_key(states[-1])
      if last_state_key in self.nodes:
        last_node = self.nodes[last_state_key]
      else:
        last_node = TreeNode(states[-1])
        self.nodes[last_state_key] = last_node

      # Simulation phase
      for try_count in range(self.max_tries):
        sim_states = states.copy()
        sim_actions = actions.copy()
        action = self.select_action(sim_states[-1])
        while action is not None:
          sim_states.append(self.env.get_next_state(sim_states[-1], action))
          sim_actions.append(action)
          action = self.select_action(sim_states[-1])
        result = self.env.get_result(sim_states[-1])
        if result is not None:
          # Result is valid
          break

      if result is None:
        # No valid simulation after max_tries tries, block the last node
        # Next loop iteration will select a different node
        last_node.blocked = True

    # Backpropagation phase
    for state, action in zip(sim_states, sim_actions):
      try:
        node = self.nodes[self.env.get_key(state)]
        self.update_node(node, action, result)
      except KeyError:
        pass

    return sim_states, sim_actions, result

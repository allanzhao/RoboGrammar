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
  def __init__(self, env):
    self.env = env
    self.nodes = dict() # Mapping from state keys to nodes
    self.nodes[env.get_key(env.initial_state)] = TreeNode(env.initial_state)
    self.blocked_state_keys = set() # Keys of states that should not be visited

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
    for action in self.env.get_available_actions(state):
      # Filter out actions that lead to a blocked state
      next_state = self.env.get_next_state(state, action)
      if self.env.get_key(next_state) not in self.blocked_state_keys:
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
    # Retry until we get a valid result
    while result is None:
      states = []
      actions = []

      # Selection and simulation
      state = self.env.initial_state
      action = self.select_action(state)
      states.append(state)
      actions.append(action)
      result = self.env.get_result(state)
      # Stop when the result is known or no more actions are possible
      while result is None and action is not None:
        state = self.env.get_next_state(state, action)
        action = self.select_action(state)
        states.append(state)
        actions.append(action)
        result = self.env.get_result(state)

      if result is None and action is None:
        # Last state is a dead end, block it from being visited again
        self.blocked_state_keys.add(self.env.get_key(state))

    # Update tree nodes (includes expansion step)
    for state, action in zip(states, actions):
      try:
        node = self.nodes[self.env.get_key(state)]
        self.update_node(node, action, result)
      except KeyError:
        # Node does not exist yet, create one
        node = TreeNode(state)
        self.update_node(node, action, result)
        self.nodes[self.env.get_key(state)] = node
        # Create at most one new node per simulation
        break

    return states, actions, result

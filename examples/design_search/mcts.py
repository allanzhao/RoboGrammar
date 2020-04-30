from collections import defaultdict
from math import log, sqrt
import random

class TreeNode(object):
  def __init__(self, state):
    self.state = state
    self.visit_count = 0
    self.result_sum = 0
    self.result_max = float('-inf')
    self.action_visit_counts = defaultdict(int)
    self.action_result_sums = defaultdict(float)
    self.action_result_maxes = defaultdict(lambda: float('-inf'))
    self.amaf_action_visit_counts = defaultdict(int)
    self.amaf_action_result_sums = defaultdict(float)
    self.amaf_action_result_maxes = defaultdict(lambda: float('-inf'))
    self.blocked = False

class TreeSearch(object):
  def __init__(self, env, max_tries, default_policy=None):
    self.env = env
    self.max_tries = max_tries
    self.nodes = dict() # Mapping from state keys to nodes
    self.nodes[env.get_key(env.initial_state)] = TreeNode(env.initial_state)
    self.default_policy = default_policy

  def uct_score(self, node, action, amaf_threshold=10):
    action_visit_count = node.action_visit_counts[action]
    action_result_max = node.action_result_maxes[action]
    amaf_action_visit_count = node.amaf_action_visit_counts[action]
    amaf_action_result_max = node.amaf_action_result_maxes[action]
    # AMAF and Monte Carlo values are weighted equally when the visit count is
    # amaf_threshold
    amaf_weight = sqrt(amaf_threshold / (3 * node.visit_count + amaf_threshold))
    if action_visit_count > 0:
      return ((1.0 - amaf_weight) * action_result_max +
              amaf_weight * amaf_action_result_max +
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
        if self.default_policy is not None:
          return self.default_policy(state, available_actions)
        else:
          return random.choice(available_actions)
    else:
      return None

  def update_node(self, node, actions_after, result):
    node.visit_count += 1
    node.result_sum += result
    node.result_max = max(node.result_max, result)
    node.action_visit_counts[actions_after[0]] += 1
    node.action_result_sums[actions_after[0]] += result
    node.action_result_maxes[actions_after[0]] = \
        max(node.action_result_maxes[actions_after[0]], result)
    # Update AMAF values (once for each unique action)
    for action in set(actions_after):
      node.amaf_action_visit_counts[action] += 1
      node.amaf_action_result_sums[action] += result
      node.amaf_action_result_maxes[action] = \
          max(node.amaf_action_result_maxes[action], result)

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
        print("Blocked node:", [self.env.rules.index(rule) for rule in last_node.state[1]])

    # Backpropagation phase
    for i, state in enumerate(sim_states[:-1]):
      actions_after = sim_actions[i:]
      try:
        node = self.nodes[self.env.get_key(state)]
        self.update_node(node, actions_after, result)
      except KeyError:
        pass

    return sim_states, sim_actions, result

import mcts
import random

class TestEnv(mcts.Env):
  """Find the largest max_len digit number which is at most max_val"""

  all_actions = list('0123456789') # All decimal digits

  def __init__(self, max_len, max_val):
    self.max_len = max_len
    self.max_val = max_val

  @property
  def initial_state(self):
    return ''

  def get_available_actions(self, state):
    if len(state) < self.max_len:
      return TestEnv.all_actions
    else:
      return []

  def get_next_state(self, state, action):
    return state + action

  def get_result(self, state):
    val = int(state)
    if val <= self.max_val:
      return random.normalvariate(val / self.max_val, 0.1)
    else:
      return 0

  def get_key(self, state):
    return state

env = TestEnv(max_len=5, max_val=50000)
tree_search = mcts.TreeSearch(env)

for i in range(10000):
  states, actions, result = tree_search.run_iteration()
  if i % 1000 == 0:
    print(i, states[-1])

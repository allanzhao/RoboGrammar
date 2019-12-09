from mcts import Env, make_initial_tree, run_mcts_iteration
from operator import add
import random

# Toy example: find the largest MAX_LEN digit number which is at most MAX_VAL
max_len = 5
max_val = 50000

all_moves = list('0123456789') # All decimal digits
def get_available_moves(state):
  if len(state) < max_len:
    return all_moves
  else:
    return []

# Noisy objective function
def evaluate(state):
  val = int(state)
  if val <= max_val:
    return random.normalvariate(val / max_val, 0.1)
  else:
    return 0

env = Env(initial_state='', get_available_moves=get_available_moves,
          get_next_state=add, evaluate=evaluate)

tree = make_initial_tree(env)
for i in range(10000):
  run_mcts_iteration(tree, env)
  if i % 1000 == 0:
    # Print best move sequence found so far
    node = tree
    while node.children:
      node = max(node.children, key=lambda child: child.visit_count)
    print(i, node.state)

print('Visit counts for children of the root node:')
print([(child.state, child.visit_count) for child in tree.children])

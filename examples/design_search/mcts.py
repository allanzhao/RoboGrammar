from collections import namedtuple
from math import log, sqrt
import random

class TreeNode(object):
  def __init__(self, state, parent, children):
    self.state = state
    self.visit_count = 0
    self.result_sum = 0
    self.sq_result_sum = 0
    self.parent = parent
    self.children = children

Env = namedtuple(
    "Env", "initial_state get_available_moves get_next_state evaluate")

def uct_score(node, ucb_weight=sqrt(2.0)):
  if node.visit_count == 0:
    return float('inf')
  else:
    return (node.result_sum / node.visit_count +
            ucb_weight * sqrt(log(node.parent.visit_count) / node.visit_count))

def select_node(tree, score_fn):
  node = tree
  while node.children:
    node = max(node.children, key=score_fn)
  return node

def expand_node(node, moves, get_next_state, score_fn):
  assert not node.children
  node.children = [TreeNode(state=get_next_state(node.state, move),
                            parent=node, children=[]) for move in moves]
  return max(node.children, key=score_fn)

def simulate(state, env):
  available_moves = env.get_available_moves(state)
  while available_moves:
    state = env.get_next_state(state, random.choice(available_moves))
    available_moves = env.get_available_moves(state)
  return env.evaluate(state)

def backpropagate_node(node, result):
  sq_result = result * result
  while node:
    node.visit_count += 1
    node.result_sum += result
    node.sq_result_sum += sq_result
    node = node.parent

def make_initial_tree(env):
  return TreeNode(state=env.initial_state, parent=None, children=[])

def run_mcts_iteration(tree, env, score_fn=uct_score):
  node = select_node(tree, score_fn)
  available_moves = env.get_available_moves(node.state)
  if available_moves:
    node = expand_node(node, available_moves, env.get_next_state, score_fn)
  result = simulate(node.state, env)
  backpropagate_node(node, result)

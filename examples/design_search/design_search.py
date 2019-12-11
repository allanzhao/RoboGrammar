import json
from mcts import Env, TreeNode, make_initial_tree, run_mcts_iteration
import numpy as np
import os
import pyrobotdesign as rd
import random

time_step = 1.0 / 240
discount_factor = 0.99
interval = 4
horizon = 64
thread_count = 16
episode_len = 250

graphs = rd.load_graphs('data/designs/grammar7.dot')
rules = [rd.create_rule_from_graph(g) for g in graphs]

os.makedirs('trees')
save_attr_names = ['visit_count', 'result_sum', 'sq_result_sum', 'children']
class DictEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, TreeNode):
      save_dict = dict()
      for attr_name in save_attr_names:
        save_dict[attr_name] = getattr(obj, attr_name)
      save_dict['rule_seq'] = [rules.index(rule) for rule in obj.state[1]]
      return save_dict
    else:
      return None

n0 = rd.Node()
n0.name = 'robot'
n0.attrs.label = 'robot'
initial_robot_graph = rd.Graph()
initial_robot_graph.nodes = [n0]

def get_applicable_matches(rule, robot_graph):
  matches = rd.find_matches(rule.lhs, robot_graph)
  applicable_matches = []
  for match in matches:
    if rd.check_rule_applicability(rule, robot_graph, match):
      applicable_matches.append(match)
  return applicable_matches

def get_available_moves(state):
  robot_graph, rule_seq = state
  applicable_rules = []
  for rule in rules:
    if get_applicable_matches(rule, robot_graph):
      applicable_rules.append(rule)
  return applicable_rules

def get_next_state(state, rule):
  robot_graph, rule_seq = state
  applicable_matches = get_applicable_matches(rule, robot_graph)
  return (rd.apply_rule(rule, robot_graph, applicable_matches[0]),
          rule_seq + [rule])

def evaluate(state):
  robot_graph, rule_seq = state

  print("Evaluating:", [rules.index(rule) for rule in rule_seq])

  robot = rd.build_robot(robot_graph)

  floor = rd.Prop(0.0, 0.9, [10.0, 1.0, 10.0])

  # Find an initial y offset that will place the robot precisely on the ground
  def find_y_offset(robot):
    temp_sim = rd.BulletSimulation(time_step)
    temp_sim.add_robot(robot, np.zeros(3), rd.Quaterniond(1.0, 0.0, 0.0, 0.0))
    lower = np.zeros(3)
    upper = np.zeros(3)
    temp_sim.get_robot_world_aabb(temp_sim.find_robot_index(robot), lower,
                                  upper)
    return -lower[1]

  y_offset = find_y_offset(robot)

  def make_sim_fn():
    sim = rd.BulletSimulation(time_step)
    sim.add_prop(floor, [0.0, -1.0, 0.0], rd.Quaterniond(1.0, 0.0, 0.0, 0.0))
    sim.add_robot(robot, [0.0, y_offset, 0.0],
                  rd.Quaterniond(1.0, 0.0, 0.0, 0.0))
    return sim

  main_sim = make_sim_fn()
  robot_idx = main_sim.find_robot_index(robot)

  dof_count = main_sim.get_robot_dof_count(robot_idx)
  value_estimator = rd.FCValueEstimator(main_sim, robot_idx, 'cpu', 64, 3, 6)
  objective_fn = rd.SumOfSquaresObjective()
  objective_fn.base_vel_ref = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
  objective_fn.base_vel_weight = np.full(6, 1.0)
  objective_fn.power_weight = 0.0001
  opt_seed = random.getrandbits(32)
  print("Optimization seed:", opt_seed)
  optimizer = rd.MPPIOptimizer(100.0, discount_factor, dof_count, interval,
                               horizon, 128, thread_count, opt_seed,
                               make_sim_fn, objective_fn, value_estimator)
  for _ in range(10):
    optimizer.update()

  main_sim.save_state()

  input_sequence = np.zeros((dof_count, episode_len))
  obs = np.zeros((value_estimator.get_observation_size(), episode_len + 1),
                 order='f')
  rewards = np.zeros(episode_len)
  for j in range(episode_len):
    optimizer.update()
    input_sequence[:,j] = optimizer.input_sequence[:,0]
    optimizer.advance(1)

    value_estimator.get_observation(main_sim, obs[:,j])
    rewards[j] = 0.0;
    for i in range(interval):
      main_sim.set_joint_target_positions(robot_idx, input_sequence[:,j])
      main_sim.step()
      rewards[j] += objective_fn(main_sim)
  value_estimator.get_observation(main_sim, obs[:,-1])

  result = rewards.sum() / episode_len
  print("Result:", result)
  return result

robot_design_env = Env(initial_state=(initial_robot_graph, []),
                       get_available_moves=get_available_moves,
                       get_next_state=get_next_state,
                       evaluate=evaluate)

tree = make_initial_tree(robot_design_env)
for i in range(250):
  print("Iteration:", i)

  run_mcts_iteration(tree, robot_design_env)

  # Print the best rule sequence found so far
  node = tree
  while node.children:
    node = max(node.children, key=lambda child: child.visit_count)
  print("Best branch:", i, [rules.index(rule) for rule in node.state[1]])

  # Save the search tree
  with open('trees/tree_{:04d}.json'.format(i), 'w') as tree_file:
    tree_file.write(DictEncoder().encode(tree))

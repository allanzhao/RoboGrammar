import argparse
import contextlib
import csv
import datetime
import mcts
import numpy as np
import os
import pyrobotdesign as rd
import random

def get_applicable_matches(rule, graph):
  """Generates all applicable matches for rule in graph."""
  for match in rd.find_matches(rule.lhs, graph):
    if rd.check_rule_applicability(rule, graph, match):
      yield match

def find_y_offset(robot):
  """Finds an initial y offset that will place the robot on the ground."""
  temp_sim = rd.BulletSimulation()
  temp_sim.add_robot(robot, np.zeros(3), rd.Quaterniond(1.0, 0.0, 0.0, 0.0))
  lower = np.zeros(3)
  upper = np.zeros(3)
  temp_sim.get_robot_world_aabb(temp_sim.find_robot_index(robot), lower, upper)
  return -lower[1]

class RobotDesignEnv(mcts.Env):
  """Robot design environment where states are (graph, rule sequence) pairs and
  actions are rule applications."""

  def __init__(self, rules, seed, time_step=1.0/240, discount_factor=0.99,
               interval=4, horizon=64, thread_count=16, episode_len=250):
    self.rules = rules
    self.rng = random.Random(seed)
    self.time_step = time_step
    self.discount_factor = discount_factor
    self.interval = interval
    self.horizon = horizon
    self.thread_count = thread_count
    self.episode_len = episode_len

    # Create initial robot graph
    n0 = rd.Node()
    n0.name = 'robot'
    n0.attrs.label = 'robot'
    self.initial_graph = rd.Graph()
    self.initial_graph.nodes = [n0]

  @property
  def initial_state(self):
    return (self.initial_graph, [])

  def get_available_actions(self, state):
    graph, rule_seq = state
    for rule in self.rules:
      if list(get_applicable_matches(rule, graph)):
        # Rule has at least one applicable match
        yield rule

  def get_next_state(self, state, rule):
    graph, rule_seq = state
    applicable_matches = list(get_applicable_matches(rule, graph))
    return (rd.apply_rule(rule, graph, applicable_matches[0]),
            rule_seq + [rule])

  def get_result(self, state):
    graph, rule_seq = state

    robot = rd.build_robot(graph)
    floor = rd.Prop(0.0, 0.9, [10.0, 1.0, 10.0])
    y_offset = find_y_offset(robot)

    def make_sim_fn():
      sim = rd.BulletSimulation(self.time_step)
      sim.add_prop(floor, [0.0, -1.0, 0.0], rd.Quaterniond(1.0, 0.0, 0.0, 0.0))
      # Rotate 180 degrees around the y axis, so the base points to the right
      sim.add_robot(robot, [0.0, y_offset, 0.0],
                    rd.Quaterniond(0.0, 0.0, 1.0, 0.0))
      return sim

    main_sim = make_sim_fn()
    robot_idx = main_sim.find_robot_index(robot)

    dof_count = main_sim.get_robot_dof_count(robot_idx)
    value_estimator = rd.FCValueEstimator(main_sim, robot_idx, 'cpu', 64, 3, 6)
    objective_fn = rd.SumOfSquaresObjective()
    objective_fn.base_vel_ref = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    objective_fn.base_vel_weight = np.array([0.0, 0.0, 0.0, 100.0, 0.0, 0.0])
    objective_fn.power_weight = 0.0 # Ignore power consumption
    opt_seed = self.rng.getrandbits(32)
    optimizer = rd.MPPIOptimizer(100.0, self.discount_factor, dof_count,
                                 self.interval, self.horizon, 128,
                                 self.thread_count, opt_seed,
                                 make_sim_fn, objective_fn, value_estimator)
    for _ in range(10):
      optimizer.update()

    main_sim.save_state()

    input_sequence = np.zeros((dof_count, self.episode_len))
    obs = np.zeros(
        (value_estimator.get_observation_size(), self.episode_len + 1),
        order='f')
    rewards = np.zeros(self.episode_len)
    for j in range(self.episode_len):
      optimizer.update()
      input_sequence[:,j] = optimizer.input_sequence[:,0]
      optimizer.advance(1)

      value_estimator.get_observation(main_sim, obs[:,j])
      rewards[j] = 0.0;
      for i in range(self.interval):
        main_sim.set_joint_target_positions(robot_idx, input_sequence[:,j])
        main_sim.step()
        rewards[j] += objective_fn(main_sim)
    value_estimator.get_observation(main_sim, obs[:,-1])

    return rewards.sum() / self.episode_len

  def get_key(self, state):
    return hash(state[0])

def main():
  parser = argparse.ArgumentParser(description="Robot design search demo.")
  parser.add_argument("grammar_file", type=str, help="Grammar file (.dot)")
  parser.add_argument("-s", "--seed", type=int, default=None,
                      help="Random seed")
  parser.add_argument("-j", "--jobs", type=int, required=True,
                      help="Number of jobs/threads")
  parser.add_argument("-i", "--iterations", type=int, required=True,
                      help="Number of MCTS iterations")
  parser.add_argument("-l", "--log_dir", type=str, default='',
                      help="Log directory")
  args = parser.parse_args()

  graphs = rd.load_graphs(args.grammar_file)
  rules = [rd.create_rule_from_graph(g) for g in graphs]
  env = RobotDesignEnv(rules, args.seed)
  tree_search = mcts.TreeSearch(env)

  os.makedirs(args.log_dir, exist_ok=True)

  log_path = os.path.join(args.log_dir,
                          f'mcts_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv')
  print(f"Logging to '{log_path}'")

  with open(log_path, 'a', newline='') as log_file:
    fieldnames = ['iteration', 'rule_seq', 'result']
    writer = csv.DictWriter(log_file, fieldnames=fieldnames)
    writer.writeheader()
    log_file.flush()

    for i in range(args.iterations):
      states, actions, result = tree_search.run_iteration()

      # Last action is always None
      rule_seq = [rules.index(rule) for rule in actions[:-1]]
      writer.writerow({'iteration': i, 'rule_seq': rule_seq, 'result': result})
      log_file.flush()

if __name__ == '__main__':
  main()

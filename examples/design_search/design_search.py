import argparse
import csv
import datetime
import mcts
import numpy as np
import os
import pyrobotdesign as rd
import random
import tasks

def get_applicable_matches(rule, graph):
  """Generates all applicable matches for rule in graph."""
  for match in rd.find_matches(rule.lhs, graph):
    if rd.check_rule_applicability(rule, graph, match):
      yield match

def presimulate(robot):
  """Find an initial y offset that will place the robot on the ground, and check
  if the robot collides in its initial configuration."""
  temp_sim = rd.BulletSimulation()
  temp_sim.add_robot(robot, np.zeros(3), rd.Quaterniond(1.0, 0.0, 0.0, 0.0))
  robot_idx = temp_sim.find_robot_index(robot)
  lower = np.zeros(3)
  upper = np.zeros(3)
  temp_sim.get_robot_world_aabb(robot_idx, lower, upper)
  temp_sim.step() # Needed to find collisions
  return -lower[1], temp_sim.robot_has_collision(robot_idx)

class RobotDesignEnv(mcts.Env):
  """Robot design environment where states are (graph, rule sequence) pairs and
  actions are rule applications."""

  def __init__(self, task, rules, seed, time_step=1.0/240, discount_factor=0.99,
               interval=4, horizon=64, thread_count=16, episode_len=250):
    self.task = task
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
    y_offset, has_self_collision = presimulate(robot)

    if has_self_collision:
      self.latest_opt_seed = 0
      return 0.0

    def make_sim_fn():
      sim = rd.BulletSimulation(self.time_step)
      self.task.add_terrain(sim)
      # Rotate 180 degrees around the y axis, so the base points to the right
      sim.add_robot(robot, [0.0, y_offset, 0.0],
                    rd.Quaterniond(0.0, 0.0, 1.0, 0.0))
      return sim

    main_sim = make_sim_fn()
    robot_idx = main_sim.find_robot_index(robot)

    dof_count = main_sim.get_robot_dof_count(robot_idx)
    value_estimator = rd.NullValueEstimator()
    objective_fn = self.task.get_objective_fn()
    opt_seed = self.rng.getrandbits(32)
    self.latest_opt_seed = opt_seed
    optimizer = rd.MPPIOptimizer(100.0, self.discount_factor, dof_count,
                                 self.interval, self.horizon, 128,
                                 self.thread_count, opt_seed,
                                 make_sim_fn, objective_fn, value_estimator)
    for _ in range(10):
      optimizer.update()

    main_sim.save_state()

    input_sequence = np.zeros((dof_count, self.episode_len))
    rewards = np.zeros(self.episode_len * self.interval)
    for j in range(self.episode_len):
      optimizer.update()
      input_sequence[:,j] = optimizer.input_sequence[:,0]
      optimizer.advance(1)

      for k in range(self.interval):
        main_sim.set_joint_target_positions(robot_idx, input_sequence[:,j])
        main_sim.step()
        rewards[j * self.interval + k] = objective_fn(main_sim)

    # Apply normalization
    # Result should be zero for a stationary robot and one for perfect tracking
    base_vel_ref = objective_fn.base_vel_ref
    base_vel_weight = objective_fn.base_vel_weight
    reward_scale = base_vel_ref.dot(np.diag(base_vel_weight)).dot(base_vel_ref)
    return np.mean(rewards) / reward_scale + 1.0

  def get_key(self, state):
    return hash(state[0])

def main():
  parser = argparse.ArgumentParser(description="Robot design search demo.")
  parser.add_argument("task", type=str, help="Task (Python class name)")
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

  random.seed(args.seed)

  task_class = getattr(tasks, args.task)
  task = task_class()
  graphs = rd.load_graphs(args.grammar_file)
  rules = [rd.create_rule_from_graph(g) for g in graphs]
  env = RobotDesignEnv(task, rules, args.seed)
  tree_search = mcts.TreeSearch(env)

  os.makedirs(args.log_dir, exist_ok=True)

  log_path = os.path.join(args.log_dir,
                          f'mcts_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv')
  print(f"Logging to '{log_path}'")

  with open(log_path, 'a', newline='') as log_file:
    fieldnames = ['iteration', 'rule_seq', 'opt_seed', 'result']
    writer = csv.DictWriter(log_file, fieldnames=fieldnames)
    writer.writeheader()
    log_file.flush()

    for i in range(args.iterations):
      states, actions, result = tree_search.run_iteration()

      # Last action is always None
      rule_seq = [rules.index(rule) for rule in actions[:-1]]
      writer.writerow({'iteration': i, 'rule_seq': rule_seq,
                       'opt_seed': env.latest_opt_seed, 'result': result})
      log_file.flush()

if __name__ == '__main__':
  main()

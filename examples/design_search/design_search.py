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

def has_nonterminals(graph):
  """Returns True if the graph contains nonterminal nodes/edges, and False
  otherwise."""
  for node in graph.nodes:
    if node.attrs.shape == rd.LinkShape.NONE:
      return True
  for edge in graph.edges:
    if edge.attrs.joint_type == rd.JointType.NONE:
      return True
  return False

def presimulate(robot):
  """Find an initial position that will place the robot on the ground behind the
  x=0 plane, and check if the robot collides in its initial configuration."""
  temp_sim = rd.BulletSimulation()
  temp_sim.add_robot(robot, np.zeros(3), rd.Quaterniond(0.0, 0.0, 1.0, 0.0))
  temp_sim.step()
  robot_idx = temp_sim.find_robot_index(robot)
  lower = np.zeros(3)
  upper = np.zeros(3)
  temp_sim.get_robot_world_aabb(robot_idx, lower, upper)
  return [-upper[0], -lower[1], 0.0], temp_sim.robot_has_collision(robot_idx)

def simulate(robot, task, opt_seed, thread_count, episode_count=1):
  """Run trajectory optimization for the robot on the given task, and return the
  resulting input sequence and result."""
  robot_init_pos, has_self_collision = presimulate(robot)

  if has_self_collision:
    return None, None

  def make_sim_fn():
    sim = rd.BulletSimulation(task.time_step)
    task.add_terrain(sim)
    # Rotate 180 degrees around the y axis, so the base points to the right
    sim.add_robot(robot, robot_init_pos, rd.Quaterniond(0.0, 0.0, 1.0, 0.0))
    return sim

  main_sim = make_sim_fn()
  robot_idx = main_sim.find_robot_index(robot)

  dof_count = main_sim.get_robot_dof_count(robot_idx)
  if episode_count >= 2:
    value_estimator = rd.FCValueEstimator(main_sim, robot_idx, 'cpu', 64, 3, 1)
  else:
    value_estimator = rd.NullValueEstimator()
  objective_fn = task.get_objective_fn()

  replay_obs = np.zeros((value_estimator.get_observation_size(), 0))
  replay_returns = np.zeros(0)

  for episode_idx in range(episode_count):
    optimizer = rd.MPPIOptimizer(1.0, task.discount_factor, dof_count,
                                 task.interval, task.horizon, 128, thread_count,
                                 opt_seed + episode_idx, make_sim_fn,
                                 objective_fn, value_estimator)
    for _ in range(10):
      optimizer.update()

    main_sim.save_state()

    input_sequence = np.zeros((dof_count, task.episode_len))
    obs = np.zeros((value_estimator.get_observation_size(), task.episode_len + 1),
                   order='f')
    rewards = np.zeros(task.episode_len * task.interval)
    for j in range(task.episode_len):
      optimizer.update()
      input_sequence[:,j] = optimizer.input_sequence[:,0]
      optimizer.advance(1)

      value_estimator.get_observation(main_sim, obs[:,j])
      for k in range(task.interval):
        main_sim.set_joint_target_positions(robot_idx,
                                            input_sequence[:,j].reshape(-1, 1))
        main_sim.step()
        rewards[j * task.interval + k] = objective_fn(main_sim)
    value_estimator.get_observation(main_sim, obs[:,-1])

    main_sim.restore_state()

    # Only train the value estimator if there will be another episode
    if episode_idx < episode_count - 1:
      returns = np.zeros(task.episode_len + 1)
      # Bootstrap returns with value estimator
      value_estimator.estimate_value(obs[:,task.episode_len], returns[-1:])
      for j in reversed(range(task.episode_len)):
        interval_reward = np.sum(
            rewards[j * task.interval:(j + 1) * task.interval])
        returns[j] = interval_reward + task.discount_factor * returns[j + 1]
      replay_obs = np.hstack((replay_obs, obs[:,:task.episode_len]))
      replay_returns = np.concatenate((replay_returns,
                                       returns[:task.episode_len]))
      value_estimator.train(replay_obs, replay_returns)

  return input_sequence, np.mean(rewards)

def make_initial_graph():
  """Make an initial robot graph."""
  n0 = rd.Node()
  n0.name = 'robot'
  n0.attrs.label = 'robot'
  initial_graph = rd.Graph()
  initial_graph.nodes = [n0]
  return initial_graph

class RobotDesignEnv(mcts.Env):
  """Robot design environment where states are (graph, rule sequence) pairs and
  actions are rule applications."""

  def __init__(self, task, rules, seed, thread_count):
    self.task = task
    self.rules = rules
    self.rng = random.Random(seed)
    self.thread_count = thread_count
    self.initial_graph = make_initial_graph()

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
    if has_nonterminals(graph):
      # Graph is incomplete
      return None
    robot = rd.build_robot(graph)
    opt_seed = self.rng.getrandbits(32)
    self.latest_opt_seed = opt_seed
    input_sequence, result = simulate(robot, self.task, opt_seed,
                                      self.thread_count)

    # FIXME: workaround for simulation instability
    # Simulation is invalid if the result is greater than result_bound
    if result is not None and result > self.task.result_bound:
      return None

    return result

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
  env = RobotDesignEnv(task, rules, args.seed, args.jobs)
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

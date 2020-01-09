import argparse
from design_search import RobotDesignEnv, make_initial_graph, presimulate, simulate
import mcts
import numpy as np
import pyrobotdesign as rd
import random
import tasks
import time

def view_trajectory(sim, robot_idx, input_sequence, time_step, interval):
  sim.save_state()
  viewer = rd.GLFWViewer()
  sim_time = time.time()
  i = 0
  j = 0
  while not viewer.should_close():
    current_time = time.time()
    while sim_time < current_time:
      if input_sequence is not None:
        sim.set_joint_target_positions(robot_idx, input_sequence[:,j])
      sim.step()
      viewer.update(time_step)
      sim_time += time_step
      i += 1
      if i >= interval:
        i = 0
        j += 1
      if input_sequence is not None and j >= input_sequence.shape[1]:
        i = 0
        j = 0
        sim.restore_state()
    viewer.render(sim)

def main():
  parser = argparse.ArgumentParser(description="Robot design viewer.")
  parser.add_argument("task", type=str, help="Task (Python class name)")
  parser.add_argument("grammar_file", type=str, help="Grammar file (.dot)")
  parser.add_argument("rule_sequence", nargs="+", help="Rule sequence to apply")
  parser.add_argument("-o", "--optim", default=False, action="store_true",
                      help="Optimize a trajectory")
  parser.add_argument("-s", "--opt_seed", type=int, default=None,
                      help="Trajectory optimization seed")
  parser.add_argument("-j", "--jobs", type=int, required=True,
                      help="Number of jobs/threads")
  args = parser.parse_args()

  task_class = getattr(tasks, args.task)
  task = task_class()
  graphs = rd.load_graphs(args.grammar_file)
  rules = [rd.create_rule_from_graph(g) for g in graphs]

  rule_sequence = [int(s) for s in args.rule_sequence]
  if args.opt_seed is not None:
    opt_seed = args.opt_seed
  else:
    opt_seed = random.getrandbits(32)

  graph = make_initial_graph()
  for r in rule_sequence:
    matches = rd.find_matches(rules[r].lhs, graph)
    if matches:
      graph = rd.apply_rule(rules[r], graph, matches[0])

  robot = rd.build_robot(graph)
  if args.optim:
    input_sequence, result = simulate(robot, task, opt_seed, args.jobs)
    print("Result:", result)
  else:
    input_sequence = None

  robot_init_pos, has_self_collision = presimulate(robot)

  if has_self_collision:
    print("Warning: robot self-collides in initial configuration")

  main_sim = rd.BulletSimulation(task.time_step)
  task.add_terrain(main_sim)
  # Rotate 180 degrees around the y axis, so the base points to the right
  main_sim.add_robot(robot, robot_init_pos, rd.Quaterniond(0.0, 0.0, 1.0, 0.0))
  robot_idx = main_sim.find_robot_index(robot)

  view_trajectory(main_sim, robot_idx, input_sequence, task.time_step,
                  task.interval)

if __name__ == '__main__':
  main()

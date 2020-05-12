import argparse
from design_search import RobotDesignEnv, make_graph, build_normalized_robot, presimulate, simulate
import mcts
import numpy as np
import pyrobotdesign as rd
import random
import tasks
import time

def view_trajectory(sim, robot_idx, input_sequence, task):
  sim.save_state()

  # Get robot bounds
  lower = np.zeros(3)
  upper = np.zeros(3)
  sim.get_robot_world_aabb(robot_idx, lower, upper)

  viewer = rd.GLFWViewer()
  viewer.camera_params.position = 0.5 * (lower + upper)
  viewer.camera_params.yaw = 0.0
  viewer.camera_params.pitch = -np.pi / 6
  viewer.camera_params.distance = 2.0 * np.linalg.norm(upper - lower)

  j = 0
  k = 0
  sim_time = time.time()
  while not viewer.should_close():
    current_time = time.time()
    while sim_time < current_time:
      if input_sequence is not None:
        sim.set_joint_target_positions(robot_idx,
                                       input_sequence[:,j].reshape(-1, 1))
      task.add_noise(sim, j * task.interval + k)
      sim.step()
      sim.get_robot_world_aabb(robot_idx, lower, upper)
      # Update camera position to track the robot smoothly
      target_pos = 0.5 * (lower + upper)
      camera_pos = viewer.camera_params.position.copy()
      camera_pos += 5.0 * task.time_step * (target_pos - camera_pos)
      viewer.camera_params.position = camera_pos
      viewer.update(task.time_step)
      sim_time += task.time_step
      k += 1
      if k >= task.interval:
        j += 1
        k = 0
      if input_sequence is not None and j >= input_sequence.shape[1]:
        j = 0
        k = 0
        sim.restore_state()
        sim.get_robot_world_aabb(robot_idx, lower, upper)
        viewer.camera_params.position = 0.5 * (lower + upper)
    viewer.render(sim)

def finalize_robot(robot):
  for link in robot.links:
    link.label = ""
    link.joint_label = ""
    if link.shape == rd.LinkShape.NONE:
      link.shape = rd.LinkShape.CAPSULE
      link.length = 0.1
      link.radius = 0.025
      link.color = [1.0, 0.0, 1.0]
    if link.joint_type == rd.JointType.NONE:
      link.joint_type = rd.JointType.FIXED
      link.joint_color = [1.0, 0.0, 1.0]

def main():
  parser = argparse.ArgumentParser(description="Robot design viewer.")
  parser.add_argument("task", type=str, help="Task (Python class name)")
  parser.add_argument("grammar_file", type=str, help="Grammar file (.dot)")
  parser.add_argument("rule_sequence", nargs="+", help="Rule sequence to apply")
  parser.add_argument("-o", "--optim", default=False, action="store_true",
                      help="Optimize a trajectory")
  parser.add_argument("-s", "--opt_seed", type=int, default=None,
                      help="Trajectory optimization seed")
  parser.add_argument("-e", "--episodes", type=int, default=1,
                      help="Number of optimization episodes")
  parser.add_argument("-j", "--jobs", type=int, required=True,
                      help="Number of jobs/threads")
  parser.add_argument("--input_sequence_file", type=str,
                      help="File to save input sequence to (.csv)")
  parser.add_argument("-l", "--episode_len", type=int, default=128, help="lenth of episode")
  parser.add_argument("--no-noise", default=False, action='store_true')
  args = parser.parse_args()

  task_class = getattr(tasks, args.task)
  if args.no_noise:
    task = task_class(force_std = 0.0, episode_len = args.episode_len)
  else:
    task = task_class(episode_len = args.episode_len)
  
  graphs = rd.load_graphs(args.grammar_file)
  rules = [rd.create_rule_from_graph(g) for g in graphs]

  rule_sequence = [int(s.strip(",")) for s in args.rule_sequence]
  if args.opt_seed is not None:
    opt_seed = args.opt_seed
  else:
    opt_seed = random.getrandbits(32)
    print("Using optimization seed:", opt_seed)

  graph = make_graph(rules, rule_sequence)
  robot = build_normalized_robot(graph)
  finalize_robot(robot)
  if args.optim:
    input_sequence, result = simulate(robot, task, opt_seed, args.jobs,
                                      args.episodes)
    print("Result:", result)
  else:
    input_sequence = None

  if args.input_sequence_file and input_sequence is not None:
    import csv
    with open(args.input_sequence_file, 'w', newline='') as input_seq_file:
      writer = csv.writer(input_seq_file)
      for col in input_sequence.T:
        writer.writerow(col)
    print("Saved input sequence to file:", args.input_sequence_file)

  robot_init_pos, has_self_collision = presimulate(robot)

  if has_self_collision:
    print("Warning: robot self-collides in initial configuration")

  main_sim = rd.BulletSimulation(task.time_step)
  task.add_terrain(main_sim)
  # Rotate 180 degrees around the y axis, so the base points to the right
  main_sim.add_robot(robot, robot_init_pos, rd.Quaterniond(0.0, 0.0, 1.0, 0.0))
  robot_idx = main_sim.find_robot_index(robot)

  view_trajectory(main_sim, robot_idx, input_sequence, task)

if __name__ == '__main__':
  main()

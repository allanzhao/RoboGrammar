import sys
import os
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'graph_learning'))
sys.path.append(os.path.join(base_dir, 'design_search'))

from PIL import Image
import argparse
import ast
from design_search import *
import mcts
import numpy as np
import pandas as pd
import pyrobotdesign as rd
import tasks
from viewer import *

def make_robot_from_rule_sequence(rule_sequence, rules):
  graph = make_initial_graph()
  for r in rule_sequence:
    matches = rd.find_matches(rules[r].lhs, graph)
    if matches:
      graph = rd.apply_rule(rules[r], graph, matches[0])

  return build_normalized_robot(graph)

def make_robot_from_rule_sequence_raw(rule_sequence, rules):
  graph = make_initial_graph()
  for r in rule_sequence:
    matches = rd.find_matches(rules[r].lhs, graph)
    if matches:
      graph = rd.apply_rule(rules[r], graph, matches[0])

  return graph

def get_robot_image(robot, task):
  sim = rd.BulletSimulation(task.time_step)
  task.add_terrain(sim)
  viewer = rd.GLFWViewer()
  if robot is not None:
    robot_init_pos, _ = presimulate(robot)
    # Rotate 180 degrees around the y axis, so the base points to the right
    sim.add_robot(robot, robot_init_pos, rd.Quaterniond(0.0, 0.0, 1.0, 0.0))
    robot_idx = sim.find_robot_index(robot)

    # Get robot position and bounds
    base_tf = np.zeros((4, 4), order='f')
    lower = np.zeros(3)
    upper = np.zeros(3)
    sim.get_link_transform(robot_idx, 0, base_tf)
    sim.get_robot_world_aabb(robot_idx, lower, upper)
    viewer.camera_params.position = base_tf[:3,3]
    viewer.camera_params.yaw = np.pi / 3
    viewer.camera_params.pitch = -np.pi / 6
    viewer.camera_params.distance = 2.0 * np.linalg.norm(upper - lower)
  else:
    viewer.camera_params.position = [1.0, 0.0, 0.0]
    viewer.camera_params.yaw = -np.pi / 3
    viewer.camera_params.pitch = -np.pi / 6
    viewer.camera_params.distance = 5.0

  viewer.update(task.time_step)
  
  return viewer.render_array(sim)

def save_robot_video_frames(robot, task, opt_seed, thread_count, save_image_dir,
                            frame_interval):
  input_sequence, _ = simulate(robot, task, opt_seed, thread_count)

  sim = rd.BulletSimulation(task.time_step)
  task.add_terrain(sim)
  viewer = rd.GLFWViewer()
  robot_init_pos, _ = presimulate(robot)
  # Rotate 180 degrees around the y axis, so the base points to the right
  sim.add_robot(robot, robot_init_pos, rd.Quaterniond(0.0, 0.0, 1.0, 0.0))
  robot_idx = sim.find_robot_index(robot)

  # Get robot position and bounds
  base_tf = np.zeros((4, 4), order='f')
  lower = np.zeros(3)
  upper = np.zeros(3)
  sim.get_link_transform(robot_idx, 0, base_tf)
  sim.get_robot_world_aabb(robot_idx, lower, upper)
  viewer.camera_params.position = base_tf[:3,3]
  viewer.camera_params.yaw = np.pi / 12
  viewer.camera_params.pitch = -np.pi / 12
  #viewer.camera_params.yaw = np.pi / 3
  #viewer.camera_params.pitch = -np.pi / 6
  viewer.camera_params.distance = 2.0 * np.linalg.norm(upper - lower)

  frame_index = 0
  for j in range(input_sequence.shape[1]):
    for k in range(task.interval):
      sim.set_joint_target_positions(robot_idx,
                                     input_sequence[:,j].reshape(-1, 1))
      sim.step()
      sim.get_link_transform(robot_idx, 0, base_tf)
      # Update camera position to track the robot smoothly
      camera_pos = viewer.camera_params.position.copy()
      camera_pos += 4.0 * task.time_step * (base_tf[:3,3] - camera_pos)
      viewer.camera_params.position = camera_pos
      viewer.update(task.time_step)
      if frame_index % frame_interval == 0:
        viewer.render(sim)
        im_data = viewer.get_image()[::-1,:,:]
        im = Image.fromarray(im_data)
        im.save(os.path.join(save_image_dir,
                             f"frame_{frame_index // frame_interval:04}.png"))
      frame_index += 1

def main():
  parser = argparse.ArgumentParser(description="Process robot design search results.")
  parser.add_argument("task", type=str, help="Task (Python class name)")
  parser.add_argument("-j", "--jobs", type=int, required=True,
                      help="Number of jobs/threads")
  parser.add_argument("--grammar-file", type=str, default = './data/designs/grammar_apr30.dot', help="Grammar file (.dot)")
  parser.add_argument("-f", "--log_file", type=str, required=True,
                      help="MCTS log file")
  parser.add_argument("-t", "--type", type=str)
  parser.add_argument("-d", "--save_image_dir", default = None, type=str)
  parser.add_argument("-i", "--iterations", type=int, nargs="+")
  parser.add_argument("-s", "--opt_seed", type=int)
  args = parser.parse_args()
  
  args.save_image_dir = os.path.join(os.path.dirname(args.log_file), args.type)
  os.makedirs(args.save_image_dir, exist_ok=True)

  os.system('cp {} {}/designs.csv'.format(args.log_file, args.save_image_dir))

  task_class = getattr(tasks, args.task)
  task = task_class()
  graphs = rd.load_graphs(args.grammar_file)
  rules = [rd.create_rule_from_graph(g) for g in graphs]
  
  iteration_df = pd.read_csv(args.log_file, index_col=0)
  iteration_df['iteration'] = list(range(1, len(iteration_df) + 1))
  
  if args.type == "iterations":
    os.makedirs(args.save_image_dir, exist_ok=True)
    mid_indices = np.arange(0, len(iteration_df) + 1, 1000)
    offset = 10
    for mid_index in mid_indices:
      start_index = max(mid_index - offset, 0)
      end_index = min(mid_index + offset, len(iteration_df))
      for index in range(start_index, end_index):
        rule_seq = ast.literal_eval(iteration_df['rule_seq'][index])
        robot = make_robot_from_rule_sequence(rule_seq, rules)
        im_data = get_robot_image(robot, task)[::-1,:,:]
        im = Image.fromarray(im_data)
        im.save(os.path.join(args.save_image_dir,
                             f"iteration_{index:05}.png"))

  if args.type == "iterations_top":
    block_size = 100
    count = 2
    total = ((len(iteration_df) - 1) // block_size + 1) * count
    for start_index in range(0, len(iteration_df), block_size):
    #   sys.stdout.write('\r Finish {}/{}'.format(start_index // block_size * count, total))
    #   sys.stdout.flush()

      end_index = min(start_index + block_size, len(iteration_df))
      block = iteration_df[start_index:end_index].copy()
      block = block.sort_values(by='reward', ascending=False).reset_index()
      for i in range(count):
        row = block.iloc[i]
        rule_seq = ast.literal_eval(row['rule_seq'])
        robot = make_robot_from_rule_sequence(rule_seq, rules)
        print('iteration = {}, reward = {}'.format(row['iteration'], row['reward']))
        im_data = get_robot_image(robot, task)[::-1,:,:]
        im = Image.fromarray(im_data)
        block_index = start_index // block_size
        im.save(os.path.join(args.save_image_dir,
                             f"iteration_{row['iteration']:05}_{row['reward']:.2f}.png"))

  if args.type == "all_top": # save all top K unique designs
    count = 50
    sorted_df = iteration_df.sort_values(by='reward', ascending=False).reset_index()
    hash_values = set()
    j = -1
    for i in range(count):
      while True:
        j = j + 1
        row = sorted_df.iloc[j]
        rule_seq = ast.literal_eval(row['rule_seq'])
        robot = make_robot_from_rule_sequence(rule_seq, rules)
        robot_raw = make_robot_from_rule_sequence_raw(rule_seq, rules)
        hash_key = hash(robot_raw)
        if hash_key in hash_values:
          continue
        hash_values.add(hash_key)
        print('iteration = {}, reward = {}, hash = {}'.format(row['iteration'], row['reward'], hash_key))
        im_data = get_robot_image(robot, task)[::-1,:,:]
        im = Image.fromarray(im_data)
        im.save(os.path.join(args.save_image_dir,
                                f"rank_{i+1:03}_iteration_{row['iteration']:05}_{row['reward']:.2f}.png"))
        break

  elif args.type == "percentiles":
    os.makedirs(args.save_image_dir, exist_ok=True)
    percentiles = np.linspace(0.0, 1.0, 11)
    offset = 10
    iteration_df.sort_values(by='reward')
    for percentile in percentiles:
      mid_index = int(round(percentile * (len(iteration_df) - 1)))
      start_index = max(mid_index - offset, 0)
      end_index = min(mid_index + offset, len(iteration_df))
      for index in range(start_index, end_index):
        rule_seq = ast.literal_eval(iteration_df['rule_seq'][index])
        robot = make_robot_from_rule_sequence(rule_seq, rules)
        im_data = get_robot_image(robot, task)[::-1,:,:]
        im = Image.fromarray(im_data)
        im.save(os.path.join(args.save_image_dir,
                             f"sorted_{index:05}.png"))

  elif args.type == "terrain":
    # Take a screenshot of the terrain alone
    os.makedirs(args.save_image_dir, exist_ok=True)
    im_data = get_robot_image(None, task)[::-1,:,:]
    im = Image.fromarray(im_data)
    im.save(os.path.join(args.save_image_dir,
                         f"terrain_{args.task}.png"))

  elif args.type == "simulate":
    results_df = pd.DataFrame(columns=['task', 'log_file', 'iteration', 'reward'])
    for iteration in args.iterations:
      row = iteration_df.ix[iteration]
      rule_seq = ast.literal_eval(row['rule_seq'])
      robot = make_robot_from_rule_sequence(rule_seq, rules)
      for i in range(10):
        _, result = simulate(robot, task, random.getrandbits(32), args.jobs)
        results_df = results_df.append({
            'task': args.task, 'log_file': args.log_file,
            'iteration': iteration, 'reward': result}, ignore_index=True)

    with open('simulate_results.csv', 'a') as f:
      results_df.to_csv(f, header=(f.tell() == 0))

  elif args.type == "video":
    os.makedirs(args.save_image_dir, exist_ok=True)
    row = iteration_df.ix[args.iterations[0]]
    rule_seq = ast.literal_eval(row['rule_seq'])
    if args.opt_seed is not None:
      opt_seed = args.opt_seed
    else:
      opt_seed = int(row['opt_seed'])
    robot = make_robot_from_rule_sequence(rule_seq, rules)
    save_robot_video_frames(robot, task, opt_seed, args.jobs,
                            args.save_image_dir, frame_interval=4)

if __name__ == '__main__':
  main()

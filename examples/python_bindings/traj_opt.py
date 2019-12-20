import numpy as np
import pyrobotdesign as rd
import time

graphs = rd.load_graphs('data/designs/grammar7.dot')
rules = [rd.create_rule_from_graph(g) for g in graphs]

n0 = rd.Node()
n0.name = 'robot'
n0.attrs.label = 'robot'
robot_graph = rd.Graph()
robot_graph.nodes = [n0]

rule_sequence = [0, 1, 1, 2, 3, 3, 3, 5, 5, 5, 7, 7, 7, 8, 8, 8, 8, 8, 8, 11, 11, 11, 10, 10, 10]
for r in rule_sequence:
  matches = rd.find_matches(rules[r].lhs, robot_graph)
  if matches:
    robot_graph = rd.apply_rule(rules[r], robot_graph, matches[0])

robot = rd.build_robot(robot_graph)

floor = rd.Prop(0.0, 0.9, [10.0, 1.0, 10.0])
bump = rd.Prop(0.0, 0.9, [0.05, 0.10, 10.0])

time_step = 1.0 / 240
discount_factor = 0.99
interval = 4
horizon = 64
thread_count = 16
opt_seed = 0
episode_len = 250

# Find an initial y offset that will place the robot precisely on the ground
def find_y_offset(robot):
  temp_sim = rd.BulletSimulation(time_step)
  temp_sim.add_robot(robot, np.zeros(3), rd.Quaterniond(1.0, 0.0, 0.0, 0.0))
  lower = np.zeros(3)
  upper = np.zeros(3)
  temp_sim.get_robot_world_aabb(temp_sim.find_robot_index(robot), lower, upper)
  return -lower[1]

y_offset = find_y_offset(robot)

def make_sim_fn():
  sim = rd.BulletSimulation(time_step)
  sim.add_prop(floor, [0.0, -1.0, 0.0], rd.Quaterniond(1.0, 0.0, 0.0, 0.0))
  for i in range(15):
    sim.add_prop(bump, [0.5 + 0.5 * i, 0.0, 0.0], rd.Quaterniond(1.0, 0.0, 0.0, 0.0))
  # Rotate 180 degrees around the y axis, so the base points to the right
  sim.add_robot(robot, [0.0, y_offset, 0.0],
                rd.Quaterniond(0.0, 0.0, 1.0, 0.0))
  return sim

main_sim = make_sim_fn()
robot_idx = main_sim.find_robot_index(robot)

dof_count = main_sim.get_robot_dof_count(robot_idx)
value_estimator = rd.NullValueEstimator()
objective_fn = rd.DotProductObjective()
# Base starts out pointing in the negative x direction
objective_fn.base_dir_weight = np.array([-10.0, 0.0, 0.0])
objective_fn.base_vel_weight = np.array([100.0, 0.0, 0.0])
optimizer = rd.MPPIOptimizer(100.0, discount_factor, dof_count, interval,
                             horizon, 128, thread_count, opt_seed, make_sim_fn,
                             objective_fn, value_estimator)
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

print('Total reward: {:f}'.format(rewards.sum()))

main_sim.restore_state()

viewer = rd.GLFWViewer()
sim_time = time.time()
i = 0
j = 0
while not viewer.should_close():
  current_time = time.time()
  while sim_time < current_time:
    main_sim.set_joint_target_positions(robot_idx, input_sequence[:,j])
    main_sim.step()
    viewer.update(time_step)
    sim_time += time_step
    i += 1
    if i >= interval:
      i = 0
      j += 1
    if j >= input_sequence.shape[1]:
      i = 0
      j = 0
      main_sim.restore_state()
  viewer.render(main_sim)

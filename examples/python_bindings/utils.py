import pyrobotdesign as rd
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
      sim.set_joint_target_positions(robot_idx, input_sequence[:,j])
      sim.step()
      viewer.update(time_step)
      sim_time += time_step
      i += 1
      if i >= interval:
        i = 0
        j += 1
      if j >= input_sequence.shape[1]:
        i = 0
        j = 0
        sim.restore_state()
    viewer.render(sim)

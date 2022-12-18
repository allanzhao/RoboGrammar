import numpy as np

import pyrobotdesign as rd
import time


class CameraTracker(object):
    def __init__(self, viewer, sim, robot_idx):
        self.viewer = viewer
        self.sim = sim
        self.robot_idx = robot_idx

        self.reset()

    def update(self, time_step):
        lower = np.zeros(3)
        upper = np.zeros(3)
        self.sim.get_robot_world_aabb(self.robot_idx, lower, upper)

        # Update camera position to track the robot smoothly
        target_pos = 0.5 * (lower + upper)
        camera_pos = self.viewer.camera_params.position.copy()
        camera_pos += 5.0 * time_step * (target_pos - camera_pos)
        self.viewer.camera_params.position = camera_pos

    def reset(self):
        lower = np.zeros(3)
        upper = np.zeros(3)
        self.sim.get_robot_world_aabb(self.robot_idx, lower, upper)

        self.viewer.camera_params.position = 0.5 * (lower + upper)
        

def set_joint_torques(sim, torques):
    torques /= np.linalg.norm(torques, ord=1)
    torques /= torques.max()
    # neural_input = 0.05 + 0.95 * (neural_input - neural_input.min()) / (neural_input.max() - neural_input.min())
    for link, _torques in zip(sim.get_robot(0).links, torques):
        link.joint_torque = _torques
    

def prepare_viewer(sim):
    viewer = rd.GLFWViewer()
    
    # Get robot bounds
    lower = np.zeros(3)
    upper = np.zeros(3)
    sim.get_robot_world_aabb(0, lower, upper)

    # Set initial camera parameters
    viewer.camera_params.yaw = -np.pi / 4
    viewer.camera_params.pitch = -np.pi / 6
    viewer.camera_params.distance = 1.5 * np.linalg.norm(upper - lower)
        
    tracker = CameraTracker(viewer, sim, 0)
    
    return viewer, tracker


def viewer_step(sim, task, actions, viewer, tracker, step=0):
    for i in range(task.interval):
        step_idx = step * task.interval + i

        sim.set_joint_targets(0, actions)
        
        task.add_noise(sim, step_idx)
        sim.step()
        tracker.update(task.time_step)
        viewer.update(task.time_step)
        
    viewer.render(sim)

def view_trajectory(sim, robot_idx, input_sequence, task, no_view=False, neuron_stream_wrapper=None):
    record_step_indices = set()

    sim.save_state()

    viewer = rd.GLFWViewer()

    # Get robot bounds
    lower = np.zeros(3)
    upper = np.zeros(3)
    sim.get_robot_world_aabb(robot_idx, lower, upper)

    # Set initial camera parameters
    task_name = type(task).__name__
    if 'Ridged' in task_name or 'Gap' in task_name:
        viewer.camera_params.yaw = 0.0
    elif 'Wall' in task_name:
        viewer.camera_params.yaw = -np.pi / 2
    else:
        viewer.camera_params.yaw = -np.pi / 4
    viewer.camera_params.pitch = -np.pi / 6
    viewer.camera_params.distance = 1.5 * np.linalg.norm(upper - lower)
        
    tracker = CameraTracker(viewer, sim, robot_idx)

    j = 0
    k = 0
    
    if neuron_stream_wrapper is not None:
        neural_input = (np.expand_dims(neuron_stream_wrapper.neuron_stream.get_spike_frequencies(), axis=0) @ neuron_stream_wrapper.weights)[0]
        set_joint_torques(sim, neural_input)
    
    sim_time = time.time()
    while not viewer.should_close():
        current_time = time.time()
        while sim_time < current_time:
            step_idx = j * task.interval + k
            
            if input_sequence is not None:
                sim.set_joint_targets(robot_idx, input_sequence[:,j].reshape(-1, 1))
            
            task.add_noise(sim, step_idx)
            sim.step()
            tracker.update(task.time_step)
            viewer.update(task.time_step)
            
            if viewer.camera_controller.should_record():
                record_step_indices.add(step_idx)
            
            sim_time += task.time_step
            k += 1
            
            if k >= task.interval:
                if neuron_stream_wrapper is not None:
                    neural_input = (np.expand_dims(neuron_stream_wrapper.neuron_stream.get_spike_frequencies(), axis=0) @ neuron_stream_wrapper.weights)[0]
                    set_joint_torques(sim, neural_input)
                j += 1
                k = 0
                    
            if input_sequence is not None and j >= input_sequence.shape[1]:
                j = 0
                k = 0
                sim.restore_state()
                tracker.reset()
        viewer.render(sim)
        
        if no_view:
            break

    sim.restore_state()
    
    if neuron_stream_wrapper:
        neuron_stream_wrapper.stop()

    return viewer.camera_params, record_step_indices
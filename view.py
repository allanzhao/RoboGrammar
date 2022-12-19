import numpy as np

# import pyrobotdesign as rd
import time
from utils import set_joint_torques


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


def viewer_step(sim, task, actions, viewer, tracker, step=0, torques=None):
    if torques is not None:
        set_joint_torques(sim, torques, norm=True)
    
    for i in range(task.interval):
        step_idx = step * task.interval + i

        sim.set_joint_targets(0, actions)
        
        task.add_noise(sim, step_idx)
        sim.step()
        tracker.update(task.time_step)
        viewer.update(task.time_step)
        
    viewer.render(sim)
    
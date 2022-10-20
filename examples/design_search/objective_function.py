from symbol import power
import numpy as np

class DotProductObjective:
    
    def __init__(self, base_dir_weight=[-2.0, 0.0, 0.0], base_up_weight=[0.0, 2.0, 0.0], base_vel_weight=[2.0, 0.0, 0.0], power_weight=0, dof_mismatch_cost=0.1):
        self.base_dir_weight = np.array(base_dir_weight).reshape((3, 1))
        self.base_up_weight = np.array(base_up_weight).reshape((3, 1))
        self.base_vel_weight = np.array(base_vel_weight).reshape((3, 1))
        self.power_weight = power_weight
        self.dof_mismatch_cost = dof_mismatch_cost

    def __call__(self, sim, neural_input=None):
        reward = 0
        for robot_idx in range(sim.get_robot_count()):
            dof_count = sim.get_robot_dof_count(robot_idx)
            
            base_transform = np.asfortranarray(np.zeros((4, 4), dtype=np.float64))
            sim.get_link_transform(robot_idx, 0, base_transform)
            
            base_vel = np.zeros(6, dtype=np.float64)
            sim.get_link_velocity(robot_idx, 0, base_vel)
            
            joint_vel = np.zeros(dof_count, dtype=np.float64)
            sim.get_joint_velocities(robot_idx, joint_vel)
            
            motor_torques = np.zeros(dof_count, dtype=np.float64)
            sim.get_joint_motor_torques(robot_idx, motor_torques)
            
            base_dir = base_transform[:3, :1]
            reward += base_dir.T.dot(self.base_dir_weight)
            
            base_up = base_transform[:3, 1:2]
            reward += base_up.T.dot(self.base_up_weight)
            
            reward += base_vel[-3:].dot(self.base_vel_weight)
            
            power = motor_torques.dot(joint_vel)
            reward += self.power_weight * power
            
            if neural_input is not None and dof_count < len(neural_input):
                reward -= self.dof_mismatch_cost * np.sum(neural_input[dof_count:])
                
        return reward.reshape(-1)[0]
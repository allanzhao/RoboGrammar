from symbol import power
import numpy as np

class DotProductObjective:
    
    def __init__(self, base_dir_weight=[-2.0, 0.0, 0.0], base_up_weight=[0.0, 2.0, 0.0], base_vel_weight=[2.0, 0.0, 0.0], power_weight=0, dof_mismatch_cost=0.1):
        self.base_dir_weight = np.array(base_dir_weight).reshape((3, 1))
        self.base_up_weight = np.array(base_up_weight).reshape((3, 1))
        self.base_vel_weight = np.array(base_vel_weight).reshape((3, 1))
        self.power_weight = power_weight
        self.dof_mismatch_cost = dof_mismatch_cost
        self.init_position = None
        self.init_transform = None
        self.prev_position = None
        self.prev_transform = None
        self.mean_velocity = None

    def __call__(self, sim, neural_input=None, print_bool=False):
        assert sim.get_robot_count() == 1
        
        reward = 0
        robot_idx = 0
        
        dof_count = sim.get_robot_dof_count(robot_idx)
        
        base_transform = np.asfortranarray(np.zeros((4, 4), dtype=np.float64))
        sim.get_link_transform(robot_idx, 0, base_transform)
        
        lower = np.zeros(3)
        upper = np.zeros(3)
        sim.get_robot_world_aabb(robot_idx, lower, upper)
        
        if self.init_transform is None:
            self.init_transform = base_transform
        
        if self.init_position is None:
            self.init_position = np.array([-upper[0], -lower[1]])
            
        # print(base_transform)
        # print(self.init_position)
        # if print_bool:
        #     print((base_transform - self.init_transform).T)
            # or just use the base_velocity
            # or use the last column from base_transform
            # print(np.linalg.norm(self.init_position - np.array([-upper[0], -lower[1]])))
        
        # reward related to the link facing in a specific direction:
        # I think this can also be used in the variable rotation scenario
        base_dir = base_transform[:3, :1]
        
        # base_dir[1] = 0
        # reward += np.linalg.norm(base_dir) * np.sum(self.base_dir_weight)
        # reward += base_dir.T.dot(self.base_dir_weight)
        
        # reward related to having the base link as high as possible:
        base_up = base_transform[:3, 1:2]
        reward += base_up.T.dot(self.base_up_weight)
        
        # reward related to the velocity of the base link (in a specific direction):
        base_vel = np.zeros(6, dtype=np.float64)
        sim.get_link_velocity(robot_idx, 0, base_vel)
        
        if self.mean_velocity is None:
            self.mean_velocity = np.array(base_vel)
        else:
            tau = 0.99
            self.mean_velocity = tau * self.mean_velocity + (1 - tau) * base_vel
            
            
        # base_vel[-2] = 0
        # reward += np.linalg.norm(base_vel[-3:]) * np.sum(self.base_vel_weight)
        mean_base_vel = self.mean_velocity[-3:]
        base_vel_weight = 1 * mean_base_vel / np.linalg.norm(mean_base_vel)
        base_vel_weight[0] = 1
        # base_vel_weight = np.where(mean_base_vel < 0, base_vel_weight * -1, base_vel_weight)
        
        if print_bool:
            print(base_vel_weight)
        reward += base_vel[-3:].dot(base_vel_weight)
        # reward += base_vel[-3:].dot(self.base_vel_weight)
        
        # maximization (?) of used power:
        joint_vel = np.zeros(dof_count, dtype=np.float64)
        sim.get_joint_velocities(robot_idx, joint_vel)
        motor_torques = np.zeros(dof_count, dtype=np.float64)
        sim.get_joint_motor_torques(robot_idx, motor_torques)
        
        power = motor_torques.dot(joint_vel)
        reward += self.power_weight * power
        
        # punishment for not using all the neural inputs:
        if neural_input is not None and dof_count < len(neural_input):
            reward -= self.dof_mismatch_cost * np.sum(neural_input[dof_count:])
                
        return reward.reshape(-1)[0]
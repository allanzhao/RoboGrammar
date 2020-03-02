import sys, os
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'rl'))

import numpy as np
import gym
from gym import utils, spaces
from gym.utils import seeding
from os import path
import copy

from simulation.simulation_utils import *
from common.common import *
import tasks

class RobotLocomotionEnv(gym.Env):
    def __init__(self, args):
        # init task and robot
        task_class = getattr(tasks, args.task)
        self.task = task_class()
        self.robot = build_robot(args)
        
        # get init pos
        self.robot_init_pos, has_self_collision = presimulate(robot)
        
        if has_self_collision:
            print_error('robot design has self collision')

        # init simulation
        self.sim = make_sim_fn(self.task, self.robot, self.robot_init_pos)
        self.robot_index = self.sim.find_robot_index(self.robot)

        # init objective function
        self.objective_fn = self.task.get_objective_fn()

        # init frame skip
        self.frame_skip = self.task.interval

        # define action space and observation space
        self.action_dim = self.sim.get_robot_dof_count(self.robot_index)
        self.action_range = np.array([-np.pi, np.pi])
        self.action_space = spaces.Box(low = np.full(self.action_dim, -1.0), 
            high = np.full(self.action_dim, 1.0), dtype = np.float32)

        observation = self.get_obs()
        self.observation_space = spaces.Box(low = np.full(observation.shape, -np.inf), 
            high = np.full(observation.shape, np.inf), dtype = np.float32)

        # init seed
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.sim.remove_robot(0)
        self.sim.add_robot(self.robot, self.robot_init_pos, rd.Quaterniond(0.0, 0.0, 1.0, 0.0))
        self.robot_index = self.sim.find_robot_index(self.robot)
        assert self.robot_index == 0
    
        return self.get_obs()

    def get_obs(self):
        state = get_robot_state(self.sim, self.robot_index)
        obs = np.hstack((state[0:6], state[7], state[9:])) # remove x, z positions from observation
        return obs

    def detect_crash(self):
        crash = False
        return crash

    # control frequency is same as the simulation frequency
    # control observation is directly infered from state
    # control output action is the same as the action in simulation
    def step(self, u):
        action = u * np.pi

        reward = 0.0
        for _ in range(self.frame_skip):
            self.sim.set_joint_target_positions(self.robot_idx, deepcopy(action.reshape(-1, 1)))
            self.sim.step()
            reward += self.objective_fn(self.sim)
            
        obs = self.get_obs()
        done = self.detect_crash()
        
        return obs, reward, done, {}
        




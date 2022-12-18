import numpy as np

from utils import set_joint_torques


class SimEnvWrapper:
    
    def __init__(self, make_sim_and_task_fn, load=False):
        self.env, self.task = make_sim_and_task_fn()
        
        if load:
            self.env.restore_state_from_file("tmp.bullet")
        else:
            self.env.save_state_to_file("tmp.bullet")
        
        self.observation_dim = (None, )
        self.action_dim = self.env.get_robot_dof_count(0)
        self.seed = None
        self.real_step = False    
    
    def step(self, action, neural_input=None):
        r = 0
        
        if neural_input is not None:
            set_joint_torques(self.env, torques=neural_input)
        
        for k in range(self.task.interval):
            self.env.set_joint_targets(0, action.reshape(-1, 1))
            self.task.add_noise(self.env, (self.task.interval * self.seed + k) % (2 ** 32))
            self.env.step()
            
            r += self.task.get_objective_fn()(self.env, neural_input)
        
        if self.real_step:
            self.env.save_state_to_file("tmp.bullet")
        
        return None, r, None, None
    
    def reset(self, seed=None):
        if self.seed is None and seed is not None:
            self.seed = seed
    
    def set_seed(self, seed=None):
        self.seed = seed
    
    def set_env_state(self):
        self.env.restore_state()
    
    def real_env_step(self, boolean):
        self.real_step = boolean
        if not boolean:
            self.env.save_state()
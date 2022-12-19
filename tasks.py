from abc import ABC, abstractmethod
import numpy as np
# import pyrobotdesign as rd

from objective_function import DotProductObjective

class ForwardSpeedTask(ABC):
    def __init__(self, time_step=1.0/240, discount_factor=0.99, interval=16,
                             horizon=16, episode_len=float("inf"), noise_seed=0, force_std=0.0,
                             torque_std=0.0):
        self.time_step = time_step
        self.discount_factor = discount_factor
        self.interval = interval
        self.horizon = horizon
        self.episode_len = episode_len
        self.noise_seed = noise_seed
        self.force_std = force_std
        self.torque_std = torque_std

        self.objective_fn = DotProductObjective()
        self.objective_fn.base_dir_weight = np.array([-2.0, 0.0, 0.0])
        self.objective_fn.base_up_weight = np.array([0.0, 2.0, 0.0])
        self.objective_fn.base_vel_weight = np.array([2.0, 0.0, 0.0])
        self.objective_fn.power_weight = 0
        self.objective_fn.dof_mismatch_cost = 0.1

        # Maximum reasonable result (designs achieving higher results are rejected)
        self.result_bound = 10.0

    def get_objective_fn(self):
        return self.objective_fn

    def add_noise(self, sim, time_step_idx):
        assert sim.get_robot_count() == 1
        noise_rng = np.random.RandomState(self.noise_seed + time_step_idx)
        sim.add_link_force_torque(0, 0, noise_rng.normal(0.0, self.force_std, size=3), noise_rng.normal(0.0, self.torque_std, size=3))

    @abstractmethod
    def add_terrain(self, sim):
        pass

class FlatTerrainTask(ForwardSpeedTask):
    """
    Task where the objective is to move forward as quickly as possible over flat
    terrain.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.floor = rd.Prop(rd.PropShape.BOX, 0.0, 0.5, [40.0, 1.0, 10.0])

    def add_terrain(self, sim):
        sim.add_prop(self.floor, [0.0, -1.0, 0.0], rd.Quaterniond(1.0, 0.0, 0.0, 0.0))

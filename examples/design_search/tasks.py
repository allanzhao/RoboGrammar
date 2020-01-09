from abc import ABC, abstractmethod
import numpy as np
import pyrobotdesign as rd

class ForwardSpeedTask(ABC):
  def __init__(self, time_step=1.0/240, discount_factor=0.99, interval=16,
               horizon=16, episode_len=128):
    self.time_step = time_step
    self.discount_factor = discount_factor
    self.interval = interval
    self.horizon = horizon
    self.episode_len = episode_len

    self.objective_fn = rd.DotProductObjective()
    self.objective_fn.base_dir_weight = np.array([-1.0, 0.0, 0.0])
    self.objective_fn.base_up_weight = np.array([0.0, 1.0, 0.0])
    self.objective_fn.base_vel_weight = np.array([2.0, 0.0, 0.0])

  def get_objective_fn(self):
    return self.objective_fn

  @abstractmethod
  def add_terrain(self, sim):
    pass

class RidgedTerrainTask(ForwardSpeedTask):
  """
  Task where the objective is to move forward as quickly as possible over ridged
  terrain.
  """

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    self.floor = rd.Prop(0.0, 0.9, [10.0, 1.0, 10.0])
    self.bump = rd.Prop(0.0, 0.9, [0.05, 0.10, 10.0])

  def add_terrain(self, sim):
    sim.add_prop(self.floor, [0.0, -1.0, 0.0],
                 rd.Quaterniond(1.0, 0.0, 0.0, 0.0))
    for i in range(15):
      sim.add_prop(self.bump, [0.5 + 0.5 * i, 0.0, 0.0],
                   rd.Quaterniond(1.0, 0.0, 0.0, 0.0))

class FrozenLakeTask(ForwardSpeedTask):
  """
  Task where the objective is to move forward as quickly as possible on a flat,
  low-friction surface.
  """

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    self.floor = rd.Prop(0.0, 0.05, [10.0, 1.0, 10.0])
    self.floor.color = [0.8, 0.9, 1.0]

  def add_terrain(self, sim):
    sim.add_prop(self.floor, [0.0, -1.0, 0.0],
                 rd.Quaterniond(1.0, 0.0, 0.0, 0.0))

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
    self.objective_fn.base_vel_weight = np.array([1.0, 0.0, 0.0])

    # Maximum reasonable result (designs achieving higher results are rejected)
    self.result_bound = 10.0

  def get_objective_fn(self):
    return self.objective_fn

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

    self.floor = rd.Prop(0.0, 0.9, [10.0, 1.0, 10.0])

  def add_terrain(self, sim):
    sim.add_prop(self.floor, [0.0, -1.0, 0.0],
                 rd.Quaterniond(1.0, 0.0, 0.0, 0.0))

class RidgedTerrainTask(ForwardSpeedTask):
  """
  Task where the objective is to move forward as quickly as possible over ridged
  terrain.
  """

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    self.floor = rd.Prop(0.0, 0.9, [10.0, 1.0, 10.0])
    self.bump = rd.Prop(0.0, 0.9, [0.1, 0.1, 10.0])

  def add_terrain(self, sim):
    sim.add_prop(self.floor, [0.0, -1.0, 0.0],
                 rd.Quaterniond(1.0, 0.0, 0.0, 0.0))
    for i in range(10):
      sim.add_prop(self.bump, [0.5 + i, 0.0, 0.0],
                   rd.Quaterniond(1.0, 0.0, 0.0, 0.0))

class GapTerrainTask(ForwardSpeedTask):
  """
  Task where the objective is to move forward as quickly as possible over
  terrain with several large gaps.
  """

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    self.floor = rd.Prop(0.0, 0.9, [10.0, 1.0, 10.0])
    self.left_platform = rd.Prop(0.0, 0.9, [5.0, 1.0, 10.0])
    self.platform = rd.Prop(0.0, 0.9, [0.35, 1.0, 10.0])

  def add_terrain(self, sim):
    sim.add_prop(self.floor, [0.0, -2.0, 0.0],
                 rd.Quaterniond(1.0, 0.0, 0.0, 0.0))
    sim.add_prop(self.left_platform, [-5.0, -1.0, 0.0],
                 rd.Quaterniond(1.0, 0.0, 0.0, 0.0))
    for i in range(10):
      sim.add_prop(self.platform, [i, -1.0, 0.0],
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

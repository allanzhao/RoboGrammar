from abc import ABC, abstractmethod
import numpy as np
import pyrobotdesign as rd

class Task(ABC):
  @abstractmethod
  def add_terrain(self, sim):
    pass

  @abstractmethod
  def get_objective_fn(self):
    pass

class RidgedTerrainTask(Task):
  """
  Task where the objective is to track a target velocity over ridged terrain.
  """

  def __init__(self):
    self.floor = rd.Prop(0.0, 0.9, [10.0, 1.0, 10.0])
    self.bump = rd.Prop(0.0, 0.9, [0.05, 0.10, 10.0])

    self.objective_fn = rd.DotProductObjective()
    self.objective_fn.base_dir_weight = np.array([-1.0, 0.0, 0.0])
    self.objective_fn.base_up_weight = np.array([0.0, 1.0, 0.0])
    self.objective_fn.base_vel_weight = np.array([1.0, 0.0, 0.0])

  def add_terrain(self, sim):
    sim.add_prop(self.floor, [0.0, -1.0, 0.0],
                 rd.Quaterniond(1.0, 0.0, 0.0, 0.0))
    for i in range(15):
      sim.add_prop(self.bump, [0.5 + 0.5 * i, 0.0, 0.0],
                   rd.Quaterniond(1.0, 0.0, 0.0, 0.0))

  def get_objective_fn(self):
    return self.objective_fn

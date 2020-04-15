from abc import ABC, abstractmethod

class Env(ABC):
  @property
  @abstractmethod
  def initial_state(self):
    """Return the initial state, which should always be the same."""
    pass

  @abstractmethod
  def get_available_actions(self, state):
    """Return an iterable containing all actions that can be taken from the
    given state."""
    pass

  @abstractmethod
  def get_next_state(self, state, action):
    """Take the action from the given state and return the resulting state."""
    pass

  @abstractmethod
  def get_result(self, state):
    """Return the result of a playout ending in the given state (None if the
    result is unknown)."""
    pass

  @abstractmethod
  def get_key(self, state):
    """Return a key identifying the given state. The key may not be unique, as
    long as collisions are very unlikely."""
    pass


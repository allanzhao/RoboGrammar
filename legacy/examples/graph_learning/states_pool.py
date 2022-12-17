import random

class StatesPool:
    def __init__(self, capacity = 10000000):
        self.capacity = capacity
        self.pool = []
        self.position = 0

    def push(self, state):
        if len(self.pool) < self.capacity:
            self.pool.append(None)
        self.pool[self.position] = state
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        return random.sample(self.pool, batch_size)
    
    def __len__(self):
        return len(self.pool)
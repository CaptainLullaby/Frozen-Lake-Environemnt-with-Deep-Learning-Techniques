from collections import deque
import random

class ReplayBuffer:
    def __init__(self, buffer_size, random_state):
        self.buffer = deque(maxlen=buffer_size)
        self.random_state = random_state

    def __len__(self):
        return len(self.buffer)

    def append(self, transition):
        self.buffer.append(transition)

    def draw(self, batch_size):
        batch = list()
        for i in range(batch_size):
            batch.append(self.buffer.popleft())
            self.buffer.rotate(random.randint(0, len(self.buffer)))
        return batch
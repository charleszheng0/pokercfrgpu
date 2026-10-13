import random
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity=10000):
        # Store at most capacity experiences, discarding the oldest.
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, next_legal):
        # Save one experience from gameplay.
        self.buffer.append((state, action, reward, next_state, done, next_legal))

    def sample(self, batch_size):
        # Randomly select previous experiences for DQN training.
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        # Return the number of stored experiences.
        return len(self.buffer)


if __name__ == "__main__":
    buffer = ReplayBuffer(capacity=100)

    buffer.push(
        state=[1, 0, 0],
        action=1,
        reward=2,
        next_state=[0, 1, 0],
        done=False,
        next_legal=[True, True, False, False, False],
    )

    print("Experiences stored:", len(buffer))
    print("Sample:", buffer.sample(1))

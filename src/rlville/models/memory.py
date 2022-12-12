import numpy as np
import torch

from numpy.typing import NDArray


class ReplayBuffer:
    """
    Implement the Replay Buffer as a class, which contains:

        - data:   a list variable to store all transition tuples.
        - add:    add new transition tuple into the buffer
        - sample: sample a batch of training data from the memory buffer

    :param size:  length of history we're keeping in the replay buffer
    :param shape: tuple of lengths of the vectors (S, A, R, S_prime, done)
    """
    def __init__(self, size: int, shape: tuple[int, ...], device: str = "cpu"):
        # total size of the replay buffer is size x sum(shape), but we need to
        # know the individual shapes to unpack samples
        self.size, self.width = size, sum(shape)
        self.memory = torch.empty((self.size, self.width)).to(torch.double)
        self.i = 0
        self.device = device

        # Compute column slice pairs to split (S, A, R, S_prime, done)
        # out of the memory
        columns_idx = [0, *np.array(shape).cumsum().tolist()]
        self.columns_idx = [
            slice(i, j) for i, j in zip(columns_idx[:-1], columns_idx[1:])
        ]

    def __len__(self):
        return min(self.size, self.i)

    def add(self, S: tuple, A: NDArray, R: float, S_prime: tuple, done: bool):
        # Pack the experience into a tensor and confirm it's the width of our memory
        transition = torch.hstack([
            torch.tensor(S).double(),
            torch.from_numpy(A).double(),
            torch.tensor(R).double(),
            torch.tensor(S_prime).double(),
            torch.tensor(done).double(),
        ]).to(self.device)

        assert len(transition) == self.width

        # Update (i mod size) with the new experience
        i = self.i % self.size
        self.memory[i] = transition

        # Increment current index
        self.i += 1

    def sample(self, size: int) -> tuple[torch.tensor, ...]:
        """Sample :param batch_size: indices with replacement"""
        # Sample random indices with replacement from our memory
        rows_idx = torch.randint(0, len(self), size=(size,))
        sample = self.memory[rows_idx]

        # Unpack our memory by taking slices of our columns
        return tuple(torch.squeeze(sample[:, c]) for c in self.columns_idx)

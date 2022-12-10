from typing import Generic

import gymnasium as gym
from gymnasium.core import ActType, ObsType
from gymnasium.spaces.utils import flatten
from numpy.typing import NDArray


class Agent(Generic[ActType, ObsType]):
    def __init__(self, env: gym.Env):
        self.env = env

    def features(self, S: ObsType) -> NDArray:
        return flatten(self.env.observation_space, S)

    def action(self, S: ObsType):
        ...


class CentralizedController(Agent):
    ...

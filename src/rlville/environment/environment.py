import os
from typing import List, Tuple

import gymnasium as gym
import numpy as np

from gymnasium.core import RenderFrame, ActType, ObsType
from numpy.typing import NDArray
from matplotlib import image as mpimage, pyplot

from rlville.constants import PROJECT
from rlville.environment.market import Market
from rlville.types import Shape

Step = Tuple[ObsType, float, bool, bool, dict]


class RLVille(gym.Env):
    """
    RLVille is a minimal FarmVille-like simulator built on gymnasium designed
    for multi-resource value optimization with delayed rewards.

    RLVille expects a :param market: from which to purchase crops / options which
    are allocated to the next available of :param n_resources:.

    RLVille has a :param starting_balance: for each episode of play, which allows
    for exploring the values of the :param market:.
    """
    def __init__(
        self,
        market: Market,
        n_resources: int = 1,
        starting_balance: int = 500,
        seed: int | None = 1,
    ):
        self.n_resources = n_resources
        self.n_products = len(market)
        self.starting_balance = starting_balance

        # The action space in RLVille is a Discrete set of :param n_products:
        # stored in the Market. Each product has some metadata about how they
        # behave w.r.t the reward function.
        self.action_space = gym.spaces.Discrete(self.n_products, seed=seed)
        self.market = market

        # The observation space in RLVille is a Dict of resources and countdown
        # timers until they harvest alongside the current balance.
        self.observation_space = gym.spaces.Dict({
            "balance": gym.spaces.Box(low=0, high=np.inf, seed=seed),

            # Resources are a MultiDiscrete because it's a Categorical, so it
            # flattens to a one-hot representation. There are :param n_resources:,
            # that each can hold one of :param n_products:
            "resources": gym.spaces.MultiDiscrete(
                [self.n_products for _ in range(self.n_resources)],
                seed=seed,
            ),

            # Since the times are a continuous variable, we use a Box to
            # represent them to the agent. Specifically, this is time in
            # % completed so that there is a clear linear relationship
            # between reward and timer
            "timers": gym.spaces.Box(
                np.zeros(self.n_resources),
                np.ones(self.n_resources),
                seed=seed,
            ),
        })

        self.balance = starting_balance
        self.resources = np.zeros(self.n_resources, dtype=int)
        self.timers = np.zeros(self.n_resources, dtype=int)

    def reset(self, seed: int | None = None, options: dict | None = None):
        self.balance = self.starting_balance
        self.resources = np.zeros(self.n_resources, dtype=int)
        self.timers = np.zeros(self.n_resources, dtype=int)
        return self.observe(), {}

    @property
    def done(self):
        # GAME OVER if...
        #     1. there are no crops on any resources, AND
        #     2. there are no available actions according to our balance
        return (self.resources == 0).all() and (self.actions == 0).all()

    @property
    def actions(self) -> NDArray[ActType]:
        """
        :return:
            The actions currently available to the Agent given the current
            balance of the environment. This ensures that the agent cannot
            overdraft
        """
        return np.argwhere(self.market.cost <= self.balance).ravel()

    @property
    def best_action(self) -> int:
        # When the resources are full, it is always best to do nothing
        if np.count_nonzero(self.resources) == self.n_resources:
            return 0

        # Get the best reward per time
        reward_per_time = (self.market.cost + self.market.revenue) / self.market.growtime
        # Get the available actions
        available_reward_per_time = reward_per_time[self.actions]
        # Get the best action based on rewards per time of the available actions
        return np.argmax(available_reward_per_time)

    def observe(self):
        """
        Return tha available balance, the observation, and the percents
        completed for all the resources.

        [% completed](crop) = 100% - [time left / total grow time(crop)]
        """
        return {
            "balance": self.balance,
            "resources": self.resources,
            "timers": 1 - (self.timers / self.market.growtime[self.resources])
        }

    @property
    def shape(self) -> Shape:
        return self.n_resources, self.n_products

    @property
    def free_space(self) -> NDArray:
        """A free space is any resource that has empty crop on it"""
        return np.argwhere(self.resources == 0).ravel()

    def harvest(self) -> float:
        # Get an index of crops that are ready
        ready = self.timers == 0
        # Map revenue onto the resources by indexing the market
        revenues = self.market.revenue[self.resources]
        # Compute their rewards, cancelling anything that's not ready
        reward = (ready.astype(int) * revenues).sum()
        # Set the harvested resources to empty
        self.resources[ready] = 0
        # Return the reward
        return reward

    def plow(self, n: int = 1):
        # Generate :param n: random resources to plow.
        resources = np.random.choice(self.n_resources, size=n)
        # Set the resources to empty pastures
        self.resources[resources] = 0

    def step(self, action: ActType) -> Step:
        reward = 0

        # Step all the timers down
        np.clip(self.timers - 1, a_min=0, a_max=np.inf, out=self.timers)

        # Harvest any ready resources
        reward += self.harvest()

        # For any action other than "empty"...
        if action != 0:
            # If there's no free space, then plow a random one for -15 reward
            # This ensures there is always an available resource when the agent
            # asks for one, and it penalizes the agent according to FarmVille's
            # plowing dynamics
            if self.free_space.size == 0:
                self.plow(1)
                reward -= 15

            # Get the next available resource and plant it with the action
            next_resource = self.free_space[0]
            self.resources[next_resource] = action
            reward += self.market.cost[action]

        self.balance += reward
        return self.observe(), reward, self.done, False, {}

    def render(self) -> RenderFrame | List[RenderFrame] | None:
        pyplot.clf()

        # A figure is a line of crops, i.e. a bunch a squares
        figure = pyplot.figure()
        subplot_kw = {"xticks": [], "yticks": []}
        axes = figure.subplots(
            *(1, self.n_resources),
            sharey="row",
            squeeze=False,
            subplot_kw=subplot_kw
        )

        # Sub-plot the icon image for each of the resources
        for i in range(self.n_resources):
            axis, resource, timer = axes.flat[i], self.resources[i], self.timers[i]

            # Render the icon from the data directory
            icon = os.path.join(PROJECT, self.market[resource]["icon"])
            image = mpimage.imread(icon)

            # Set this specific square to the image
            axis.set_xlabel(f"timer={timer}")
            axis.imshow(image)

        figure.show()
        return

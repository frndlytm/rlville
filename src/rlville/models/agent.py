import random
from functools import reduce
from typing import Generic

import gymnasium as gym
import numpy as np
import torch

from gymnasium.core import ActType, ObsType
from gymnasium.spaces.utils import flatdim, flatten
from gymnasium.wrappers import FlattenObservation
from torch import nn, optim
from tqdm import trange

from rlville.environment import RLVille
from rlville.models.blocks import MultiLayerPerceptron
from rlville.models.memory import ReplayBuffer
from rlville.parametric import constant, ParametricFloat


class Agent(Generic[ActType, ObsType]):
    def __init__(
        self,
        name: str,
        env: RLVille,
        alpha: float = 1e-3,
        epsilon: ParametricFloat = constant(1e-3),
        gamma: ParametricFloat = constant(0.9),
        device: str = "cpu"
    ):
        # Agent metadata
        self.name = name
        self.env = env
        self.device = torch.device(device)

        # Learning parameters
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

    @property
    def shape(self):
        """The shape of an agent is a flattened (observation, action) space"""
        return (
            flatdim(self.env.observation_space),
            flatdim(self.env.action_space),
        )

    def feature(self, S: ObsType) -> torch.Tensor:
        return (
            torch
            .from_numpy(flatten(self.env.observation_space, S))
            .double().to(self.device)
        )

    def action(self, S: torch.Tensor, t: int):
        ...

    def train(self, *args, **kwargs):
        ...


class DQNAgent(Agent):
    memory: ReplayBuffer
    Q_t: nn.Module
    Q_b: nn.Module

    def __init__(
        self,
        *args,
        n_layers: int = 2,
        d_hidden: int = 128,
        memory_size: int = 10_000,
        **kwargs
    ):

        super().__init__(*args, **kwargs)
        self.d_obs, self.d_act = self.shape

        # Set up the agents memory of size
        #   :param memory_size: x (|S|, |A|, |R|, |S|, |bool|)
        self.memory = ReplayBuffer(
            memory_size, (self.d_obs, self.d_act, 1, self.d_obs, 1)
        )

        # Set up the learned Q functions
        self.Q_b, self.Q_t = (
            MultiLayerPerceptron(n_layers, d_hidden, self.d_obs, self.d_act)
            .to(self.device),
            MultiLayerPerceptron(n_layers, d_hidden, self.d_obs, self.d_act)
            .to(self.device),
        )

        # Learning configuration
        self.optimizer = optim.Adam(self.Q_b.parameters(), lr=self.alpha)
        self.loss_fn = nn.MSELoss()
        self.reset()

    @staticmethod
    def initializer(m: nn.Module):
        # compute the gain
        gain = nn.init.calculate_gain('relu')

        # init the convolutional layer
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            # init the params using uniform
            nn.init.xavier_uniform_(m.weight, gain=gain)
            nn.init.constant_(m.bias, 0)

    def reset(self):
        self.Q_b.apply(self.initializer)  # Initialized the behavior network, Q_b
        self.update()                     # Initialize the target network Q_t to Q_b
        return self

    def action(self, S: torch.Tensor, t: int) -> int:
        with torch.no_grad():
            available = self.env.actions

            # with probability eps, the agent selects a random action
            if random.random() < self.epsilon(t):
                return np.random.choice(available)

            # with probability 1 - eps, the agent selects a greedy policy
            else:
                q_values = self.Q_b(S)
                return int(q_values.argmax().item())

    def replay(self, t: int, size: int) -> float:
        """Update the behavior policy by replaying a batch of data from memory.

        :param t:
        :param size:
        :return:
        """
        # get the transition data
        S, A, R, S_prime, done = self.memory.sample(size)

        # Compute the predicted Q values using the behavior policy network
        Q_pred = torch.sum(self.Q_b(S) * A, dim=1)
        Q_t = self.Q_t(S_prime).max(dim=1)[0]

        # compute the TD target using the target network
        # (1 - done) ensures that if the step was terminal, then Q_t -> 0
        y = R + self.gamma(t) * (1.0 - done) * Q_t
        loss = self.loss_fn(Q_pred, y)

        # compute and propagate the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update(self):
        """Update the target network by copying the behavior network"""
        self.Q_t.load_state_dict(self.Q_b.state_dict())

    def save(self):
        torch.save(self.Q_b.state_dict(), f"./models/{self.name}.pt")

    def train(
        self,
        T: int,
        t_wait: int,
        f_update_behaviour: int,
        f_update_target: int,
        replay_size: int,
    ):
        # reset the environment to get the start state
        self.reset()
        S, _ = self.env.reset()

        # training variables
        balances, actions, best_actions, losses = [S[0]], [], [], []

        for t in (progress := trange(T)):
            # Step one epsilon-greedy action in the environment and remember it
            A = self.action(self.feature(S), t)
            S_prime, R, done, term, _ = self.env.step(A)

            balances.append(S_prime[0])
            actions.append(A)
            best_actions.append(self.env.best_action)

            A = gym.spaces.flatten(self.env.action_space, A)
            self.memory.add(S, A, R, S_prime, done or term)

            # check termination
            if done or term:
                # reset the environment, using S_prime since we are going to update
                # S with this later
                (S_prime, _), rewards = self.env.reset(), []

            # Update S
            S = S_prime

            if t > t_wait:
                # CODE HERE: update the behavior model
                if not bool(t % f_update_behaviour):
                    batch_loss = self.replay(t, replay_size)
                    losses.append(batch_loss)

                # CODE HERE: update the target model
                if not bool(t % f_update_target):
                    self.update()

        # save the results
        return (
            np.array(balances),
            np.array(actions),
            np.array(best_actions),
            np.array(losses)
        )

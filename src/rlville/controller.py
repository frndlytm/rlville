import queue

import gymnasium as gym
import networkx as nx

from .agent import Agent


class AgentSystem:
    """
    A Controller manages a weighted communication network between agents in a
    Multi-Agent Reinforcement Learning solution with Networked Agents. The network
    needs to be stored somewhere, so it might as well be here.

    Implementing the network inside a decoupled controller enables me to extend
    its features for centralized control when the time comes.
    """
    def __init__(self, env: gym.Env, *agents: Agent, p: float = 0.5, seed: int = None):
        """
        Given a list of agents and a random seed, construct a random weighted,
        directed graph representing communication channels between agents.

        :param agents:
        :param seed:
        """
        self.agents = agents
        self.d_agents = len(agents)
        # Randomize an adjacency matrix
        self.A = nx.adjacency_matrix(
            nx.fast_gnp_random_graph(self.d_agents, p, seed=seed, directed=True)
        )

        # Set up queues so that I can simulate time without actually looping noops
        self.active = queue.Queue(maxsize=self.d_agents)
        self.idle = queue.Queue(maxsize=self.d_agents)




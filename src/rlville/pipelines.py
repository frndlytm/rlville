from rlville.models.agents import Agent


def run_trials(agent: Agent, n: int = 5, **kwargs):
    # Create an experiment with n trials
    returns, losses, episodes = [], [], float("inf")

    for _ in range(n):
        returns_i, losses_i = agent.train(**kwargs)
        returns.append(returns_i)
        losses.append(losses_i)
        episodes = min(episodes, len(returns_i))

    # Cap all returns at the same number of episodes each
    return [r[:episodes] for r in returns], losses
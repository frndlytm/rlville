import gymnasium as gym
import pandas as pd


class Market:
    """
    A Market is a Discrete database space

    :param n: The number of crops we want to enable in the market.
    :param seed: Control the random state so we can always sample the same crops
        and get deterministic results.
    """
    def __init__(self, n: int | None = None, seed: int | None = None):
        df = pd.read_csv("./data/processed/market.tsv", sep="\t", index_col="id")
        n = n or df.index.size
        assert n <= df.index.size

        # Always include the empty action so the agent has something to do at
        # every time step. And sample the rest of the n-1 options from the market
        empty, sample = df.iloc[0], df.iloc[1:].sample(n - 1, random_state=seed)
        self.db = sample.append(empty).sort_index().reset_index()

    def __len__(self):
        # Returns the number of available actions
        return self.db.shape[0]

    def __getitem__(self, i: int):
        # Get a row from the database as a dictionary
        return self.db.iloc[i].to_dict()

    def __getattribute__(self, name: str):
        """
        Returns a column from the db.

        :param name:
        :return: A column from the market database, like "cost" or "revenue"
        """
        # Get a column from the database as a numpy array
        if name in self.db.columns:
            return getattr(self.db, name).to_numpy()

        # otherwise, disallow the access
        else:
            raise AttributeError("Only allowed to access columns from the market db")

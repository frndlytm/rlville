import pandas as pd


class Market:
    """
    A Market is a Discrete database space
    :param n: The number of crops we want to enable in the market.
    :param seed: Control the random state so we can always sample the same crops
        and get deterministic results.
    """
    def __init__(self, n: int | None = None, seed: int | None = None):
        df = pd.read_csv("./data/processed/market.csv", index_col="id")
        n = n or df.index.size
        assert n <= df.index.size

        self.db = df.sample(n, random_state=seed).reset_index()

    def __getitem__(self, i: int):
        return self.db.iloc[i].to_dict()

import torch
from torch import nn


class MultiLayerPerceptron(nn.Sequential):
    def __init__(self, n_layers: int, d_hidden: int, d_in: int = 1, d_out: int = 1):
        super().__init__(
            # Input to a linear layer with ReLU activation
            nn.Linear(d_in, d_hidden).double(),
            nn.ReLU().double(),
            # Unpack :param n_layers: fully-connected hidden layers with ReLU activation
            *(
                n_layers * [
                    nn.Linear(d_hidden, d_hidden).double(),
                    nn.ReLU().double(),
                ]
            ),
            # Output to a linear layer
            nn.Linear(d_hidden, d_out).double()
        )

        self.to(dtype=torch.double)

import torch
import torch.nn as nn


class GELU(nn.Module):
    """
    Applies the Gaussian Error Linear Unit (GELU) activation function.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


class FeedForward(nn.Module):
    """
    A feed-forward neural network module.

    Args:
        cfg (dict): Configuration dictionary containing model parameters.
            It should have keys "emb_dim" representing the embedding dimension.

    Attributes:
        layers (nn.Sequential): Sequential container of linear and activation layers.
    """

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)

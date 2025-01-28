import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Layer normalization layer for transforming the input tensor.

    Args:
        emb_dim (int): The dimension of the embedding.
        *args: Additional arguments to pass to nn.Module.
        **kwargs: Additional keyword arguments to pass to nn.Module.

    Attributes:
        eps (float): A small constant to prevent division by zero. Default is 1e-5.
        scale (nn.Parameter): Learnable scaling factor for normalization. Initialized to ones.
        shift (nn.Parameter): Learnable shifting term for normalization. Initialized to zeros.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor: Applies layer normalization to the input tensor.
    """

    def __init__(self, emb_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)

        return self.scale * norm_x + self.shift

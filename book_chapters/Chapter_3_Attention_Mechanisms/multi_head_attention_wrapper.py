import torch
import torch.nn as nn
from causal_attention import CausalAttention


class MultiHeadAttentionWrapper(nn.Module):
    """
    A wrapper for multi-head attention mechanism using causal attention.

    This module applies multiple causal attention heads in parallel and
    concatenates the results. It is useful in tasks where sequential
    dependencies are important, such as language modeling.

    Args:
        d_in (int): Dimension of input features.
        d_out (int): Dimension of output features per head.
        context_length (int): Length of the context for causal attention.
        dropout (float): Dropout rate to apply in each causal attention head.
        num_heads (int): Number of causal attention heads.
        qkv_bias (bool, optional): If True, add bias to query, key, and value
            projections. Defaults to False.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(
        self,
        d_in,
        d_out,
        context_length,
        dropout,
        num_heads,
        qkv_bias=False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.heads = nn.ModuleList(
            [
                CausalAttention(
                    d_in=d_in,
                    d_out=d_out,
                    context_length=context_length,
                    dropout=dropout,
                    qkv_bias=qkv_bias,
                )
                for _ in range(num_heads)
            ]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


# INFERENCE:
torch.manual_seed(123)
inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],  # Your
        [0.55, 0.87, 0.66],  # journey
        [0.57, 0.85, 0.64],  # starts
        [0.22, 0.58, 0.33],  # with
        [0.77, 0.25, 0.10],  # one
        [0.05, 0.80, 0.55],  # step
    ]
)

dropout = 0.5
batch = torch.stack((inputs, inputs), dim=0)
context_length = batch.shape[1]
d_in, d_out = 3, 2

mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print("context_vecs.shape:", context_vecs.shape)
print(context_vecs)

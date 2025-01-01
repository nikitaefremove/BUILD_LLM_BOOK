import torch
import torch.nn as nn


class SelfAttention_v1(nn.Module):
    """
    A simple implementation of the self-attention mechanism.

    This module performs self-attention on the input tensor by transforming it into
    queries, keys, and values using linear transformations defined by learned weight matrices.
    The attention scores are computed between queries and keys, and softmax is applied to these scores
    to obtain attention weights. These weights are then used to compute a weighted sum of the values,
    resulting in the context vector.

    Args:
        d_in (int): Dimensionality of the input tensor.
        d_out (int): Dimensionality of the query, key, and value transformations.

    Returns:
        torch.Tensor: Context vector after applying self-attention, with shape (batch_size, seq_len, d_out).
    """

    def __init__(self, d_in, d_out, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        """
        Performs the self-attention operation on the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_in).

        Returns:
            torch.Tensor: Context vector after applying self-attention,
                          with shape (batch_size, seq_len, d_out).
        """
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        context_vec = attn_weights @ values

        return context_vec

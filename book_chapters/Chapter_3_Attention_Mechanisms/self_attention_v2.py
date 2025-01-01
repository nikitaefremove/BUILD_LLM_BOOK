import torch
import torch.nn as nn


class SelfAttention_v2(nn.Module):
    """
    A simplified implementation of the self-attention mechanism.

    Args:
        d_in (int): The input dimension.
        d_out (int): The output dimension.
        qkv_bias (bool, optional): Whether to add bias terms to the query, key, and value linear layers. Defaults to False.
    """

    def __init__(self, d_in, d_out, qkv_bias=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.W_query = nn.Linear(in_features=d_in, out_features=d_out, bias=qkv_bias)
        self.W_key = nn.Linear(in_features=d_in, out_features=d_out, bias=qkv_bias)
        self.W_value = nn.Linear(in_features=d_in, out_features=d_out, bias=qkv_bias)

    def forward(self, x):
        """
        Computes the self-attention output given input x.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_in).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_out).
        """
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec

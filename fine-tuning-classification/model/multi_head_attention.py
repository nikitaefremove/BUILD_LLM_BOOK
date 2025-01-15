import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    Implementation of the multi-head attention mechanism.

    Args:
        d_in (int): Input dimension.
        d_out (int): Output dimension, must be divisible by num_heads.
        context_length (int): Length of the input sequence to attend over.
        dropout (float): Dropout rate for attention weights.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional): If True, add bias terms to query, key and value projections. Default: False.

    Returns:
        context_vec (torch.Tensor): The output tensor after applying multi-head attention.

    Raises:
        AssertionError: If d_out is not divisible by num_heads.
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
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        """
        Forward pass through the multi-head attention layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_tokens, d_in).

        Returns:
            context_vec (torch.Tensor): Output tensor after applying multi-head attention of shape (batch_size, num_tokens, d_out).
        """

        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        queries = queries.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec

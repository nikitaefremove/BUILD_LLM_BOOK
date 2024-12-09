import torch
import torch.nn as nn


class CausalAttention(nn.Module):
    """
    A CausalAttention module computes attention scores in a causal manner,
    ensuring that each token in the sequence only attends to the previous
    tokens and itself. This is achieved by applying an upper triangular mask
    to the attention score matrix.

    Attributes:
        d_out (int): Output dimension of the linear projection for queries, keys, and values.
        W_query (nn.Linear): Linear layer for projecting inputs to query vectors.
        W_keys (nn.Linear): Linear layer for projecting inputs to key vectors.
        W_values (nn.Linear): Linear layer for projecting inputs to value vectors.
        dropout (nn.Dropout): Dropout applied to the attention weights.
        mask (torch.Tensor): Upper triangular matrix used to mask out future tokens.

    Returns:
        torch.Tensor: Output tensor with context vectors for each token, shape (batch_size, num_tokens, d_out).
    """

    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_keys = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_values = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_keys(x)
        queries = self.W_query(x)
        values = self.W_values(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill(self.mask.bool()[:num_tokens, :num_tokens], torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values

        return context_vec

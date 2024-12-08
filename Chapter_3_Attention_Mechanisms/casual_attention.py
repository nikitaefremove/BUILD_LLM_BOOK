import torch
from self_attention_v2 import SelfAttention_v2

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
sa_v2 = SelfAttention_v2(d_in=3, d_out=2)

queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)

attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

# print(attn_weights)

context_length = attn_scores.shape[0]

mask_simple = torch.tril(torch.ones(context_length, context_length))
masked_simple = attn_weights * mask_simple
# print(masked_simple)

row_sums = masked_simple.sum(dim=1, keepdim=True)
masked_simple_norm = masked_simple / row_sums

# print(masked_simple_norm)

mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
# masked = attn_scores.masked_fill(mask.bool(), 0.)
attn_weights = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=-1)

print(attn_weights)

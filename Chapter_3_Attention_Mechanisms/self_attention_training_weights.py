import torch

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

torch.manual_seed(123)

x_2 = inputs[1]

# In chat gpt input dim and output dim usually the same.
d_in = inputs.shape[1]
d_out = 2


W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

query_2 = x_2 @ W_query
keys = inputs @ W_key
values = inputs @ W_value

# Compute attention score for second element.
keys_2 = keys[1]
attn_score_22 = query_2.dot(keys_2)
# print(attn_score_22)

# Compute attention score for all elements.
attn_scores_2 = query_2 @ keys.T
# print(attn_scores_2)

# Compute attention weights. Before scale it. (scaled-dot product attention)
d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
# print(attn_weights_2)

# Compute Context Vector
context_vec_2 = attn_weights_2 @ values
print(context_vec_2)

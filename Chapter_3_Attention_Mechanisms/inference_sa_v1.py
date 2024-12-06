import torch
from self_attention_v1 import SelfAttention_v1


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

sa_v1 = SelfAttention_v1(d_in=3, d_out=2)

print(sa_v1(inputs))

from self_attention_v2 import SelfAttention_v2

sa_v2 = SelfAttention_v2(d_in=3, d_out=2)

print(sa_v2(inputs))


print("W_query v1:\n", sa_v1.W_query)
print("W_query v2:\n", sa_v2.W_query.weight.T)

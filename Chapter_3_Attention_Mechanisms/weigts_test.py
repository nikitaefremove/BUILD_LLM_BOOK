import torch
import torch.nn as nn

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

x = nn.Parameter(torch.rand(3, 2))

y = nn.Linear(in_features=3, out_features=2)

x.data = y.weight

# print(x.data)

# print(y.weight)

from self_attention_v1 import SelfAttention_v1
from self_attention_v2 import SelfAttention_v2

sa_v1 = SelfAttention_v1(d_in=3, d_out=2)
sa_v2 = SelfAttention_v2(d_in=3, d_out=2)

sa_v1.W_query.data = sa_v2.W_query.weight.data.T
sa_v1.W_key.data = sa_v2.W_key.weight.data.T
sa_v1.W_value.data = sa_v2.W_value.weight.data.T

print(sa_v1(inputs))
print(sa_v2(inputs))

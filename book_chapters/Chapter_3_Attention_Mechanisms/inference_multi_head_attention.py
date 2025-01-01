import torch
import torch.nn as nn
from multi_head_attention import MultiHeadAttention

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
batch_size, context_length, d_in = batch.shape
d_out = 2

mha = MultiHeadAttention(
    d_in=d_in, d_out=d_out, context_length=context_length, num_heads=2, dropout=dropout
)

context_vecs = mha(batch)

print(context_vecs.shape)
print(context_vecs)

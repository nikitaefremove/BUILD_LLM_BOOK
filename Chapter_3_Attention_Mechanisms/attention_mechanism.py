"""
There are three Steps for creating simple attention mechanism:

- Compute Attention Scores (dot product between inputs (each to each))
- Create weigths - just normalized scores
- Compute Context Vector - weighted sum over the inputs
"""

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

# Calculate Attention Scores
attn_scores = torch.zeros(6, 6)


# Slow verstion using for loop
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)

# print(attn_scores)


attn_scores = inputs @ inputs.T


attn_weights = torch.softmax(attn_scores, dim=0)


# Compute Context Vectors
all_context_vecs = attn_weights @ inputs

print(all_context_vecs)

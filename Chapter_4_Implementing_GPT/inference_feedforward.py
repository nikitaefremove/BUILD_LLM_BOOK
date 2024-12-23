import torch
import torch.nn as nn

from feedforward import FeedForward
from config import GPT_CONFIG_124M

ffn = FeedForward(GPT_CONFIG_124M)

x = torch.rand(2, 3, 768)

out = ffn(x)

print(out.shape)

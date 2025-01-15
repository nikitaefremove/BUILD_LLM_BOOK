import torch

from model.architecture import GPTModel
from core.model_config import GPT_CONFIG_124M


model = GPTModel(GPT_CONFIG_124M)

# Freeze the model:
for param in model.parameters():
    param.requires_grad = False

torch.manual_seed(123)

num_classes = 2
model.out_head = torch.nn.Linear(
    in_features=GPT_CONFIG_124M["emb_dim"], out_features=num_classes
)

# Open last transformer block and layer norm for training
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True

for param in model.final_norm.parameters():
    param.requires_grad = True

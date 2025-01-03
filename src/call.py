import torch
import torch.nn as nn

from utils.train_val_loaders import train_loader, val_loader
from core.model_config import GPT_CONFIG_124M
from utils.models import GPTModel

from utils.calc_loss import calc_loss_loader


# print("Train Loader:")
# for x, y in train_loader:
#     print(x.shape, y.shape)

# print("\nValidation Loader:")
# for x, y in val_loader:
#     print(x.shape, y.shape)


model = GPTModel(cfg=GPT_CONFIG_124M)
device = torch.device("mps" if torch.mps.is_available() else "cpu")
print(f"Device: {device}")
model.to(device)

with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)

print(f"Train Loss: {train_loss}")
print(f"Validation Loss: {val_loss}")

import time
import torch

from model.architecture import GPTModel
from core.model_config import GPT_CONFIG_355M

from utils.loss import calc_loss_batch, calc_loss_loader

# from utils.train_model_simple import train_model_simple

from utils.data_processing.dataloaders import train_loader, val_loader, test_loader


# # Loading model and state dict (pretrained weights)
device = torch.device("mps" if torch.mps.is_available() else "cpu")
print(f"Device: {device}")

state_dict = torch.load(
    "fine-tuning-instruct/model/weights/gpt_with_hf_weights.pth", weights_only=True
)
model = GPTModel(GPT_CONFIG_355M)
model.load_state_dict(state_dict)
model.to(device)

torch.manual_seed(123)


# Test
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)


print("Training loss:", train_loss)
print("Validation loss:", val_loss)

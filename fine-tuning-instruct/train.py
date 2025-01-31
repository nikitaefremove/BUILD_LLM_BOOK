import time
import torch
import tiktoken

from model.architecture import GPTModel
from core.model_config import GPT_CONFIG_355M

from utils.loss import calc_loss_batch, calc_loss_loader
from utils.train_model_simple import train_model_simple
from utils.data_processing.dataloaders import train_loader, val_loader, test_loader
from utils.data_processing.format_input import format_input
from utils.data_processing.partitioning_dataset import val_data
from utils.plot_values import plot_values


# # Loading model and state dict (pretrained weights):
device = torch.device("mps" if torch.mps.is_available() else "cpu")
print(f"Device: {device}")

tokenizer = tiktoken.get_encoding("gpt2")

state_dict = torch.load(
    "fine-tuning-instruct/model/weights/gpt_with_hf_weights.pth", weights_only=True
)
model = GPTModel(GPT_CONFIG_355M)
model.load_state_dict(state_dict)
model.to(device)


# Training Pipeline:
start_time = time.time()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)
num_epochs = 2

train_losses, val_losses, tokens_seen = train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs=num_epochs,
    eval_freq=5,
    eval_iter=5,
    start_context=format_input(val_data[0]),
    tokenizer=tokenizer,
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60

print(f"Training completed in {execution_time_minutes:.2f} minutes.")

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_values(epochs_tensor, tokens_seen, train_losses, val_losses)

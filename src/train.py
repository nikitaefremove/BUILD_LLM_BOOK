import torch
import tiktoken

from utils.visualize_training_process import plot_losses

from core.model_config import GPT_CONFIG_124M
from utils.models import GPTModel

from utils.train_model_simple import train_model_simple
from utils.train_val_loaders import train_loader, val_loader


def train(
    num_epochs: int = 10,
    learning_rate: float = 0.001,
    start_context: str = "Every efforts moves you",
):

    torch.manual_seed(123)
    device = "mps" if torch.mps.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = tiktoken.get_encoding("gpt2")

    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.1
    )

    train_losses, val_losses, tokens_seen = train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=num_epochs,
        eval_freq=5,
        eval_iter=5,
        start_context=start_context,
        tokenizer=tokenizer,
    )

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))

    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)


print(train())
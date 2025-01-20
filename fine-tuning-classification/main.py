import time
import torch

from utils.data_processing.dataloaders import train_loader, val_loader, test_loader
from model.classification_model import model
from utils.loss import calc_loss_loader
from utils.train_classifier import train_classifier
from utils.plot_values import plot_values


device = "mps" if torch.mps.is_available() else "cpu"
model.to(device)

start_time = time.time()
torch.manual_seed(123)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
num_epochs = 10

train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq=50,
    eval_iter=5,
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Total time for training: {execution_time_minutes} minutes")

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))
plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)


# Calculate final accuracy scores
train_accuracy = calc_loss_loader(train_loader, model, device)
val_accuracy = calc_loss_loader(val_loader, model, device)
test_accuracy = calc_loss_loader(test_loader, model, device)

print(f"Train Accuracy Score: {train_accuracy}")
print(f"Validation Accuracy Score: {val_accuracy}")
print(f"Test Accuracy Score: {test_accuracy}")

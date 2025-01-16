import torch

from utils.data_processing.dataloaders import train_loader, val_loader, test_loader
from model.classification_model import model
from utils.loss import calc_loss_loader


device = "mps" if torch.mps.is_available() else "cpu"
model.to(device)

train_accuracy = calc_loss_loader(train_loader, model, device, num_batches=10)
val_accuracy = calc_loss_loader(val_loader, model, device, num_batches=10)
test_accuracy = calc_loss_loader(test_loader, model, device, num_batches=10)


print(train_accuracy, val_accuracy, test_accuracy)

from utils.data_processing.dataloaders import train_loader, val_loader, test_loader


print("Train loader:")
for inputs, targets in train_loader:
    print(inputs.shape, targets.shape)

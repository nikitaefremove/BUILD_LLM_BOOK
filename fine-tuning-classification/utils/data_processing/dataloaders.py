import tiktoken
import torch

from torch.utils.data import DataLoader
from utils.data_processing.dataset import SpamDataset


torch.manual_seed(123)
tokenizer = tiktoken.get_encoding("gpt2")
batch_size = 8
num_workers = 0

# Create Datasets
train_dataset = SpamDataset(
    csv_file="fine-tuning-classification/data/train.csv", tokenizer=tokenizer, max_length=None
)
test_dataset = SpamDataset(
    csv_file="fine-tuning-classification/data/test.csv", tokenizer=tokenizer, max_length=None
)
val_dataset = SpamDataset(
    csv_file="fine-tuning-classification/data/validation.csv", tokenizer=tokenizer, max_length=None
)

# Create Dataloaders
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=True,
    drop_last=True,
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=True,
    drop_last=True,
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=True,
    drop_last=True,
)


for input_batch, target_batch in train_loader:
    pass

print("Input batch dimensions:", input_batch.shape)
print("Label batch dimensions", target_batch.shape)


print(f"{len(train_loader)} training batches")
print(f"{len(val_loader)} validation batches")
print(f"{len(test_loader)} test batches")
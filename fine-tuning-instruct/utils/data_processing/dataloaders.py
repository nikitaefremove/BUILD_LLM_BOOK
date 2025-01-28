import torch
import tiktoken

from torch.utils.data import DataLoader

from utils.data_processing.instruction_dataset import InstructionDataset
from utils.data_processing.collate_fn import customized_collate_fn
from utils.data_processing.partitioning_dataset import train_data, val_data, test_data


torch.manual_seed(123)
tokenizer = tiktoken.get_encoding("gpt2")
nums_workers = 0
batch_size = 8


train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=nums_workers,
)

val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=nums_workers,
)

test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=nums_workers,
)

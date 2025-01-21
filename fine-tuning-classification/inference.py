import torch
import tiktoken

from utils.classify import classify
from model.classification_model import model
from utils.data_processing.dataloaders import train_dataset


tokenizer = tiktoken.get_encoding("gpt2")
device = "mps" if torch.mps.is_available() else "cpu"
model.to(device)

checkpoint_path = "fine-tuning-classification/model/weights/model_and_optimizer.pth"
checkpoint = torch.load(checkpoint_path, weights_only=True)

model.load_state_dict(checkpoint["model_state_dict"])


text_1 = "You are a winner you have been specially selected to receive $1000 cash or a $2000 award."
text_2 = "Hey, just wanted to check if we're still on for dinner tonight? Let me know!"


print(classify(text_1, model, tokenizer, device, max_length=train_dataset.max_length))
print(classify(text_2, model, tokenizer, device, max_length=train_dataset.max_length))

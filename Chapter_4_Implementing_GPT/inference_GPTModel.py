import torch
import tiktoken

from GPTModel import GPTModel
from config import GPT_CONFIG_124M


torch.manual_seed(123)

tokenizer = tiktoken.get_encoding("gpt2")
batch = []

txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))

batch = torch.stack(batch, dim=0)


model = GPTModel(cfg=GPT_CONFIG_124M)

out = model(batch)

# print("Input batch:\n", batch)
# print("\nOutput shape:", out.shape)
# print(out)


total_params = sum(p.numel() for p in model.parameters())
final_params = total_params - sum(p.numel() for p in model.out_head.parameters())
print(f"Total number of parameters: {final_params:,}")

total_size_bytes = total_params * 4
total_size_gb = total_size_bytes / (1024**3)
print(f"Total size: {total_size_gb:.2f} GB")

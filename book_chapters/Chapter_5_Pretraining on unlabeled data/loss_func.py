import torch
import torch.nn as nn

from utils.GPTModel import GPTModel
from utils.config import GPT_CONFIG_124M


model = GPTModel(cfg=GPT_CONFIG_124M)

inputs = torch.tensor(
    [[16833, 3636, 6100], [40, 1107, 588]]
)  ## Every efforts moves \  I really like

targets = torch.tensor(
    [[3626, 6100, 345], [1107, 588, 11311]]
)  ## efforts moves you \ really like chocolate

with torch.no_grad():
    logits = model(inputs)


# Create a probabilities from logits
probs = torch.softmax(logits, dim=-1)
# print(probs.shape)

token_ids = torch.argmax(probs, dim=-1, keepdim=True)
# print("Token ID's: \n", token_ids)


text_idx = 0
target_probs_1 = probs[text_idx, [0, 1, 2], targets[text_idx]]
# print("Text 1:", target_probs_1)

text_idx = 1
target_probs_2 = probs[text_idx, [0, 1, 2], targets[text_idx]]
# print("Text 2:", target_probs_2)

log_probs = torch.log(torch.cat((target_probs_1, target_probs_2)))
# print(log_probs)

avg_log_probs = torch.mean(log_probs)
# print(avg_log_probs)

neg_avg_log_probs = avg_log_probs * -1
print(neg_avg_log_probs)

logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()
# print(f"Flatten logits: {logits_flat.shape}")
# print(f"Flatten targets: {targets_flat.shape}")

loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print(f"Loss: {loss}")

perplexity = torch.exp(loss)
print(f"Perplexity: {perplexity}")


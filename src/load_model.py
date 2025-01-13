import torch
import tiktoken

from core.model_config import GPT_CONFIG_124M

from utils.models import GPTModel
from utils.generate_text import generate_text, text_to_token_ids, token_ids_to_text


model = GPTModel(GPT_CONFIG_124M)
device = "mps" if torch.mps.is_available() else "cpu"
tokenizer = tiktoken.get_encoding("gpt2")

state_dict = torch.load("src/model_weights/model_and_optimizer.pth")
model.load_state_dict(state_dict["model_state_dict"])

model.to(device)


torch.manual_seed(123)

token_ids = generate_text(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=25,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=50,
    temperature=1.5,
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

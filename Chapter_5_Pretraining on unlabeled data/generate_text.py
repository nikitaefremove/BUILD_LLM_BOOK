import tiktoken
import torch
from utils.evaluation import generate_text_simple
from utils.GPTModel import GPTModel
from utils.config import GPT_CONFIG_124M


tokenizer = tiktoken.get_encoding("gpt2")
model = GPTModel(cfg=GPT_CONFIG_124M)


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensors = torch.tensor(encoded).unsqueeze(0)

    return encoded_tensors


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)

    return tokenizer.decode(flat.tolist())


start_context = "Every effort moves you"
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text=start_context, tokenizer=tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"],
)

print("Output text: \n", token_ids_to_text(token_ids=token_ids, tokenizer=tokenizer))

import tiktoken
import torch
from utils.models import GPTModel
from core.model_config import GPT_CONFIG_124M


tokenizer = tiktoken.get_encoding("gpt2")
model = GPTModel(cfg=GPT_CONFIG_124M)


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    Generates text using a simple greedy approach.

    Args:
        model (GPTModel): The pre-trained GPT model to use.
        idx (torch.Tensor): The input token indices.
        max_new_tokens (int): Maximum number of new tokens to generate.
        context_size (int): Number of previous tokens to consider for prediction.

    Returns:
        torch.Tensor: The generated token indices.
    """

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def text_to_token_ids(text, tokenizer):
    """
    Converts a string of text to token IDs.

    Args:
        text (str): The input text.
        tokenizer (tiktoken.Encoding): The tokenizer to use.

    Returns:
        torch.Tensor: The tensor of token IDs.
    """

    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensors = torch.tensor(encoded).unsqueeze(0)

    return encoded_tensors


def token_ids_to_text(token_ids, tokenizer):
    """
    Converts a tensor of token IDs to a string of text.

    Args:
        token_ids (torch.Tensor): The tensor of token IDs.
        tokenizer (tiktoken.Encoding): The tokenizer to use.

    Returns:
        str: The decoded text.
    """

    flat = token_ids.squeeze(0)

    return tokenizer.decode(flat.tolist())


def generation_pipeline(text: str) -> str:
    """
    Generates text using a pre-trained GPT model.

    Args:
        text (str): The input text to generate more text from.

    Returns:
        str: The generated text.
    """

    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(text=text, tokenizer=tokenizer),
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"],
    )

    output_text = token_ids_to_text(token_ids=token_ids, tokenizer=tokenizer)

    return output_text

import torch
import tiktoken

from utils.data_processing.format_input import format_input
from utils.data_processing.partitioning_dataset import val_data
from model.architecture import GPTModel
from core.model_config import GPT_CONFIG_355M

from utils.generate_text import generation_pipeline

torch.manual_seed(123)
input_text = format_input(val_data[0])

print(f"Input: {input_text}")


response_text = generation_pipeline(text=input_text)


print(f"Output: {response_text}")

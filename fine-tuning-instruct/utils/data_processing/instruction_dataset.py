import torch
from torch.utils.data import Dataset

from format_input import format_input


class InstructionDataset(Dataset):
    """
    A dataset class for instruction-following tasks.

    Args:
        data (list of dict): A list of dictionaries where each dictionary contains an "input" and "output".
        tokenizer (Tokenizer): A tokenizer object used to encode the input and output.

    Attributes:
        data (list of dict): The original dataset.
        encoded_texts (list of int): Encoded texts corresponding to the dataset entries.

    Methods:
        __init__(self, data, tokenizer): Initializes the InstructionDataset instance.
        __getitem__(self, index): Returns the encoded text at a specified index.
        __len__(self): Returns the number of entries in the dataset.
    """

    def __init__(self, data, tokenizer):
        super().__init__()
        self.data = data
        self.encoded_texts = []

        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n ## Response: \n {entry["output"]}"
            full_text = instruction_plus_input + response_text

            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, index):
        return self.encoded_texts[index]
    
    def __len__(self):
        return len(self.data)
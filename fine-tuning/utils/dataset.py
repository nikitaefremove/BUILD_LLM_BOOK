import pandas as pd
import torch
from torch.utils.data import Dataset


class SpamDataset(Dataset):

    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        """
        Initializes a dataset for spam classification.

        Args:
            csv_file (str): The path to the CSV file containing the dataset.
                The CSV should have at least two columns: 'Text' and 'Label'.
            tokenizer (Tokenizer): A tokenizer object from a library like Hugging Face Transformers.
            max_length (int, optional): The maximum length of encoded texts. If None, it will be set to
                the longest text in the dataset. Defaults to None.
            pad_token_id (int, optional): The token ID used for padding. Defaults to 50256.

        Attributes:
            data (pd.DataFrame): The DataFrame containing the dataset.
            encoded_texts (list of list of int): Encoded texts from the dataset.
            max_length (int): The maximum length of encoded texts.
        """

        super().__init__()
        self.data = pd.read_csv(csv_file)

        if "Text" not in self.data.columns or "Label" not in self.data.columns:
            raise ValueError("CSV file must contain 'Text' and 'Label' columns.")

        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length

        self.encoded_texts = [
            self._pad_or_truncate(encoded_text) for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        """
        Returns the item at the specified index.

        Args:
            index (int): The index of the item to return.

        Returns:
            tuple: A tuple containing the encoded text and its corresponding label.
        """

        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]

        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )

    def __len__(self):
        """
        Returns the number of items in the dataset.

        Returns:
            int: The number of items.
        """

        return len(self.data)

    def _longest_encoded_length(self):
        """
        Computes the longest encoded length among all texts in the dataset.

        Returns:
            int: The maximum length.
        """

        return max(len(encoded_text) for encoded_text in self.encoded_texts)

    def _pad_or_truncate(self, encoded_text):
        """
        Pads or truncates an encoded text to the specified maximum length.

        Args:
            encoded_text (list of int): The encoded text to pad or truncate.

        Returns:
            list of int: The padded or truncated encoded text.
        """

        if len(encoded_text) > self.max_length:
            return encoded_text[: self.max_length]
        else:
            return encoded_text + [50256] * (self.max_length - len(encoded_text))

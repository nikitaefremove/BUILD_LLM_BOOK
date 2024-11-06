import re


class SimpleTokenizerV1:
    """
    A simple tokenizer that encodes and decodes text using a predefined vocabulary.

    Attributes:
        str_to_int (dict): A dictionary mapping strings to unique integer ids.
        int_to_str (dict): A dictionary mapping integer ids to original strings.

    Methods:
        encode(text):
            Converts a given text into a list of integers based on the vocabulary.

        decode(ids):
            Converts a list of integers back into the corresponding text.
    """

    def __init__(self, vocab):
        """
        Initializes the SimpleTokenizerV1 with a given vocabulary.

        Parameters:
            vocab (dict): A dictionary where keys are strings and values are unique integer ids.
        """
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        """
        Encodes the input text to a sequence of integer ids based on the vocabulary.

        Parameters:
            text (str): The input text to be encoded.

        Returns:
            list: A list of integers representing the encoded text.
        """
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]

        return ids

    def decode(self, ids):
        """
        Decodes a sequence of integer ids back to the original text.

        Parameters:
            ids (list): A list of integer ids to be decoded.

        Returns:
            str: The decoded text corresponding to the input ids.
        """
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)

        return text

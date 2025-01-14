import pandas as pd
import torch
from torch.utils.data import Dataset

class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        super().__init__()
        # Загружаем данные
        self.data = pd.read_csv(csv_file)
        
        # Проверка на наличие необходимых колонок
        if "Text" not in self.data.columns or "Label" not in self.data.columns:
            raise ValueError("CSV file must contain 'Text' and 'Label' columns.")

        # Кодируем тексты
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]

        # Устанавливаем максимальную длину
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length

        # Приводим все тексты к единой длине
        self.encoded_texts = [
            self._pad_or_truncate(encoded_text) for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]

        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        """Определяет максимальную длину среди закодированных текстов."""
        return max(len(encoded_text) for encoded_text in self.encoded_texts)

    def _pad_or_truncate(self, encoded_text):
        """Обрезает или дополняет текст до заданной длины."""
        if len(encoded_text) > self.max_length:
            return encoded_text[:self.max_length]
        else:
            return encoded_text + [50256] * (self.max_length - len(encoded_text))

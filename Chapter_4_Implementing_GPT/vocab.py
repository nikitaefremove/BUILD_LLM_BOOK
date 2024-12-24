import re

filepath = "data/the-verdict.txt"

with open(filepath, "r", encoding="utf-8") as file:
    raw_text = file.read()


preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]


all_words = sorted(set(preprocessed))
vocab_size = len(all_words)

all_words = sorted(list(set(preprocessed)))
all_words.extend(["<|endoftext|>", "<|unk|>"])

vocab = {token: integer for integer, token in enumerate(all_words)}

print(vocab)

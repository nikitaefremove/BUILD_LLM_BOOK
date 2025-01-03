import re

filepath = "data/the-verdict.txt"

with open(filepath, "r", encoding="utf-8") as file:
    text_data = file.read()

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text_data)
preprocessed = [item.strip() for item in preprocessed if item.strip()]


all_words = sorted(set(preprocessed))
vocab_size = len(all_words)

all_words = sorted(list(set(preprocessed)))
all_words.extend(["<|endoftext|>", "<|unk|>"])

vocab = {token: integer for integer, token in enumerate(all_words)}


# Check information about vocabulary
# from tokenizer import SimpleTokenizerV2

# tokenizer = SimpleTokenizerV2(vocab=vocab)

# total_characters = len(text_data)
# total_tokens = len(tokenizer.encode(text_data))

# print(f"Total characters: {total_characters}")
# print(f"Total tokens: {total_tokens}")

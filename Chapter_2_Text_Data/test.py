from simple_tokenizer_v2 import SimpleTokenizerV2
from vocab import vocab

tokenizer = SimpleTokenizerV2(vocab)

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))

ids = tokenizer.encode(text)
print(ids)

print(tokenizer.decode(ids))

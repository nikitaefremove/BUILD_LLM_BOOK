from simple_tokenizer import SimpleTokenizerV1
from vocab import vocab

tokenizer = SimpleTokenizerV1(vocab)

text = """
"It's the last he painted, you know,"
Mrs. Gisburn said with pardonable pride."""

ids = tokenizer.encode(text)
print(ids)

print(tokenizer.decode(ids))

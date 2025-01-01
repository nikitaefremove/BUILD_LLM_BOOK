# Example of using BPE (Byte Pair Encoding)

import tiktoken

tokenizer = tiktoken.get_encoding("o200k_base")


word = tokenizer.encode("a world, Nikita Efremov!")

print(word)

decoded_word = tokenizer.decode(word)
print(decoded_word)

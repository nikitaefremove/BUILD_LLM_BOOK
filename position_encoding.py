import torch

from test_dataloader import create_dataloader_v1, raw_text

# device = torch.device("mps" if torch.mps.is_available() else "cpu")
device = torch.device("cpu")
print(f"Device: {device}")

vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim).to(device)

max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
# print("Token IDs:\n", inputs)
# print("\nInputs shape:\n", inputs.shape)


token_embeddings = token_embedding_layer(inputs)
# print(token_embeddings)

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
# print(pos_embeddings)


input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)

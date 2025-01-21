import torch


def classify(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    """
    Classify a given text as either "spam" or "not spam".

    Args:
        text (str): The input text to classify.
        model (object): The pre-trained model used for classification.
        tokenizer (object): The tokenizer used to convert the text into input IDs.
        device (str): The device on which to perform the computation ('cpu', 'mps' or 'cuda').
        max_length (int, optional): The maximum length of the input sequence. Defaults to None.
        pad_token_id (int, optional): The ID of the padding token. Defaults to 50256.

    Returns:
        str: The predicted label ("spam" or "not spam").
    """

    model.eval()

    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[1]

    input_ids = input_ids[: min(max_length, supported_context_length)]

    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]

    predicted_label = torch.argmax(logits, dim=-1).item()

    return "spam" if predicted_label == 1 else "not spam"

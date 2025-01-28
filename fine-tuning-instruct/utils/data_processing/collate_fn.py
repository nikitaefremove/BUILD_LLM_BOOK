import torch


def custom_collate_fn(
    batch, pad_token_id=50256, ignore_index=-100, allowed_max_length=None, device="cpu"
):
    """
    Custom collate function for PyTorch DataLoader that pads and processes batches of tokenized sequences.

    Args:
        batch (list): A list of tokenized sequences where each sequence is a list of integers.
        pad_token_id (int): The token ID used for padding. Default is 50256.
        ignore_index (int): The index at which to ignore tokens during training. Default is -100.
        allowed_max_length (int, optional): Maximum length for each sequence in the batch. If provided,
            sequences longer than this will be truncated. Defaults to None.
        device (str): Device on which to store tensors ('cpu' or 'cuda'). Default is 'cpu'.

    Returns:
        tuple: A tuple containing two PyTorch tensors, `input_tensor` and `target_tensor`, where each tensor
        represents a batch of sequences.
    """
    
    batch_max_length = max(len(item) + 1 for item in batch)
    input_1st, target_1st = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))

        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()

        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        input_1st.append(inputs)
        target_1st.append(targets)

    input_tensor = torch.stack(input_1st).to(device)
    target_tensor = torch.stack(target_1st).to(device)

    return input_tensor, target_tensor

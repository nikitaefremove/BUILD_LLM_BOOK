import torch
import torch.nn as nn


def calc_loss_batch(input_batch, target_batch, model, device):
    """
    Calculates the batch loss for a given input and target batch using a specified model on a given device.

    Parameters:
        input_batch (torch.Tensor): The input batch tensor of shape (batch_size, num_features).
        target_batch (torch.Tensor): The target batch tensor of shape (batch_size,).
        model (nn.Module): The PyTorch model to be used for forward pass.
        device (str or torch.device): The device on which the tensors and model are located ('mps', 'cpu' or 'cuda').

    Returns:
        loss (torch.Tensor): The calculated loss value as a scalar tensor.

    """

    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())

    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """
    Calculate the average loss over a subset of batches from a given data loader.

    Args:
        data_loader (DataLoader): The data loader containing the input and target batches.
        model (nn.Module): The PyTorch model to calculate the loss for.
        device (torch.device): The device on which the model and data should be moved (e.g., 'cpu', 'cuda').
        num_batches (Optional[int], optional): Number of batches to calculate the loss for. If None, uses all batches. Defaults to None.

    Returns:
        float: The average loss over the specified number of batches. If no batches are available, returns `nan`.

    """

    total_loss = 0

    if len(data_loader) == 0:
        return float("nan")

    elif num_batches is None:
        num_batches = len(data_loader)

    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()

        else:
            break

    return total_loss / num_batches

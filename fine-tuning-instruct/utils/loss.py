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
    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = target_batch.view(-1)

    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)

    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """
    Calculates the loss over a specified number of batches from a data loader.

    Args:
        data_loader (torch.utils.data.DataLoader): The data loader containing the input and target batches.
        model (torch.nn.Module): The model to evaluate on the data.
        device (str): The device to run the model and computations on (e.g., 'cpu', 'cuda').
        num_batches (int, optional): The number of batches to process. If None, processes all batches in the data loader.

    Returns:
        float: The average loss over the specified number of batches.
    """

    model.eval()
    total_loss = 0.0
    num_examples = 0

    if len(data_loader) == 0:
        return float("nan")

    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)
                logits_flat = logits.view(-1, logits.size(-1))
                targets_flat = target_batch.view(-1)

                loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)

            total_loss += loss.item() * input_batch.size(0)
            num_examples += input_batch.size(0)

        else:
            break

    return total_loss / num_examples

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
    logits = model(input_batch)[:, -1, :]
    loss = torch.nn.functional.cross_entropy(logits, target_batch)

    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """
    Calculates the accuracy of a model on a given data loader for a specified number of batches.

    Args:
        data_loader (torch.utils.data.DataLoader): The data loader containing the dataset to evaluate.
        model (torch.nn.Module): The model to evaluate.
        device (str or torch.device): The device on which the model and data are stored ('cpu' or 'cuda').
        num_batches (int, optional): The number of batches to evaluate. If None, evaluates all batches in the loader.

    Returns:
        float: The accuracy of the model on the given data loader for the specified number of batches.
              Returns `nan` if the data loader is empty.
    """

    model.eval()
    correct_predictions, num_examples = 0, 0

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
                logits = model(input_batch)[:, -1, :]

            predicted_labels = torch.argmax(logits, dim=-1)
            num_examples += predicted_labels.shape[0]

            correct_predictions += (predicted_labels == target_batch).sum().item()

        else:
            break

    return correct_predictions / num_examples

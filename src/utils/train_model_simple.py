import torch

from utils.generate_text import (
    generate_text_simple,
    text_to_token_ids,
    token_ids_to_text,
)
from utils.calc_loss import calc_loss_loader, calc_loss_batch


def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
):
    """
    Train a LLM using simple training loop.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        optimizer (torch.optim.Optimizer): Optimizer used for updating the model parameters.
        device (str): Device to run the computations on, e.g., 'cpu' or 'cuda'.
        num_epochs (int): Number of epochs to train the model.
        eval_freq (int): Frequency of evaluating the model during training in terms of global steps.
        eval_iter (int): Number of iterations per validation evaluation.
        start_context (str): Context string from which to generate samples at the end of each epoch.
        tokenizer (transformers.Tokenizer): Tokenizer used for tokenizing input and target batches.

    Returns:
        tuple: A tuple containing three lists:
            - train_losses (list): List of training losses per evaluation frequency.
            - val_losses (list): List of validation losses per evaluation frequency.
            - track_tokens_seen (list): List of tokens seen during training per evaluation frequency.
    """
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):

        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                print(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}"
                )

        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """
    Evaluates the model on both training and validation datasets.

    This function computes the loss for both the training and validation datasets
    over a specified number of batches. The model is temporarily set to evaluation mode,
    which may affect certain layers like dropout and batch normalization.

    Parameters:
    - model (torch.nn.Module): The machine learning model to be evaluated.
    - train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
    - val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
    - device (str or torch.device): The device on which the computation should occur, e.g., 'cpu' or 'cuda'.
    - eval_iter (int): The number of batches to evaluate for each dataset.

    Returns:
    - train_loss (float): The average loss over the specified number of training batches.
    - val_loss (float): The average loss over the specified number of validation batches.
    """

    model.eval()

    with torch.no_grad():

        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)

        model.train()

        return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    """
    Generates and prints sample text from a trained model.

    Args:
        model: The pre-trained model to use for generation.
        tokenizer: The tokenizer used to convert text to token IDs.
        device: The device (e.g., 'cpu' or 'cuda') on which the model is located.
        start_context: The starting context as a string from which to generate text.

    Returns:
        None
    """
    model.eval()

    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)

    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded, max_new_tokens=50, context_size=context_size
        )

        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))

        model.train()

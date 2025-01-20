import torch

from utils.loss import calc_loss_batch, calc_loss_loader


def train_classifier(
    model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter
):
    """
    Train a classifier using the provided data loaders and optimization settings.

    Parameters:
        model (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        optimizer (torch.optim.Optimizer): The optimizer used for updating the model parameters.
        device (str): The device ('cpu' or 'cuda') on which the model and data should be trained.
        num_epochs (int): Number of epochs to train the model.
        eval_freq (int): Frequency of evaluation during training in terms of global steps.
        eval_iter (int): Number of batches to evaluate on during each epoch.

    Returns:
        tuple: A tuple containing four lists - train_losses, val_losses, train_accs, and val_accs,
               representing the loss and accuracy values for both the training and validation sets
               across all epochs, as well as the total number of examples seen.
    """

    checkpoint_path = "fine-tuning-classification/model/weights/model_and_optimizer.pth"

    try:
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Checkpoint loaded successfully from {checkpoint_path}")

    except FileNotFoundError:
        print(
            f"No checkpoint found at {checkpoint_path}. Starting training from scratch."
        )

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()

            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            examples_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                print(
                    f"Ep {epoch + 1} (Step {global_step:06d}):"
                    f"Train loss {train_loss:.3f},"
                    f"Val loss {val_loss:.3f}"
                )

        train_accuracy = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_accuracy = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
        print(f"Training Accuracy: {train_accuracy}")
        print(f"Validation Accuracy: {val_accuracy}")
        val_accs.append(val_accuracy)

        # Save the weights of the model and optimizer after each epoch
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            checkpoint_path,
        )

    return train_losses, val_losses, train_accs, val_accs, examples_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """
    Evaluate the performance of a classification model on both training and validation datasets.

    Args:
        model (torch.nn.Module): The trained model to be evaluated.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        device (str): The device ('cpu' or 'cuda') on which the model and data will be operated.
        eval_iter (int): Number of batches to evaluate per epoch.

    Returns:
        tuple: A tuple containing the average training loss and the average validation loss.
    """

    model.eval()

    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)

    model.train()

    return train_loss, val_loss

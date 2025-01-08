import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    """
    Plots the training and validation loss over a series of epochs.

    Parameters:
        epochs_seen (list): A list of integers representing the number of epochs seen.
        tokens_seen (list): A list of integers representing the number of tokens processed so far.
        train_losses (list): A list of floats representing the training loss at each epoch.
        val_losses (list): A list of floats representing the validation loss at each epoch.

    Returns:
        None
    """

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()
    plt.show()

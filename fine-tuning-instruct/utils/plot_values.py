import matplotlib.pyplot as plt


def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    """
    Plots the training and validation values over epochs and examples seen.

    Parameters:
        epochs_seen (list of int): List of epoch numbers at which values were recorded.
        examples_seen (list of int): List of example counts at which values were recorded.
        train_values (list of float): List of training values corresponding to each epoch.
        val_values (list of float): List of validation values corresponding to each epoch.
        label (str, optional): The label for the plot. Defaults to "loss".
    """

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")

    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend(loc="upper right")

    ax2 = ax1.twiny()
    ax2.plot(examples_seen, train_values, alpha=0)
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()
    plt.savefig(f"fine-tuning-classification/data/{label}-plot.pdf")
    plt.show()

import matplotlib.pyplot as plt


def plot_loss_curve(trainer, save_path):
    # Creating a figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plotting train loss
    ax.plot(range(len(trainer.model.train_epoch_loss)), trainer.model.train_epoch_loss, label='Train Loss',
            color='blue')

    # Plotting validation loss
    valid_epoch_loss = trainer.model.valid_epoch_loss[:len(trainer.model.train_epoch_loss)]
    ax.plot(range(len(valid_epoch_loss)), valid_epoch_loss, label='Validation Loss',
            color='red')

    ax.set_title("Training and Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()

    # Saving the plot
    plt.savefig(f"{save_path}/combined_loss_curve.png")
    plt.clf()  # Clear the current figure after saving


def plot_histogram(y_true, y_pred, save_path):
    plt.figure(figsize=(13, 5))

    plt.title("Histogram")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.xticks(range(0, 100, 5))
    plt.hist(y_true, bins=50, alpha=0.5, label='Test', color='blue')
    plt.hist(y_pred, bins=50, alpha=0.5, label='Pred', color='red')

    plt.legend(loc='upper left')

    # Saving the plot
    plt.savefig(f"{save_path}/score_frequency_histogram.png")

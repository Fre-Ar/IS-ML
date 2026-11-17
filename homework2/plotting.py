import numpy as np
import matplotlib.pyplot as plt
from data import CLASS_NAMES
from training import Trainer

FOLDER = "homework2/figures/"
def plot_loss_curves(trainer: Trainer, save_path: str = FOLDER+"loss_curves.png"):
    """
    Plot training and validation loss over epochs and report final validation MSE.
    """
    train_loss = trainer.history["train_loss"]
    val_loss = trainer.history["val_loss"]
    epochs = np.arange(1, len(train_loss) + 1)

    best_epoch = int(np.argmin(val_loss)) + 1
    best_val_mse = trainer.history["val_mse"][best_epoch - 1]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, train_loss, label=f"Train Loss ({trainer.loss_str.upper()})", linewidth=2)
    ax.plot(epochs, val_loss, label=f"Validation Loss ({trainer.loss_str.upper()})", linewidth=2)

    if trainer.early_stopping:
        ax.axvline(
            best_epoch,
            color="gray",
            linestyle="--",
            alpha=0.8,
            label=f"Restored Checkpoint (Epoch {best_epoch})",
        )

    ax.set_title("Training vs. Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"Loss ({trainer.loss_str.upper()})")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"Final Validation MSE (pixel-wise in [0, 1]): {best_val_mse:.6f}")


def plot_reconstructions_per_category(
    model, val_x, val_y, save_path: str = FOLDER+"reconstructions.png"
):
    """
    Plot 1 original vs. reconstructed validation image for each of the 10 categories.
    """
    fig, axes = plt.subplots(2, 10, figsize=(16, 3.8))

    for c in range(10):
        # Pick the first validation sample belonging to category c
        idx = np.where(val_y == c)[0][0]
        orig = val_x[idx : idx + 1]
        recon = model.forward(orig)
        sample_mse = float(np.mean((orig - recon) ** 2))

        # Top row: Original
        axes[0, c].imshow(orig.reshape(28, 28), cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, c].set_title(CLASS_NAMES[c], fontsize=8.5)
        axes[0, c].axis("off")

        # Bottom row: Reconstruction
        axes[1, c].imshow(recon.reshape(28, 28), cmap="gray", vmin=0.0, vmax=1.0)
        axes[1, c].set_title(f"MSE: {sample_mse:.4f}", fontsize=8)
        axes[1, c].axis("off")

    # Add row labels on the far left
    fig.text(0.01, 0.72, "Original", va="center", rotation="vertical", fontsize=11, fontweight="bold")
    fig.text(0.01, 0.28, "Reconstructed", va="center", rotation="vertical", fontsize=11, fontweight="bold")

    plt.suptitle("Validation Set Reconstructions Across Categories", fontsize=13, y=1.02)
    plt.tight_layout(rect=[0.025, 0, 1, 1])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_mse_per_category(
    model, val_x: np.ndarray, val_y: np.ndarray, save_path: str = FOLDER+"mse_per_category.png"
):
    """Compute and plot mean validation MSE for each of the 10 Fashion-MNIST categories."""
    recons = model.forward(val_x)
    # Per-image pixel-wise MSE in [0, 1]
    per_sample_mse = np.mean((val_x - recons) ** 2, axis=1)

    mean_mses = []
    std_mses = []
    for c in range(10):
        cat_mses = per_sample_mse[val_y == c]
        mean_mses.append(float(np.mean(cat_mses)))
        std_mses.append(float(np.std(cat_mses)))

    # Print formatted table for your report
    print(f"{'Category':<18} | {'Mean MSE':<10} | {'Std MSE':<10}")
    print("-" * 44)
    for c in range(10):
        print(f"{CLASS_NAMES[c]:<18} | {mean_mses[c]:<10.6f} | {std_mses[c]:<10.6f}")
    print("-" * 44)
    print(f"{'Overall Val MSE':<18} | {np.mean(per_sample_mse):<10.6f} |")

    # Bar plot
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x_pos = np.arange(10)
    bars = ax.bar(x_pos, mean_mses, color="steelblue", edgecolor="black", alpha=0.85)

    ax.axhline(
        np.mean(per_sample_mse),
        color="crimson",
        linestyle="--",
        linewidth=1.5,
        label=f"Overall Mean MSE ({np.mean(per_sample_mse):.4f})",
    )

    # Annotate exact values above bars
    for bar, val in zip(bars, mean_mses):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.0004,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_ylabel("Mean Squared Error (MSE)")
    ax.set_title("Mean Validation Reconstruction Error per Category")
    ax.set_ylim(0, max(mean_mses) * 1.15)
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
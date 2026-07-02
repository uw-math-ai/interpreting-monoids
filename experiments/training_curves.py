"""Training loss and accuracy curves from a saved checkpoint."""
import argparse
import os

import matplotlib.pyplot as plt

from core.data import get_device
from core.checkpoint import load_model, checkpoint_path, image_dir


def plot_training_curves(checkpoint, P, save_path=None):
    """Plot loss and accuracy curves from a training checkpoint."""
    losses = checkpoint["losses"]
    train_accs = checkpoint["train_accs"]
    eval_accs = checkpoint["eval_accs"]
    eval_epochs = checkpoint["eval_epochs"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(losses)
    ax1.set(xlabel="Epoch", ylabel="Cross-Entropy Loss", title=f"Training Loss (P={P})")
    ax1.grid(True, alpha=0.3)

    ax2.plot(train_accs, linestyle="--", label="Train")
    ax2.plot(eval_epochs, eval_accs, marker="o", label="Eval")
    ax2.set(xlabel="Epoch", ylabel="Accuracy", title="Train vs Eval Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, format="pdf", bbox_inches="tight")
    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot training curves from checkpoint")
    parser.add_argument("--P", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--mlp_dim", type=int, default=512)
    parser.add_argument("--save_dir", type=str, default="experiments")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = get_device() if args.device is None else __import__("torch").device(args.device)
    ckpt = checkpoint_path(args.save_dir, args.P, args.embed_dim, args.n_heads, args.mlp_dim, args.seed)
    _, checkpoint = load_model(ckpt, device)

    img = image_dir(args.P, args.embed_dim, args.n_heads, args.mlp_dim, args.seed)
    save_path = os.path.join(img, "training_curves.pdf")
    plot_training_curves(checkpoint, args.P, save_path)
    print(f"Saved -> {save_path}")


if __name__ == "__main__":
    main()

"""MLP hidden activation analysis per J-class."""
import argparse
import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from core.data import generate_mod_mult_dataset, get_device
from core.checkpoint import load_model, checkpoint_path, image_dir
from core.algebra import jclass_permutation


def extract_mlp_activations(model, P, device):
    """Extract pre-ReLU MLP activations at the '=' position for the full dataset.

    Returns:
        Tensor of shape [P*P, mlp_hidden] with pre-ReLU activations.
    """
    inputs, _ = generate_mod_mult_dataset(P)

    model.eval()
    with torch.no_grad():
        model(inputs.to(device), return_mlp=True)
        # mlp_activations is [B, 3, mlp_hidden]; take '=' position (index 2)
        activations = model.mlp_activations[:, -1, :].cpu()

    return activations


def plot_neuron_heatmaps(activations, P, perm, num_neurons=50, save_path=None):
    """Plot per-neuron heatmaps of MLP activations permuted by J-class."""
    idx = perm["sorted_indices"]
    boundaries = perm["boundaries"]
    tick_positions = perm["tick_positions"]
    tick_labels = perm["tick_labels"]

    num_neurons = min(num_neurons, activations.shape[1])
    cols = 5
    rows = (num_neurons + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows),
                             constrained_layout=True)
    axes = axes.flatten()

    for i in range(num_neurons):
        vals = activations[:, i].numpy()
        mat = vals.reshape(P, P)
        mat_perm = mat[np.ix_(idx, idx)]

        ax = axes[i]
        im = ax.imshow(mat_perm, origin="lower", cmap="Blues", aspect="equal",
                       interpolation="nearest")

        for b in boundaries:
            ax.axhline(b - 0.5, color="red", lw=1.0, ls="--", alpha=0.8)
            ax.axvline(b - 0.5, color="red", lw=1.0, ls="--", alpha=0.8)

        r, c = i // cols, i % cols
        if r == rows - 1:
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45, fontsize=7)
        else:
            ax.set_xticks([])
        if c == 0:
            ax.set_yticks(tick_positions)
            ax.set_yticklabels(tick_labels, fontsize=7)
        else:
            ax.set_yticks([])

        ax.set_title(f"N{i}", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.035, pad=0.01)

    for i in range(num_neurons, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(f"MLP Hidden Neurons permuted by J-class (P={P})", fontsize=16, y=1.002)
    if save_path:
        plt.savefig(save_path, format="svg", bbox_inches="tight")
    # plt.show()  # uncomment for interactive display


def main():
    parser = argparse.ArgumentParser(description="MLP frequency analysis")
    parser.add_argument("--P", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_neurons", type=int, default=50)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--mlp_dim", type=int, default=512)
    parser.add_argument("--save_dir", type=str, default="experiments")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else get_device()

    ckpt = checkpoint_path(args.save_dir, args.P, args.embed_dim, args.n_heads, args.mlp_dim, args.seed)
    model, _ = load_model(ckpt, device)

    print(f"Extracting MLP activations for P={args.P}, seed={args.seed}...")
    activations = extract_mlp_activations(model, args.P, device)
    perm = jclass_permutation(args.P)

    img = image_dir(args.P, args.embed_dim, args.n_heads, args.mlp_dim, args.seed)
    plot_neuron_heatmaps(activations, args.P, perm, args.num_neurons,
                         os.path.join(img, "mlp_neurons.svg"))

    print(f"MLP plots saved to {img}")


if __name__ == "__main__":
    main()

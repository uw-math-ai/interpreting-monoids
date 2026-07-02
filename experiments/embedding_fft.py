"""Block-wise FFT of embedding matrix per J-class."""
import argparse
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from core.data import get_device
from core.checkpoint import load_model, checkpoint_path, image_dir
from core.algebra import algebraic_jclass_order, prime_factors


def compute_fft_by_jclass(model, P):
    """Compute N-D FFT of embeddings per J-class block.

    For each J-class, the embedding block is reshaped into the CRT group
    structure and an N-D FFT is computed. DC component is zeroed out.

    Returns:
        (results, perm) where results is a list of dicts with keys:
            j, count, fft_mag (tensor), shape (tuple of group orders)
    """
    perm = algebraic_jclass_order(P)
    emb = model.embed.weight.detach().cpu()
    emb_sorted = emb[perm["sorted_indices"]]

    results = []
    start = 0

    for j, count in zip(perm["unique_j"], perm["class_counts"]):
        j = int(j)
        count = int(count)
        block = emb_sorted[start:start + count]

        if j == P:
            results.append({"j": j, "count": count, "fft_mag": torch.zeros_like(block), "shape": (1,)})
            start += count
            continue

        # Determine group structure: primes of P/j give the cyclic factor orders
        modulus = P // j
        active_primes = prime_factors(modulus)
        orders = [p - 1 for p in active_primes]
        ndim = len(orders)
        d_model = block.shape[1]

        if ndim >= 1:
            block_nd = block.view(*orders, d_model)
            fft_block = torch.fft.fftn(block_nd, dim=tuple(range(ndim)))
            fft_mag = torch.abs(fft_block)
            # Zero DC component
            dc_idx = tuple([0] * ndim + [slice(None)])
            fft_mag[dc_idx] = 0
            fft_mag = fft_mag.view(count, d_model)
        else:
            fft_mag = torch.abs(block)

        results.append({"j": j, "count": count, "fft_mag": fft_mag, "shape": tuple(orders)})
        start += count

    return results, perm


def plot_fft_heatmaps(fft_results, P, save_path=None):
    """Plot FFT magnitude heatmaps and energy bar charts per J-class."""
    n = len(fft_results)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 10))
    if n == 1:
        axes = axes.reshape(2, 1)

    for idx, res in enumerate(fft_results):
        fft_mag = res["fft_mag"]
        count = res["count"]
        j = res["j"]

        # Row 1: heatmap
        ax_heat = axes[0, idx]
        fft_mag_log = torch.log1p(fft_mag)
        im = ax_heat.imshow(
            fft_mag_log.cpu().numpy(),
            aspect="auto",
            cmap="viridis",
            origin="upper",
            interpolation="nearest",
        )
        ax_heat.set_title(f"J_{j} (N={count})")
        ax_heat.set_xlabel("Embed Dim")
        if idx == 0:
            ax_heat.set_ylabel("Frequency")
        else:
            ax_heat.set_yticks([])

        # Row 2: energy bar chart
        ax_bar = axes[1, idx]
        freq_energy = torch.norm(fft_mag, dim=1).numpy()
        ax_bar.bar(range(count), freq_energy, color="royalblue", edgecolor="black", alpha=0.8)
        ax_bar.set_xlabel("Frequency k")
        if idx == 0:
            ax_bar.set_ylabel("L2 Energy")
        ax_bar.set_xlim(-0.5, count - 0.5)

    fig.colorbar(im, ax=axes[0, :].ravel().tolist(), label="log(1 + Magnitude)",
                 fraction=0.015, pad=0.02)
    plt.suptitle(f"Fourier Analysis of Embeddings by J-class (P={P})", fontsize=16)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="svg", bbox_inches="tight")
    # plt.show()  # uncomment for interactive display


def _find_key_frequencies(spec, rel_thresh=0.5):
    """Find frequency coordinates above rel_thresh * panel max (excluding DC)."""
    spec = spec.copy()
    spec[0, 0] = 0
    threshold = rel_thresh * np.max(spec)
    nrows, ncols = spec.shape
    freqs = [(y, x) for y in range(nrows) for x in range(ncols)
             if spec[y, x] >= threshold]
    freqs.sort(key=lambda c: spec[c[0], c[1]], reverse=True)
    return freqs


def _add_discrete_grid(ax, nrows, ncols):
    """Add cell-boundary gridlines to a 2D frequency plot."""
    ax.set_xticks(np.arange(ncols))
    ax.set_yticks(np.arange(nrows))
    ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8, alpha=0.28)
    ax.tick_params(which="minor", bottom=False, left=False)


def _highlight_frequency(ax, freq, color="red"):
    """Draw a star marker and box around a key frequency cell."""
    y, x = freq
    ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False,
                            edgecolor=color, linewidth=2.5))
    ax.plot(x, y, marker="*", color=color, markersize=11,
            markeredgecolor="white", markeredgewidth=0.8)
    ax.text(x + 0.15, y + 0.25, rf"$({freq[0]},{freq[1]})$",
            color=color, fontsize=10, fontweight="bold", ha="left", va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=1.5))


def plot_fft_2d_frequency_reuse(fft_results, P, save_path=None):
    """Plot 2D FFT spectra for multi-generator J-classes with key frequency highlighting.

    Selects the neuron with the strongest 2D Fourier coefficient across all
    multi-generator J-classes, then plots per-class 2D spectra with peaks marked.
    """
    # Filter to J-classes with exactly 2 cyclic factors (2D FFT)
    multi_gen = [r for r in fft_results if len(r["shape"]) == 2]
    if not multi_gen:
        print("No multi-generator J-classes found for 2D FFT plot")
        return

    # Reshape FFT magnitudes back to 2D for each J-class
    specs_2d = {}
    for r in multi_gen:
        mag = r["fft_mag"].numpy().reshape(*r["shape"], -1)
        mag[0, 0, :] = 0  # zero DC
        specs_2d[r["j"]] = {"mag": mag, "shape": r["shape"]}

    # Pick target neuron: strongest single 2D coefficient across all classes
    max_per_dim = [np.max(s["mag"], axis=(0, 1)) for s in specs_2d.values()]
    combined = np.maximum.reduce(max_per_dim)
    target_neuron = int(np.argmax(combined))

    n = len(multi_gen)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), constrained_layout=True)
    if n == 1:
        axes = [axes]

    thresholds = {2: 0.5}  # default rel_thresh; small classes get lower threshold

    for idx, r in enumerate(multi_gen):
        j = r["j"]
        shape = r["shape"]
        spec = specs_2d[j]["mag"][:, :, target_neuron]
        rel_thresh = 0.1 if shape[0] * shape[1] <= 10 else 0.5

        key_freqs = _find_key_frequencies(spec, rel_thresh=rel_thresh)

        ax = axes[idx]
        ax.imshow(spec, origin="lower", cmap="plasma", aspect="equal",
                  interpolation="nearest")

        orders_str = r" \times ".join([rf"C_{{{o}}}" for o in shape])
        ax.set_title(rf"2D FFT of Neuron {target_neuron} in $J_{{{j}}} \cong {orders_str}$")
        ax.set_xlabel(rf"$k \in \mathbb{{Z}}_{{{shape[1]}}}$")
        ax.set_ylabel(rf"$j \in \mathbb{{Z}}_{{{shape[0]}}}$")
        _add_discrete_grid(ax, shape[0], shape[1])

        for freq in key_freqs:
            _highlight_frequency(ax, freq)

    fig.suptitle(f"2D Fourier Feature Reuse Across J-classes (P={P}, neuron {target_neuron})",
                 fontsize=14)

    if save_path:
        plt.savefig(save_path, format="svg", bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(description="Embedding FFT analysis by J-class")
    parser.add_argument("--P", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--mlp_dim", type=int, default=512)
    parser.add_argument("--save_dir", type=str, default="experiments")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else get_device()

    ckpt = checkpoint_path(args.save_dir, args.P, args.embed_dim, args.n_heads, args.mlp_dim, args.seed)
    model, _ = load_model(ckpt, device)

    print(f"Computing FFT analysis for P={args.P}, seed={args.seed}...")
    fft_results, perm = compute_fft_by_jclass(model, args.P)

    img = image_dir(args.P, args.embed_dim, args.n_heads, args.mlp_dim, args.seed)
    plot_fft_heatmaps(fft_results, args.P, os.path.join(img, "embedding_fft.svg"))
    plot_fft_2d_frequency_reuse(fft_results, args.P, os.path.join(img, "fft_2d_frequency_reuse.svg"))

    print(f"FFT plots saved to {img}")


if __name__ == "__main__":
    main()

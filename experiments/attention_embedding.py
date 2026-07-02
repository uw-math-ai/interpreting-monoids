"""Attention and embedding analysis for trained models."""
import argparse
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from core.data import generate_mod_mult_dataset, get_device
from core.checkpoint import load_model, checkpoint_path, image_dir
from core.algebra import jclass_permutation, algebraic_jclass_order


def plot_attention(model, P, device, save_path=None):
    """Plot raw attention heatmaps from '=' -> 'a' token, per head."""
    inputs, _ = generate_mod_mult_dataset(P)

    model.eval()
    with torch.no_grad():
        h = model.embed(inputs.to(device))
        _, attn_weights = model.attn(
            h, h, h,
            attn_mask=model.causal_mask,
            need_weights=True,
            average_attn_weights=False,
        )

    n_heads = model.n_heads
    QUERY_POS, KEY_POS = 2, 0

    fig, axes = plt.subplots(1, n_heads, figsize=(6 * n_heads, 5))
    if n_heads == 1:
        axes = [axes]

    for head_idx in range(n_heads):
        attn_to_a = attn_weights[:, head_idx, QUERY_POS, KEY_POS].view(P, P).cpu().numpy()
        ax = axes[head_idx]
        im = ax.imshow(attn_to_a, origin="lower", cmap="Blues", aspect="auto", vmin=0)
        ax.set(title=f"Head {head_idx}", xlabel="Input b", ylabel="Input a")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("Attention: '=' -> 'a' (raw order)", fontsize=16)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="svg")
    # plt.show()  # uncomment for interactive display


def plot_attention_jclass(model, P, device, save_path=None):
    """Plot algebraic J-class permuted attention heatmaps."""
    inputs, _ = generate_mod_mult_dataset(P)
    perm = algebraic_jclass_order(P)

    model.eval()
    with torch.no_grad():
        h = model.embed(inputs.to(device))
        _, attn_weights = model.attn(
            h, h, h,
            attn_mask=model.causal_mask,
            need_weights=True,
            average_attn_weights=False,
        )

    n_heads = model.n_heads
    QUERY_POS, KEY_POS = 2, 0
    idx = perm["sorted_indices"]
    boundaries = perm["boundaries"]

    fig, axes = plt.subplots(1, n_heads, figsize=(7 * n_heads, 6))
    if n_heads == 1:
        axes = [axes]

    for head_idx in range(n_heads):
        attn_to_a = attn_weights[:, head_idx, QUERY_POS, KEY_POS].view(P, P).cpu().numpy()
        attn_rearranged = attn_to_a[np.ix_(idx, idx)]

        ax = axes[head_idx]
        im = ax.imshow(attn_rearranged, origin="lower", cmap="Blues", aspect="auto", vmin=0)

        for b in boundaries:
            ax.axhline(b - 0.5, color="red", linewidth=3, linestyle="--", alpha=0.7)
            ax.axvline(b - 0.5, color="red", linewidth=3, linestyle="--", alpha=0.7)

        ax.set_xticks(perm["tick_positions"])
        ax.set_xticklabels(perm["tick_labels"], rotation=45)
        ax.set_yticks(perm["tick_positions"])
        ax.set_yticklabels(perm["tick_labels"])
        ax.set(title=f"Head {head_idx}", xlabel="Input b (J-class)", ylabel="Input a (J-class)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(f"Attention permuted by J-class (P={P})", fontsize=16)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="svg")
    # plt.show()  # uncomment for interactive display


def plot_embedding(model, P, save_path=None):
    """Plot raw and algebraic J-class sorted embedding heatmaps."""
    perm = algebraic_jclass_order(P)
    emb = model.embed.weight.detach().cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    ax1.imshow(emb, aspect="auto")
    ax1.set(title="Embedding (raw order)", xlabel="Dimension", ylabel="Token Index")

    emb_sorted = emb[perm["sorted_indices"]]
    im = ax2.imshow(emb_sorted, aspect="auto", cmap="viridis")
    for b in perm["boundaries"]:
        ax2.axhline(b - 0.5, color="red", linewidth=2, linestyle="--", alpha=0.8)
    ax2.set_yticks(perm["tick_positions"])
    ax2.set_yticklabels(perm["tick_labels"])
    ax2.set(title=f"Embedding sorted by J-class (P={P})", xlabel="Dimension", ylabel="J-class")
    plt.colorbar(im, ax=ax2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="svg")
    # plt.show()  # uncomment for interactive display


def plot_pca(model, P, save_path_2d=None, save_path_3d=None):
    """Plot PCA projections of token embeddings (excluding the '=' token)."""
    emb = model.embed.weight[:P].detach().cpu().numpy()

    pca = PCA(n_components=3)
    proj = pca.fit_transform(emb)
    colors = np.arange(P)

    # 2D PCA
    fig = plt.figure(figsize=(8, 6))
    scatter = plt.scatter(proj[:, 0], proj[:, 1], c=colors, cmap="hsv", alpha=0.8)
    plt.colorbar(scatter, label="Token index")
    plt.title(f"2D PCA of Embeddings (P={P})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, alpha=0.3)
    if save_path_2d:
        plt.savefig(save_path_2d, format="svg")
    # plt.show()  # uncomment for interactive display

    # 3D PCA with J-class coloring
    try:
        import plotly.express as px
        j_classes = np.gcd(np.arange(P), P)
        labels = [f"J_{g}" for g in j_classes]
        fig3d = px.scatter_3d(
            x=proj[:, 0], y=proj[:, 1], z=proj[:, 2],
            color=labels,
            title=f"3D PCA of Embeddings by J-class (P={P})",
        )
        fig3d.update_traces(marker=dict(size=3))
        if save_path_3d:
            fig3d.write_html(save_path_3d)
        fig3d.show()
    except Exception:
        print("plotly/pandas not available, skipping 3D plot")


def main():
    parser = argparse.ArgumentParser(description="Attention & embedding analysis")
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

    img = image_dir(args.P, args.embed_dim, args.n_heads, args.mlp_dim, args.seed)

    print(f"Generating plots for P={args.P}, seed={args.seed}...")
    plot_attention(model, args.P, device, os.path.join(img, "attention_raw.svg"))
    plot_attention_jclass(model, args.P, device, os.path.join(img, "attention_jclass.svg"))
    plot_embedding(model, args.P, os.path.join(img, "embedding_jclass.svg"))
    plot_pca(model, args.P, os.path.join(img, "pca_2d.svg"), os.path.join(img, "pca_3d.html"))

    print(f"Plots saved to {img}")


if __name__ == "__main__":
    main()

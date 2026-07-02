"""Principal angle alignment between OV circuit and J-class embedding subspaces (Figure 4)."""
import argparse
import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from core.data import get_device
from core.checkpoint import load_model, checkpoint_path, image_dir
from core.algebra import algebraic_jclass_order


def get_head_ov(model, head_idx=0):
    """Extract the OV circuit matrix W_OV = W_V^T @ W_O_block^T for one head."""
    attn = model.attn
    d_model = attn.embed_dim
    n_heads = attn.num_heads
    d_head = d_model // n_heads

    _, _, W_v = attn.in_proj_weight.chunk(3, dim=0)
    start = head_idx * d_head
    end = (head_idx + 1) * d_head

    W_V_head = W_v[start:end, :]  # [d_head, d_model]
    W_O_head = attn.out_proj.weight[:, start:end]  # [d_model, d_head]

    W_OV = W_V_head.T @ W_O_head.T  # [d_model, d_model]
    return W_OV.detach()


def compute_principal_angles(model, P, head_idx=0, k=16):
    """Compute principal angle cosines between OV output subspace and J-class embedding subspaces.

    Returns:
        list of dicts with keys: j, size, cosines, r
    """
    perm = algebraic_jclass_order(P)
    emb = model.embed.weight[:P].detach().cpu()
    emb_sorted = emb[perm["sorted_indices"]]

    W_OV = get_head_ov(model, head_idx).cpu()
    U, S, Vh = torch.linalg.svd(W_OV)
    top_output_dirs = U[:, :k]

    # Orthonormalize OV directions
    OV_dirs, _ = torch.linalg.qr(top_output_dirs)

    results = []
    start = 0

    for j, count in zip(perm["unique_j"], perm["class_counts"]):
        j = int(j)
        count = int(count)
        block = emb_sorted[start:start + count].float()
        start += count

        if count <= 1:
            continue

        # Mean-center the embedding block
        emb_centered = block - block.mean(dim=0, keepdim=True)

        # SVD to get embedding subspace directions
        _, S_block, Vh_block = torch.linalg.svd(emb_centered, full_matrices=False)
        r = min(k, Vh_block.shape[0])
        if r == 0:
            continue

        J_dirs = Vh_block[:r, :].T  # [d_model, r]
        J_dirs, _ = torch.linalg.qr(J_dirs)

        # Principal angle cosines
        cosines = torch.linalg.svdvals(OV_dirs.T @ J_dirs)
        cosines_np = cosines.detach().cpu().numpy()

        results.append({
            "j": j,
            "size": count,
            "cosines": cosines_np,
            "r": r,
        })

    return results


def plot_principal_angles(results, P, head_idx, save_path=None):
    """Plot principal angle cosines per J-class."""
    plt.figure(figsize=(8, 5))

    for row in results:
        j = row["j"]
        cosines = row["cosines"]
        x = np.arange(1, len(cosines) + 1)
        plt.plot(x, cosines, marker="o", linewidth=1.5, label=rf"$J_{{{j}}}$")

    plt.axhline(y=0.8, color="red", linestyle=":", linewidth=1.5, label="0.8 threshold")

    plt.ylim(0, 1.05)
    plt.xlabel("Principal direction index")
    plt.ylabel("Cosine of principal angle")
    plt.title(
        rf"Principal-Angle Alignment: Head {head_idx} OV Output vs. "
        rf"$\mathcal{{J}}$-class Embedding Subspaces (P={P})"
    )
    plt.legend(title=r"$\mathcal{J}$-class", fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="svg", bbox_inches="tight")

    # Print summary table
    print(f"\nOV Principal Angle Summary (Head {head_idx}, k=top SVD directions)")
    print(f"{'J-class':>10} {'Size':>6} {'r':>4} {'Mean cos':>10} {'#>=0.80':>8}")
    print("-" * 42)
    for row in results:
        cos = row["cosines"]
        print(f"  J_{row['j']:<6} {row['size']:>6} {row['r']:>4} "
              f"{cos.mean():>10.4f} {np.sum(cos >= 0.80):>8}")


def main():
    parser = argparse.ArgumentParser(description="OV principal angles (Figure 4)")
    parser.add_argument("--P", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--head_idx", type=int, default=0)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--mlp_dim", type=int, default=512)
    parser.add_argument("--save_dir", type=str, default="experiments")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else get_device()

    ckpt = checkpoint_path(args.save_dir, args.P, args.embed_dim, args.n_heads, args.mlp_dim, args.seed)
    model, _ = load_model(ckpt, device)

    print(f"Computing OV principal angles for P={args.P}, seed={args.seed}, head={args.head_idx}...")
    results = compute_principal_angles(model, args.P, args.head_idx, args.k)

    img = image_dir(args.P, args.embed_dim, args.n_heads, args.mlp_dim, args.seed)
    plot_principal_angles(results, args.P, args.head_idx,
                         os.path.join(img, f"ov_principal_angles_head{args.head_idx}.svg"))

    print(f"Plot saved to {img}")


if __name__ == "__main__":
    main()

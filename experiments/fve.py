"""Top-K Spectral Energy Concentration (SEC) per J-class."""
import argparse

import numpy as np
import torch

from core.data import get_device
from core.checkpoint import load_model, checkpoint_path
from core.algebra import algebraic_jclass_order, prime_factors


def compute_fve(model, P, top_k=5, k_map=None):
    """Compute top-K spectral energy concentration per J-class.

    Args:
        top_k: default K for all J-classes (used when k_map is None or key missing)
        k_map: optional dict {j: K} for per-J-class K values

    Returns:
        list of dicts per J-class with keys:
            j, count, group_shape, K, SEC, uniform_baseline, enrichment,
            top_freqs, total_energy
    """
    perm = algebraic_jclass_order(P)
    emb = model.embed.weight.detach().cpu()
    emb_sorted = emb[perm["sorted_indices"]]

    results = []
    start = 0

    for j, count in zip(perm["unique_j"], perm["class_counts"]):
        j, count = int(j), int(count)
        block = emb_sorted[start:start + count]

        if count <= 1:
            results.append({
                "j": j, "count": count, "group_shape": None, "K": 0,
                "SEC": float("nan"), "uniform_baseline": float("nan"),
                "enrichment": float("nan"), "top_freqs": [], "total_energy": 0.0,
            })
            start += count
            continue

        modulus = P // j
        group_shape = tuple(p - 1 for p in prime_factors(modulus))

        block_nd = block.view(*group_shape, -1)
        fft_axes = tuple(range(len(group_shape)))
        fft_block = torch.fft.fftn(block_nd, dim=fft_axes)
        energy_grid = torch.sum(torch.abs(fft_block) ** 2, dim=-1)

        dc_idx = (0,) * len(group_shape)
        energy_grid[dc_idx] = 0.0

        energy_np = energy_grid.detach().cpu().numpy()
        flat = energy_np.ravel()
        total_energy = float(flat.sum())
        num_non_dc = len(flat) - 1

        if total_energy <= 0 or num_non_dc <= 0:
            results.append({
                "j": j, "count": count, "group_shape": group_shape, "K": 0,
                "SEC": float("nan"), "uniform_baseline": float("nan"),
                "enrichment": float("nan"), "top_freqs": [], "total_energy": total_energy,
            })
            start += count
            continue

        K = min(k_map.get(j, top_k) if k_map else top_k, num_non_dc)
        top_idx = np.argsort(flat)[::-1][:K]
        top_freqs = [np.unravel_index(i, energy_np.shape) for i in top_idx]
        sec = float(flat[top_idx].sum()) / total_energy
        uniform_baseline = K / num_non_dc

        results.append({
            "j": j, "count": count, "group_shape": group_shape, "K": K,
            "SEC": sec, "uniform_baseline": uniform_baseline,
            "enrichment": sec / uniform_baseline,
            "top_freqs": top_freqs, "total_energy": total_energy,
        })
        start += count

    return results


def print_fve_report(results, P, show_top_freqs_for=1):
    """Print SEC report. Optionally prints top frequencies for one J-class."""
    print("=" * 80)
    print(f"{'J-class':<9} | {'|J|':<5} | {'Freqs (K/Total)':<15} | "
          f"{'SEC (Top K)':<12} | {'Baseline':<10} | {'Enrichment'}")
    print("-" * 80)
    for r in results:
        j_name = f"J_{r['j']}"
        if r["K"] == 0 or np.isnan(r["SEC"]):
            print(f"{j_name:<9} | {r['count']:<5} | {'N/A':<15} | {'N/A':<12} | {'N/A':<10} | N/A")
            continue
        num_non_dc = int(np.prod(r["group_shape"])) - 1
        k_str = f"{r['K']} / {num_non_dc}"
        print(f"{j_name:<9} | {r['count']:<5} | {k_str:<15} | "
              f"{r['SEC']*100:.1f}%{'':<7} | {r['uniform_baseline']*100:.1f}%{'':<5} | "
              f"{r['enrichment']:.1f}x")
    print("=" * 80)

    if show_top_freqs_for is not None:
        r = next((r for r in results if r["j"] == show_top_freqs_for), None)
        if r and r["top_freqs"]:
            print(f"\nTop Frequencies for J_{show_top_freqs_for}:")
            for freq in r["top_freqs"]:
                print(f"  Coord: {freq}")


def main():
    parser = argparse.ArgumentParser(description="Top-K Spectral Energy Concentration per J-class")
    parser.add_argument("--P", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--mlp_dim", type=int, default=512)
    parser.add_argument("--save_dir", type=str, default="experiments")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    ckpt = checkpoint_path(args.save_dir, args.P, args.embed_dim, args.n_heads, args.mlp_dim, args.seed)
    model, _ = load_model(ckpt, device)

    results = compute_fve(model, args.P, top_k=args.top_k)
    print_fve_report(results, args.P)


if __name__ == "__main__":
    main()

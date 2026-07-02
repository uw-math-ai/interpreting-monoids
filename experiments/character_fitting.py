"""Character fitting logit FVE (R^2) per J-class (Table 3)."""
import argparse
import math
import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from core.data import get_device
from core.checkpoint import load_model, checkpoint_path, image_dir
from core.algebra import (
    get_jclass_elements, local_inverse_map, build_multi_log_table,
    character, prime_factors,
)


def normalize_freqs(key_freqs, orders):
    """Normalize frequency specs to tuples matching the group dimension."""
    normalized = []
    for freq in key_freqs:
        if isinstance(freq, int):
            normalized.append((freq,))
        else:
            normalized.append(tuple(freq))
    return normalized


def run_jclass_character_fit(model, P, j_class, left_j, right_j, key_freqs, device):
    """Run multi-character logit fit for a single J-class.

    Samples all prompts from J_left x J_right landing in J_j, constructs
    synthetic logits from group characters, and fits via least squares.

    Returns:
        dict with keys: j_class, r2, coef, ys, ys_pred, and metadata
    """
    log_table, generators, orders, primes = build_multi_log_table(j_class, P)
    key_freq_vecs = normalize_freqs(key_freqs, orders)

    J_a = get_jclass_elements(left_j, P)
    J_b = get_jclass_elements(right_j, P)
    J_d = get_jclass_elements(j_class, P)
    inv_Jd, e_Jd = local_inverse_map(J_d, P)

    # Sample all valid prompts
    prompts = []
    for a in J_a:
        for b in J_b:
            prod = (a * b) % P
            if math.gcd(prod, P) == j_class:
                prompts.append((a, b, prod))

    inputs = torch.tensor(
        [[a, b, P] for (a, b, _) in prompts],
        dtype=torch.long, device=device,
    )

    model.eval()
    with torch.no_grad():
        logits = model(inputs)

    candidate_tokens = torch.tensor(J_d, dtype=torch.long, device=device)
    logits_Jd = logits[:, candidate_tokens]

    X_rows = []
    ys = []

    for i, (a, b, prod) in enumerate(prompts):
        X_vals = []
        y_vals = []

        for c_idx, c in enumerate(J_d):
            c_sharp = inv_Jd[c]
            z = (prod * c_sharp) % P

            features = [
                character(z, fv, orders, log_table) for fv in key_freq_vecs
            ]
            X_vals.append(features)
            y_vals.append(logits_Jd[i, c_idx].item())

        X_vals = np.array(X_vals)
        y_vals = np.array(y_vals)

        # Center per prompt
        X_vals = X_vals - X_vals.mean(axis=0, keepdims=True)
        y_vals = y_vals - y_vals.mean()

        for c_idx in range(len(J_d)):
            X_rows.append(X_vals[c_idx])
            ys.append(y_vals[c_idx])

    X = np.array(X_rows)
    ys = np.array(ys)

    X_design = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(X_design, ys, rcond=None)
    ys_pred = X_design @ coef

    ss_res = np.sum((ys - ys_pred) ** 2)
    ss_tot = np.sum((ys - ys.mean()) ** 2)
    r2 = float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot

    return {
        "j_class": j_class,
        "left_j": left_j,
        "right_j": right_j,
        "key_freqs": key_freqs,
        "orders": orders,
        "r2": r2,
        "coef": coef,
        "ys": ys,
        "ys_pred": ys_pred,
    }


def plot_character_fit(result, save_path=None):
    """Scatter plot of predicted vs actual logits for a single J-class fit."""
    j_class = result["j_class"]
    r2 = result["r2"]
    ys = result["ys"]
    ys_pred = result["ys_pred"]

    plt.figure(figsize=(6, 5))
    plt.scatter(ys_pred, ys, alpha=0.55, s=24, edgecolor="none")

    lo = min(ys_pred.min(), ys.min())
    hi = max(ys_pred.max(), ys.max())
    plt.plot([lo, hi], [lo, hi], linewidth=2, label=rf"$R^2={r2:.3f}$")

    plt.xlabel("Predicted logit from composed characters")
    plt.ylabel("Actual model logit")
    orders_str = r" \times ".join([f"C_{{{o}}}" for o in result["orders"]])
    plt.title(rf"Character Logit Fit: $J_{{{j_class}}} \cong {orders_str}$")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")


# Hardcoded experiment specs for P=165
EXPERIMENT_SPECS_165 = [
    {"j_class": 55, "left_j": 5, "right_j": 11, "key_freqs": [1]},
    {"j_class": 33, "left_j": 3, "right_j": 11, "key_freqs": [1, 2, 3]},
    {"j_class": 15, "left_j": 3, "right_j": 5, "key_freqs": [2, 3, 7, 8]},
    {"j_class": 11, "left_j": 1, "right_j": 11,
     "key_freqs": [(0, 1), (0, 2), (0, 3), (1, 0)]},
    {"j_class": 5, "left_j": 1, "right_j": 5,
     "key_freqs": [(0, 2), (0, 3), (0, 7), (0, 8), (1, 0)]},
    {"j_class": 3, "left_j": 1, "right_j": 3,
     "key_freqs": [(0, 2), (0, 3), (0, 7), (0, 8), (1, 0), (2, 0), (3, 0)]},
    {"j_class": 1, "left_j": 1, "right_j": 1,
     "key_freqs": [(0, 0, 2), (0, 0, 3), (0, 0, 7), (0, 0, 8),
                   (0, 1, 0), (0, 2, 0), (0, 3, 0), (1, 0, 0)]},
]


def run_all_fits(model, P, device):
    """Run character fitting for all J-classes with known key frequencies.

    Currently only P=165 has hardcoded experiment specs from the paper.
    """
    if P != 165:
        print(f"Warning: experiment specs only defined for P=165, got P={P}")
        return {}

    all_results = {}
    for spec in EXPERIMENT_SPECS_165:
        result = run_jclass_character_fit(
            model, P,
            j_class=spec["j_class"],
            left_j=spec["left_j"],
            right_j=spec["right_j"],
            key_freqs=spec["key_freqs"],
            device=device,
        )
        all_results[f"J_{spec['j_class']}"] = result
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Character fitting logit FVE (Table 3)")
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

    print(f"Running character fitting for P={args.P}, seed={args.seed}...")
    all_results = run_all_fits(model, args.P, device)

    if not all_results:
        return

    img = image_dir(args.P, args.embed_dim, args.n_heads, args.mlp_dim, args.seed)
    for name, result in all_results.items():
        plot_character_fit(result, os.path.join(img, f"character_fit_{name}.pdf"))

    # Print summary table (Table 3)
    specs = EXPERIMENT_SPECS_165 if args.P == 165 else []
    print(f"\n{'='*56}")
    print("Character Fitting Summary (Table 3)")
    print(f"{'='*56}")
    print(f"{'Class':<8} | {'Source':<13} | {'Freqs':<22} | {'R^2':>8}")
    print(f"{'-'*56}")
    for spec in specs:
        name = f"J_{spec['j_class']}"
        result = all_results[name]
        source = f"J_{spec['left_j']}xJ_{spec['right_j']}"
        freqs = str(spec["key_freqs"])
        print(f"{name:<8} | {source:<13} | {freqs:<22} | {result['r2']:>8.4f}")
    print(f"{'='*56}")

    print(f"Plots saved to {img}")


if __name__ == "__main__":
    main()

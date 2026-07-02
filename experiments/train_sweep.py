"""Train models for multiple P values and seeds."""
import argparse
import os

from core.data import get_device
from core.checkpoint import checkpoint_path, save_checkpoint
from core.trainer import train_single_run


def run_sweep(Ps, num_runs, epochs=25000, embed_dim=128, n_heads=4, mlp_dim=512,
              lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98), train_frac=0.3,
              eval_frac=0.3, eval_every=100, base_seed=1, save_dir="experiments",
              device=None, force=False):
    """Train models for multiple P values and seeds.

    Returns:
        dict mapping P -> list of run result dicts
    """
    if device is None:
        device = get_device()

    os.makedirs(save_dir, exist_ok=True)
    results = {}

    for P in Ps:
        results[P] = []
        for i in range(num_runs):
            seed = base_seed + i
            save_path = checkpoint_path(save_dir, P, embed_dim, n_heads, mlp_dim, seed)

            if not force and os.path.exists(save_path):
                print(f"\n=== P={P} seed={seed} — SKIPPED (checkpoint exists) ===")
                continue

            print(f"\n=== P={P} seed={seed} ===")
            result = train_single_run(
                P=P, embed_dim=embed_dim, n_heads=n_heads, mlp_dim=mlp_dim,
                epochs=epochs, lr=lr, weight_decay=weight_decay, betas=betas,
                train_frac=train_frac, eval_frac=eval_frac, eval_every=eval_every,
                seed=seed, device=device,
            )
            save_checkpoint(result, P, embed_dim, n_heads, mlp_dim, seed, save_path)
            results[P].append(result)

    return results


def main():
    parser = argparse.ArgumentParser(description="Train models for multiple P values and seeds")
    parser.add_argument("--Ps", type=int, nargs="+", required=True, help="List of moduli")
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=25000)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--mlp_dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1.0)
    parser.add_argument("--betas", type=float, nargs=2, default=[0.9, 0.98])
    parser.add_argument("--train_frac", type=float, default=0.3)
    parser.add_argument("--eval_frac", type=float, default=0.3)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--base_seed", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="experiments")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    import torch
    device = torch.device(args.device) if args.device else None

    run_sweep(
        Ps=args.Ps, num_runs=args.num_runs, epochs=args.epochs,
        embed_dim=args.embed_dim, n_heads=args.n_heads, mlp_dim=args.mlp_dim,
        lr=args.lr, weight_decay=args.weight_decay, betas=tuple(args.betas),
        train_frac=args.train_frac, eval_frac=args.eval_frac,
        eval_every=args.eval_every, base_seed=args.base_seed,
        save_dir=args.save_dir, device=device, force=args.force,
    )


if __name__ == "__main__":
    main()

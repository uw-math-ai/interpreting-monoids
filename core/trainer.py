import argparse
import os
import random

import torch
import torch.nn as nn

from core.model import ModMultDecoderOnly
from core.data import generate_mod_mult_dataset, split_dataset, get_device
from core.checkpoint import checkpoint_path, save_checkpoint


def train_single_run(P, embed_dim, n_heads, mlp_dim, epochs, lr, weight_decay, betas,
                     train_frac, eval_frac, eval_every, seed, device):
    """Train a single model with full determinism.

    Returns:
        dict with:
            model: trained ModMultDecoderOnly
            losses: list of length `epochs` — per-epoch training cross-entropy loss
            train_accs: list of length `epochs` — per-epoch training accuracy
            eval_accs: list of length `epochs // eval_every` — eval accuracy sampled every eval_every epochs
            eval_epochs: list of length `epochs // eval_every` — epoch numbers matching eval_accs
    """
    torch.manual_seed(seed)
    random.seed(seed)

    inputs, targets = generate_mod_mult_dataset(P)
    splits = split_dataset(inputs, targets, train_frac, eval_frac, seed=seed, device=device)

    model = ModMultDecoderOnly(P, embed_dim, n_heads, mlp_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    loss_fn = nn.CrossEntropyLoss()

    losses, train_accs, eval_accs, eval_epochs = [], [], [], []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        logits = model(splits["train_inputs"])
        loss = loss_fn(logits, splits["train_targets"])

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        train_accs.append((logits.argmax(dim=-1) == splits["train_targets"]).float().mean().item())

        if (epoch + 1) % eval_every == 0:
            model.eval()
            with torch.no_grad():
                eval_logits = model(splits["eval_inputs"])
                acc = (eval_logits.argmax(dim=-1) == splits["eval_targets"]).float().mean().item()
            eval_accs.append(acc)
            eval_epochs.append(epoch + 1)
            print(
                f"[Seed {seed}] Epoch {epoch+1} | "
                f"Loss {loss.item():.4f} | "
                f"Train acc {train_accs[-1]:.3f} | "
                f"Eval acc {acc:.3f}"
            )

    return {
        "model": model,
        "losses": losses,
        "train_accs": train_accs,
        "eval_accs": eval_accs,
        "eval_epochs": eval_epochs,
    }


def main():
    parser = argparse.ArgumentParser(description="Train ModMultDecoderOnly on modular multiplication")
    parser.add_argument("--P", type=int, required=True, help="Modulus (required)")
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
    parser.add_argument("--seed", type=int, default=1, help="Base seed")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of runs (seeds: seed..seed+N-1)")
    parser.add_argument("--save_dir", type=str, default="experiments")
    parser.add_argument("--device", type=str, default=None, help="Device (auto-detect if omitted)")
    parser.add_argument("--force", action="store_true", help="Retrain even if matching checkpoint exists")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    print(f"Device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    for run in range(args.num_runs):
        seed = args.seed + run
        save_path = checkpoint_path(args.save_dir, args.P, args.embed_dim, args.n_heads, args.mlp_dim, seed)

        if not args.force and os.path.exists(save_path):
            print(f"\n=== Run {run + 1}/{args.num_runs} (seed={seed}) — SKIPPED (checkpoint exists: {save_path}) ===")
            continue

        print(f"\n=== Run {run + 1}/{args.num_runs} (seed={seed}) ===")

        result = train_single_run(
            P=args.P, embed_dim=args.embed_dim, n_heads=args.n_heads, mlp_dim=args.mlp_dim,
            epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, betas=tuple(args.betas),
            train_frac=args.train_frac, eval_frac=args.eval_frac, eval_every=args.eval_every,
            seed=seed, device=device,
        )

        save_checkpoint(result, args.P, args.embed_dim, args.n_heads, args.mlp_dim, seed, save_path)


if __name__ == "__main__":
    main()

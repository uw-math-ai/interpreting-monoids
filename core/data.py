import torch
import random


def generate_mod_mult_dataset(P):
    """Generate the full modular multiplication dataset for a given modulus P.

    Produces all (a, b) pairs in [0, P) and their targets (a * b) % P.

    Args:
        P: the modulus

    Returns:
        inputs: [P*P, 3] tensor of [a, b, P] token sequences
        targets: [P*P] tensor of (a * b) % P
    """
    inputs = torch.tensor(
        [[a, b, P] for a in range(P) for b in range(P)],
        dtype=torch.long,
    )
    targets = torch.tensor(
        [(a * b) % P for a in range(P) for b in range(P)],
        dtype=torch.long,
    )
    return inputs, targets


def split_dataset(inputs, targets, train_frac=0.3, eval_frac=0.3, seed=1, device=None):
    """Split dataset into train/eval sets with deterministic seeding.

    Data beyond train_frac + eval_frac is discarded (e.g. with defaults 0.3+0.3,
    40% of examples are dropped).

    Args:
        inputs: [N, 3] tensor
        targets: [N] tensor
        train_frac: fraction of N to use for training
        eval_frac: fraction of N to use for evaluation
        seed: random seed for reproducible splits
        device: if provided, move all output tensors to this device

    Returns:
        dict with keys: train_inputs, train_targets, eval_inputs, eval_targets
    """
    torch.manual_seed(seed)
    random.seed(seed)

    N = len(inputs)
    train_size = int(train_frac * N)
    eval_size = int(eval_frac * N)

    perm = torch.randperm(N)

    train_idx = perm[:train_size]
    eval_idx = perm[train_size:train_size + eval_size]

    result = {
        "train_inputs": inputs[train_idx],
        "train_targets": targets[train_idx],
        "eval_inputs": inputs[eval_idx],
        "eval_targets": targets[eval_idx],
    }

    if device is not None:
        for key in result:
            result[key] = result[key].to(device)

    return result


def get_device():
    """Auto-detect the best available device (cuda > mps > cpu)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

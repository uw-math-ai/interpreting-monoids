import os

import torch

from core.model import ModMultDecoderOnly


def experiment_name(P, embed_dim, n_heads, mlp_dim, seed):
    return f"P{P}_d{embed_dim}_h{n_heads}_mlp{mlp_dim}_s{seed}"


def checkpoint_path(save_dir, P, embed_dim, n_heads, mlp_dim, seed):
    """Build checkpoint filepath."""
    name = experiment_name(P, embed_dim, n_heads, mlp_dim, seed)
    return os.path.join(save_dir, f"{name}.pt")


def image_dir(P, embed_dim, n_heads, mlp_dim, seed, base_dir="images"):
    """Build image output directory for an experiment."""
    name = experiment_name(P, embed_dim, n_heads, mlp_dim, seed)
    path = os.path.join(base_dir, name)
    os.makedirs(path, exist_ok=True)
    return path


def save_checkpoint(run_result, P, embed_dim, n_heads, mlp_dim, seed, save_path):
    """Save model checkpoint with config and training history.

    Checkpoint keys: model_state_dict, seed, losses, train_accs, eval_accs, eval_epochs,
    and config dict with keys: P, embed_dim, num_heads, mlp_dim.
    Note: config uses 'embed_dim'/'num_heads'/'mlp_dim' which map to the constructor
    params d_model/n_heads/mlp_hidden.
    """
    checkpoint = {
        "model_state_dict": run_result["model"].state_dict(),
        "config": {
            "P": P,
            "embed_dim": embed_dim,
            "num_heads": n_heads,
            "mlp_dim": mlp_dim,
        },
        "seed": seed,
        "losses": run_result["losses"],
        "train_accs": run_result["train_accs"],
        "eval_accs": run_result["eval_accs"],
        "eval_epochs": run_result["eval_epochs"],
    }
    torch.save(checkpoint, save_path)
    print(f"Saved -> {save_path}")


def load_model(path, device):
    """Load a model from a checkpoint file.

    Returns:
        (model, checkpoint) where model is in eval mode and checkpoint is the full
        dict containing losses, train_accs, eval_accs, eval_epochs, seed, and config.
    """
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint["config"]

    model = ModMultDecoderOnly(
        config["P"], config["embed_dim"], config["num_heads"], config["mlp_dim"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint

# mod-mult-interp

Training and analyzing decoder-only transformers that learn modular multiplication: **a * b = c (mod P)**.

The model is a single-layer decoder-only transformer (embedding → causal multihead attention → MLP → unembed). We study how it internalizes algebraic structure — J-classes, cyclic groups, and monoid decompositions — through embedding geometry and attention patterns.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Training

Train a single model:

```bash
uv run python -m core.trainer --P 165
```

Train multiple seeds or P values:

```bash
uv run python -m experiments.train_sweep --Ps 165 --num_runs 5
```

### Key flags

| Flag | Default | Description |
|---|---|---|
| `--P` / `--Ps` | *required* | Modulus (single or list) |
| `--epochs` | 25000 | Training epochs |
| `--embed_dim` | 128 | Embedding dimension |
| `--n_heads` | 4 | Number of attention heads |
| `--mlp_dim` | 512 | MLP hidden width |
| `--lr` | 1e-3 | Learning rate |
| `--weight_decay` | 1.0 | AdamW weight decay |
| `--train_frac` | 0.3 | Fraction of data for training |
| `--eval_frac` | 0.3 | Fraction of data for evaluation |
| `--seed` | 1 | Base random seed |
| `--num_runs` | 1 | Number of runs with consecutive seeds |
| `--save_dir` | `experiments` | Checkpoint directory |
| `--force` | off | Retrain even if checkpoint exists |

### Checkpoint naming

Checkpoints are saved to `experiments/` as `P{P}_d{embed_dim}_h{n_heads}_mlp{mlp_dim}_s{seed}.pt`:

```
experiments/P165_d128_h4_mlp512_s1.pt
```

If a matching checkpoint exists, training is skipped. Use `--force` to retrain.

## Analysis scripts

All scripts are importable as modules and runnable as CLI. Plots are saved as SVGs to `images/P{P}_d{embed_dim}_h{n_heads}_mlp{mlp_dim}_s{seed}/`.

P=165 (= 3 × 5 × 11, squarefree) is the primary reference modulus. Scripts marked *(squarefree P only)* require P to have no repeated prime factors.

---

**Attention heatmaps and embedding PCA**
```bash
uv run python -m experiments.attention_embedding --P 165 --seed 1
```
- `attention_raw.svg` — per-head attention from the "=" token to "a", indexed by raw token order
- `attention_jclass.svg` — same heatmaps with rows and columns permuted by algebraic J-class ordering
- `embedding_jclass.svg` — token embedding matrix sorted by J-class, revealing learned block structure
- `pca_2d.svg` — 2D PCA projection of all token embeddings colored by token index
- `pca_3d.html` — interactive 3D PCA scatter of token embeddings colored by J-class (requires plotly)

---

**Block-wise embedding FFT per J-class** *(squarefree P only)*
```bash
uv run python -m experiments.embedding_fft --P 165 --seed 1
```
- `embedding_fft.svg` — Fourier amplitude heatmap per J-class block: embedding dimensions vs frequency indices
- `fft_2d_frequency_reuse.svg` — 2D FFT spectra for multi-generator J-classes showing which frequency coordinates carry the most energy

---

**PCA components vs random baseline** *(squarefree P only)*
```bash
uv run python -m experiments.pca_fve --P 165 --seed 1
```
- `pca_fve.svg` — PCA components needed for 95% variance explained: J-class subspaces (solid) vs random subsets of the same size ± 1 std

---

**OV principal angle alignment** *(squarefree P only)*
```bash
uv run python -m experiments.ov_principal_angles --P 165 --seed 1
```
- `ov_principal_angles_head0.svg` — cosines of principal angles between head 0's OV output subspace and each J-class embedding subspace

---

**Top-K spectral energy concentration** *(squarefree P only)*
```bash
uv run python -m experiments.fve --P 165 --seed 1
```
Prints a table (no SVG): top-K SEC fraction, uniform baseline, and enrichment factor per J-class.

---

**Character fitting logit R²** *(P=165 only)*
```bash
uv run python -m experiments.character_fitting --P 165 --seed 1
```
- `character_fit_J_{j}.svg` (7 plots) — predicted vs actual logits for each J-class group character fit
- Prints R² per J-class (Table 3 from the paper)

---

**MLP hidden neuron activations**
```bash
uv run python -m experiments.mlp_frequencies --P 165 --seed 1 --num_neurons 50
```
- `mlp_neurons.svg` — pre-ReLU MLP activation heatmaps for top neurons sorted by J-class

---

## Using the model in code

```python
from core import ModMultDecoderOnly, generate_mod_mult_dataset, get_device, load_model

# Create a new model
model = ModMultDecoderOnly(P=165, d_model=128, n_heads=4, mlp_hidden=512)

# Load a trained model
model, checkpoint = load_model("experiments/P165_d128_h4_mlp512_s1.pt", device=get_device())

# Generate dataset
inputs, targets = generate_mod_mult_dataset(P=165)

# Forward pass
logits = model(inputs[:8])                          # [B, P] logits
logits, attn = model(inputs[:8], return_attn=True)  # also returns [B, heads, 3, 3] attention weights
```

## Reproducibility note

Trained weights are **device-dependent**: floating-point non-determinism across hardware means models trained on different devices (CPU, MPS, CUDA) will converge to different weight values even with identical seeds and hyperparameters. The weights and figures in the paper were produced on a **T4 GPU via Google Colab**. Example notebooks are in the `notebooks/` folder for reference.

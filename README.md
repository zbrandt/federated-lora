# Federated LoRA

Federated LoRA is a Python harness for benchmarking federated, differentially 
private LoRA fine-tuning methods. It fine-tunes a Vision Transformer (ViT) on 
CIFAR-100 across a federation of clients, and reproduces and compares four 
methods from the federated and private LoRA literature:

- **LA-LoRA**
- **DP-LoRA**
- **FFA-LoRA**
- **RoLoRA**

[Federated learning]() trains a shared model across many clients under the 
orchestration of a central server, keeping each client's data decentralized and 
reducing the cost and privacy risk of storing it centrally. [LoRA]() (Low-Rank 
Adaptation) adapts a large pre-trained model to a downstream task without 
retraining all of its weights: it freezes the pre-trained weights and injects 
small, trainable low-rank matrices into each layer, reducing the number of 
trainable parameters. This harness combines the two under 
[differential privacy](), applying per-sample gradient clipping and Gaussian 
noise to bound each client's privacy loss.

## Installation

Use the package manager [uv](https://github.com/astral-sh/uv) to install the 
dependencies and development tools.

```bash
uv sync
```

The project runs on [simulation servers]() with a CUDA-capable GPU available at 
the ICE.

## Usage

Run a method on a task for a given seed.

```bash
# fine-tune ViT-Base on CIFAR-100 with LA-LoRA
python -m federated_lora run --method lalora --task cifar100 --seed 42
```

Each run auto-saves to `results/<method>_<task>_seed<seed>.json`

The CLI exposes `--method`, `--task`, `--seed`, and `--results-dir`. The full 
harness configuration of hyperparameters (client count, global rounds, LoRA 
rank, learning rates, privacy budget, etc.) lives in `federated_lora/config.py`. 

Repeat across seeds to build the mean and standard-deviation bands.

```bash
for seed in 42 43 44; do
    python -m federated_lora run --method lalora --task cifar100 --seed $seed
done
```

Plot accuracy against communication round. One figure per task is written to 
`figures/`, with methods overlaid as mean lines and shaded ±1 
standard-deviation bands.

```bash
python -m federated_lora plot results/ --output-dir figures
```

<!-- To run the full comparison in one step — all four methods across three seeds, preceded by a smoke test and followed by plotting — use the benchmark script:

```bash
bash run_benchmark.sh
``` -->

## Contributing

Pull requests are welcome. For major changes, please open an issue first to 
discuss what you would like to change.

These commands to lint and format ensure high quality code.

```bash
uv run ruff format .
uv run ruff check .
```

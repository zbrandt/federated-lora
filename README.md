# Federated LoRA

Federated LoRA is a Python harness for benchmarking federated, differentially 
private LoRA fine-tuning methods. It fine-tunes either the Swin or Vision 
Transformer on CIFAR-100 and Tiny-ImageNet across a federation of clients, and 
reproduces four methods from the federated and private LoRA literature:

- [LA-LoRA](https://arxiv.org/abs/2602.19926v1)
- [DP-LoRA](https://arxiv.org/abs/2312.17493)
- [FFA-LoRA](https://arxiv.org/abs/2403.12313)
- [RoLoRA](https://arxiv.org/abs/2409.02346)

[Federated learning](https://arxiv.org/abs/1912.04977) trains a shared model 
across many clients under the orchestration of a central server, keeping each 
client's data decentralized and reducing the cost and privacy risk of storing 
it centrally. [LoRA](https://arxiv.org/abs/2106.09685) (Low-Rank Adaptation) 
adapts a large pre-trained model to a downstream task without retraining all of 
its weights: it freezes the pre-trained weights and injects small, trainable 
low-rank matrices into each layer, reducing the number of trainable parameters. 
This harness combines the two under 
[differential privacy](https://en.wikipedia.org/wiki/Differential_privacy), 
applying per-sample gradient clipping and Gaussian noise to bound each client's 
privacy loss.

## Installation

Use the package manager [uv](https://github.com/astral-sh/uv) to install the 
dependencies and development tools.

```bash
uv sync
```

The project runs on simulation servers with a CUDA-capable GPU available at 
the [ICE](https://collab.dvb.bayern/spaces/TUMice).

## Usage

Run a task with a federated fine-tuning method for a given seed.

```bash
# fine-tune ViT-Base on CIFAR-100 with RoLoRA
python experiment.py --task cifar100 --method rolora --seed 42
```

Each run auto-saves to `results/<method>_<task>_seed<seed>.json`

The CLI exposes required flags `--task` and `--method`, while setting batch 
size, target epsilon, learning rates for LoRA matrix A & B and classifier head 
parameters, seed, and output directory are optional.

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

## Authors

- [Zachary Brandt](https://www.github.com/zbrandt)
- [Megan Campbell](https://github.com/mjc180501)
- [Yue Xia](https://github.com/yuexia8)
- [Rawad Bitar](https://github.com/yuexia8)

## Contributing

Pull requests are welcome. For major changes, please open an issue first to 
discuss what you would like to change.

These commands to lint and format ensure high quality code.

```bash
uv run ruff format .
uv run ruff check .
```

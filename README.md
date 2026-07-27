# Federated LoRA

Federated LoRA is a Python harness for working with methods for federated 
machine learning and parameter-efficient fine-tuning using low-rank adaptation.

Federated learning is a machine learning technique in a setting where a 
federation of many clients, e.g., mobile devices, collaboratively train a model 
under the orchestration of a central server, allowing for their training data to 
remain decentralized in the process. Compared to centralized machine learning, 
this technique can mitigate systemic costs and privacy risks from centrally 
storing data.[^1]

Low-Rank Adaptation, or LoRA, enables adaption of large-scale pre-trained models
with general domain capabilites to particular tasks without having to retrain 
all model parameters, i.e., parameter-efficient fine-tuning. LoRA freezes 
pre-trained model weights and injects trainable rank decomposition matrices into 
each layer of the model architecture, reducing the number of trainable 
parameters for downstream tasks.[^2]

This project can simulate approaches to applying LoRA in federated learning for
fine-tuning large language models, including:

- [LaLoRA](la_lora/README.md)
- [FLoRA](flora/README.md)

## Setup

Use the package manager [uv](https://github.com/astral-sh/uv) to install the 
project dependencies.

```bash
uv sync
```

## Usage

Run a federated LoRA method against a GLUE task and seed. Each run 
auto-saves to `results/<method>_<task>_seed<seed>.json`:

``````bash
# fine-tune RoBERTa-base on SST-2 with LA-LoRA (paper defaults: 100 rounds)
python main.py run --method lalora --task sst2 --seed 42
``````

The `run` subcommand forwards its flags to the selected method. The full 
default configuration (client count, rounds, LoRA rank, learning rates, 
...) lives in `la_lora/config.py`; the CLI currently exposes `--method`, 
`--task`, `--seed`, and `--results-dir`.

Repeat across seeds to build the mean and standard deviation bands the 
plotter expects:

``````bash
for seed in 42 43 44; do
    python main.py run --method lalora --task sst2 --seed $seed
done
``````

Then plot accuracy against round number for every task in the results 
directory. One figure per task is written to `figures/`, with methods 
overlaid as mean lines and shaded ±1 standard deviation bands:

``````bash
python main.py plot results/ --output-dir figures
``````

Before a long run, shrink the configuration in `la_lora/config.py` for a 
quick smoke test that exercises the whole pipeline in seconds — set e.g. 
`rounds = 2`, `num_clients = 4`, `train_split = "train[:1%]"`, run once to 
confirm it trains end to end, then restore the paper defaults.


## Roadmap

The harness implements the *local alternating* half of LA-LoRA: each 
client trains the LoRA `A` and `B` matrices on alternating local steps 
(`la_lora/client.py`), which decouples the coupled `B·A` gradient and is 
the first of the method's two ingredients.[^3] What remains is the 
differential-privacy machinery that makes the alternating update pay off.

- **Per-client gradient clipping and Gaussian noise.** Bound each update's 
  sensitivity by clipping gradients to an `ℓ₂` norm `C`, then add Gaussian 
  noise with a multiplier `σ` set by the privacy budget `(ε, δ)`. This 
  slots into the flagged `# TODO: Add differential privacy` block in 
  `local_update`. Because the updates alternate, only one of `A` or `B` is 
  noised on any given step, so the perturbation stays linear instead of 
  picking up the quadratic `N_B·N_A` cross term that destabilises 
  simultaneous updates.[^3]
- **Low-pass smoothing filter before aggregation.** Apply an optional 
  Gaussian low-pass filter to each client's noised update before it is 
  uploaded, filtering out the high-frequency components of the DP 
  perturbation. This suppresses residual noise variance and steers 
  aggregation toward flatter minima, improving cross-client consistency 
  and generalisation under a strict privacy budget.[^3] It sits between 
  `local_update` and `aggregate`.
- **A privacy configuration surface.** Add `clip_norm`, `noise_multiplier` 
  (or a target `(ε, δ)`), and a `smoothing` toggle to `Config`, exposed on 
  the CLI, so sweeps over the privacy budget are reproducible and land in 
  the run JSON alongside the existing hyperparameters.

## References

[^1]: https://doi.org/10.48550/arXiv.1912.04977
[^2]: https://doi.org/10.48550/arXiv.2106.09685
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

## Contributing

Pull requests are welcome. Please make sure to follow the steps below to open an
issue and make changes.

### Installation

Use the package manager [uv](https://github.com/astral-sh/uv) to install the 
project dependencies and development tools.

```bash
uv sync
```

This creates a `.venv/` and installs the runtime dependencies plus the
`dev` dependency group (including [ruff](https://docs.astral.sh/ruff/)).
No environment variables or external services are required. Datasets and
model weights are pulled from the Hugging Face Hub on first run, so an
internet connection is needed the first time you execute `main.py`.

Confirm your setup by running a quick smoke test — shrink the config in
`la_lora/config.py` (e.g. `rounds = 2`, `num_clients = 4`,
`train_split = "train[:1%]"`) and run once to confirm the pipeline trains
end to end. Afterwards, you can restore the paper defaults.

```bash
python main.py run --method lalora --task sst2 --seed 42
```

### Linting and formatting

Use ruff for both [linting](https://docs.astral.sh/ruff/linter/) and
[formatting](https://docs.astral.sh/ruff/formatter/). The rule set and
style (79-character lines, single quotes, tab indentation) are configured
under `[tool.ruff]` in `pyproject.toml`. Please run both before opening a
pull request so changes stay consistent and don't introduce regressions.

```bash
# Report lint issues
uv run ruff check .

# Auto-fix what can be fixed safely
uv run ruff check --fix .

# Format the code
uv run ruff format .
```

`ruff check .` should report no errors and
`ruff format --check .` should report no changes.

### Tests

Please verify changes manually with the smoke-test run above and, where 
relevant, regenerate a figure to confirm the plotting path still works.

```bash
python main.py plot results/ --output-dir figures
```

## References

[^1]: McMahan, B., et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data." arXiv preprint arXiv:1602.05629, 2016. 
https://arxiv.org/abs/1602.05629
[^2]: Hu, E. J., et al. "LoRA: Low-Rank Adaptation of Large Language Models." 
arXiv preprint arXiv:2106.09685, 2021. https://arxiv.org/abs/2106.09685
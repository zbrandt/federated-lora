# Federated LoRA

Minimal harness for comparing centralized LoRA with a federated variant.

The current first pass implements two methods:

- `fedit`: FedAvg-style averaging of the full LoRA adapter state.
- `ffa`: FFA-LoRA-style training where `lora_A` stays frozen on the client and only `lora_B` is communicated.

## Structure

- `train_lora.py`: the original single-node LoRA baseline.
- `main.py`: command-line entry point for the federated harness.
- `config.py`: dataclass for model, dataset, LoRA, and federation settings.
- `data.py`: dataset loading, tokenization, and client sharding.
- `client.py`: local client training loop and adapter-state handling.
- `aggregate.py`: weighted averaging of client adapter states.
- `server.py`: federated round orchestration.
- `eval.py`: validation loss and perplexity evaluation.

## Requirements

- Python 3.13
- `datasets`
- `peft`
- `transformers`

Install dependencies with your preferred environment manager, or with `uv` if you are using the existing project setup.

## Quick start

Run the federated harness with the default FedIT-style setting:

```bash
python main.py
```

Run the FFA-LoRA variant instead:

```bash
python main.py --method ffa
```

Write the round-by-round JSON output to a file:

```bash
python main.py --method ffa --output results.json
```

## CLI options

The most useful arguments are:

- `--method {fedit,ffa}`: choose the aggregation/training strategy.
- `--rounds`: number of communication rounds.
- `--clients`: number of federated clients.
- `--local-epochs`: local epochs per client per round.
- `--batch-size`: client and eval batch size.
- `--learning-rate`: local optimizer learning rate.
- `--rank`: LoRA rank.
- `--train-split` and `--eval-split`: dataset slices to use.
- `--output`: optional JSON file for logging the run.

## Current defaults

The current harness uses these defaults unless you override them on the command line:

- model: GPT-2
- dataset: WikiText-2 raw
- train split: `train[:1%]`
- eval split: `validation[:1%]`
- clients: 4
- rounds: 2
- local epochs: 1
- batch size: 2
- LoRA rank: 8

## Output

Each run prints a JSON object with:

- the resolved config,
- one entry per communication round,
- training loss,
- validation loss,
- perplexity,
- bytes uploaded in the round,
- cumulative uploaded bytes.

That output is meant to feed later plotting notebooks for perplexity-vs-round and perplexity-vs-communication curves.

<!-- ## Notes on the first pass

This repository is focused on the FedIT vs FFA-LoRA comparison first. HetLoRA, FLoRA, and the broader comparison table stay in the plan document for now.

WikiText-2 is the default dataset because it matches the existing baseline and gives the fastest path from the current single-node script to a federated benchmark. TinyStories is a reasonable alternative later if you want cleaner, faster runs. -->

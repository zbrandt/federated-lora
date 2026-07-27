# LA-LoRA

LA-LoRA (**L**ocal **A**lternative **LoRA**) is a federated machine learning 
method for parameter-efficient fine-tuning with low-rank adaption of large 
language models. Fine-tuning with LoRA in a differentially-private federated 
learning setting is hindered by a fundamental privacy-utility tradeoff: naively
adding differential privacy on top of LoRA degrades accuracy. LA-LoRA identifies 
this to three problems:

1) gradient coupling from updating two low-rank matrices simultaneously,
2) compounded noise amplification under differential privacy, and
3) shaprness of the globale model in the parameter space.

LA-LoRA addresses these problems by implementing local alternating updates, 
i.e., within each round a client updates only the `A` matrix on even-indexed 
steps and the `B` matrix on odd-indexed steps, decoupling the `B * A` gradient,
and, with differential privacy, keeping noise linear since only one matrix 
perturbed per step. An optional Gaussian smoothing fliter is applied to each 
client's update before aggregation, reducing model sharpness.[^1]

This project can simulate LA-LoRA on RoBERTa-base[^2] over the GLUE[^3] 
benchmark, using LoRA adapters injected into the attention `query` and `value` 
projections. 

# Setup

All hyperparameters live in the `Config` dataclass in `config.py`, which 
doubles as the single source of truth for a run. The fields group into a 
federated setup (`num_clients`, `client_sample_rate`, `partition_strategy`, 
`dirichlet_alpha`, `rounds`, `local_steps`, `batch_size`, per-matrix 
learning rates `lr_a` / `lr_b` / `lr_head`) and a LoRA setup (`lora_rank`, 
`lora_alpha`, `lora_dropout`, `target_modules`, `train_classifier_head`). 
The GLUE task is selected with `dataset_task`, and outputs are written under 
`results_dir` unless an explicit `output` path is given.

A subset of these is exposed on the command line; the rest are edited in 
`config.py` directly:

````bash
python la_lora --method lalora --task sst2 --seed 42 --results-dir results
````

| flag             | meaning                                             |
| ---------------- | --------------------------------------------------- |
| `--method`       | run label used for the output filename and legend   |
| `--task`         | GLUE task: `sst2`, `qnli`, `qqp`, or `mnli`         |
| `--seed`         | RNG seed for partitioning, selection, and the bands |
| `--results-dir`  | directory the run JSON is written to                |

## Usage

Run LA-LoRA on a task and seed; the run auto-saves to 
`results/<method>_<task>_seed<seed>.json`:

````bash
python la_lora --method lalora --task sst2 --seed 42
````

Sweep a few seeds so the plotter can draw mean ±1 std bands, then plot from 
the repository root:

````bash
for seed in 42 43 44; do
    python la_lora --method lalora --task sst2 --seed $seed
done
python main.py plot results/ --output-dir figures
````

For a fast end-to-end check, shrink `config.py` (`rounds = 2`, 
`num_clients = 4`, `train_split = "train[:1%]"`), run once, and restore the 
defaults — accuracy will sit near chance, but a clean per-round log 
confirms the alternating update, aggregation, and head round-trip are all 
wired correctly.

## Roadmap

The local alternating update is implemented; the differential-privacy layer 
that gives LA-LoRA its purpose is not yet.

- **Differential privacy in `local_update`.** At the flagged 
  `# TODO: Add differential privacy` step, clip each client's gradients to 
  an `ℓ₂` norm `C` to bound sensitivity, then add Gaussian noise with a 
  multiplier `σ` calibrated to the privacy budget `(ε, δ)`. The alternating 
  schedule already ensures only one matrix is perturbed per step, so this 
  is where the linear-vs-quadratic noise advantage is realised.[^1][^5]
- **The smoothing filter.** Add the optional Gaussian low-pass filter over 
  each client's noised update prior to `aggregate`, removing high-frequency 
  perturbation components to reduce residual variance and flatten the 
  aggregated model.[^1]
- **Privacy hyperparameters.** Extend `Config` with `clip_norm`, 
  `noise_multiplier` / target `(ε, δ)`, and a `smoothing` toggle, expose 
  them on the CLI, and record them in the run JSON so accuracy can be 
  reported against the privacy budget.

## References

[^1]: https://doi.org/10.48550/arXiv.2602.19926
[^2]: https://doi.org/10.48550/arXiv.2106.09685
[^3]: https://doi.org/10.48550/arXiv.1907.11692
[^4]: https://doi.org/10.48550/arXiv.1804.07461
[^5]: https://doi.org/10.1145/2976749.2978318
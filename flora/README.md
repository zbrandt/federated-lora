# FLoRA

FLoRA (**F**ederated **LoRA**) is a federated machine learning method for
parameter-efficient fine-tuning with low-rank adaptation of large language
models. Its motivation is a specific flaw in the obvious way to average LoRA
updates across clients: FedAvg applied directly to the `A`/`B` factors
computes `mean(B_k) @ mean(A_k)`, not `mean(B_k @ A_k)`. Expanding two
clients:

```
(1/2)(B1+B2) @ (1/2)(A1+A2) = 1/4 (B1A1 + B1A2 + B2A1 + B2A2)
```

The cross terms `B1A2` and `B2A1` are pure noise — they pair one client's
down-projection with another client's up-projection, and neither corresponds
to any client's actual local update.

FLoRA removes this exactly rather than approximately: instead of averaging,
the server **concatenates** every client's `A` along the rank dimension and
every client's `B` along the rank dimension. Matrix multiplication
distributes over block-concatenation, so `B_cat @ A_cat` reproduces
`Σ_k scaling · B_k @ A_k` with no cross terms at all. The cost is that the
stacked rank grows with the number of participating clients each round
(`r → r·K`), so this implementation folds the reconstructed update directly
into the frozen base weights and reinitializes a fresh rank-`r` adapter
(matching PEFT's own default LoRA init: Kaiming-uniform `A`, zero `B`) for
the next round to train from — a periodic merge-and-reinit step that keeps
every round's local training at the configured rank.

This project can simulate FLoRA on RoBERTa-base[^1] over the GLUE[^2]
benchmark, using LoRA adapters injected into the attention `query` and
`value` projections, the same setup used by [LA-LoRA](../la_lora/README.md).

## Differential privacy (DP-FLoRA)

Setting `noise_multiplier` in `Config` enables DP-FLoRA: each client trains
under Opacus[^3] per-example DP-SGD (per-sample gradient clipping to
`max_grad_norm`, calibrated Gaussian noise) instead of plain SGD, with a
persistent `PrivacyEngine` kept on each `Client` for its whole lifetime so its
`ε` accumulates correctly across every round it is actually selected for —
not just the current round.

This uses `noise_multiplier` as the primary privacy knob rather than solving
for a target `epsilon` up front (`make_private_with_epsilon`), because in a
federated setting a client doesn't know in advance how many future rounds it
will be selected for, so there is no fixed "total steps" to solve a target
epsilon against. Instead, `epsilon` actually spent is *reported* per round
(`RoundMetrics.epsilon_spent`, the max across that round's selected clients)
via `PrivacyEngine.get_epsilon(delta)`.

FLoRA's stacking aggregation has no averaging step for `A`/`B` — each
client's contribution is reconstructed exactly, not diluted by an average
over `K` clients. This is worth noting against LA-LoRA's DP rationale (linear
vs. quadratic noise from alternating updates): FLoRA's noise-handling
tradeoff is different in kind, since exact reconstruction means DP noise
injected into any one client's `A`/`B` propagates fully into the merged base
weights rather than being averaged down — an open question for how the
per-client privacy budget should be set relative to the number of clients
selected per round.

## Setup

All hyperparameters live in the `Config` dataclass in `config.py`, mirroring
`la_lora`'s layout: a federated setup (`num_clients`, `client_sample_rate`,
`partition_strategy`, `dirichlet_alpha`, `rounds`, `local_steps`,
`batch_size`, a single `lr` since FLoRA trains `A`/`B` jointly rather than
alternately), a LoRA setup (`lora_rank`, `lora_alpha`, `lora_dropout`,
`target_modules`, `train_classifier_head`), and a privacy setup
(`max_grad_norm`, `noise_multiplier`, `delta`).

A subset of these is exposed on the command line; the rest are edited in
`config.py` directly:

````bash
python main.py run --method flora --task sst2 --seed 42
````

| flag                  | meaning                                                    |
| --------------------- | ----------------------------------------------------------|
| `--method`            | run label used for the output filename and legend         |
| `--task`               | GLUE task: `sst2`, `qnli`, `qqp`, or `mnli`               |
| `--seed`              | RNG seed for partitioning, selection, and reinitialization |
| `--results-dir`       | directory the run JSON is written to                      |
| `--noise-multiplier`  | Gaussian noise multiplier for DP-SGD; unset disables DP    |
| `--max-grad-norm`     | per-example gradient clipping norm under DP-SGD            |

## Usage

Run plain FLoRA on a task and seed; the run auto-saves to
`results/<method>_<task>_seed<seed>.json`:

````bash
python main.py run --method flora --task sst2 --seed 42
````

Run DP-FLoRA by setting a noise multiplier:

````bash
python main.py run --method flora --task sst2 --seed 42 --noise-multiplier 1.0
````

Sweep a few seeds so the plotter can draw mean ±1 std bands, then plot from
the repository root (this plots any `la_lora`/`flora` results found in the
same directory together):

````bash
for seed in 42 43 44; do
    python main.py run --method flora --task sst2 --seed $seed
done
python main.py plot results/ --output-dir figures
````

For a fast end-to-end check, shrink `config.py` (`rounds = 2`,
`num_clients = 4`, `train_split = "train[:1%]"`), run once (with and without
`--noise-multiplier`), and restore the defaults — accuracy will sit near
chance, but a clean per-round log confirms the joint update, stacking
aggregation, and merge-and-reinit are all wired correctly.

## Roadmap

- **Heterogeneous per-client ranks.** Concatenation along the rank dimension
  doesn't require every client's `A`/`B` to share the same shape, so
  per-client ranks are a natural extension of `aggregate()` — this
  implementation keeps a single global `lora_rank` in `Config` for parity
  with `la_lora`, so it isn't wired up yet.
- **LA-LoRA's smoothing filter.** The optional Gaussian low-pass filter from
  LA-LoRA's roadmap is specific to that method's alternating-update noise
  profile and isn't applicable here as-is; whether an analogous filter helps
  FLoRA's merge-and-reinit step is an open question.

## References

[^1]: https://doi.org/10.48550/arXiv.2106.09685
[^2]: https://doi.org/10.48550/arXiv.1804.07461
[^3]: https://doi.org/10.48550/arXiv.2109.12298

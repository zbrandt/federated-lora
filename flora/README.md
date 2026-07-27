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

Setting `noise_multiplier` in `Config` enables DP-FLoRA. This is **client-level**
DP-SGD, following Liu et al., "Differentially Private Low-Rank Adaptation of
Large Language Model Using Federated Learning"[^4], Algorithm 1 -- not
Opacus's[^3] per-example DP-SGD. Each local step computes an ordinary
mini-batch gradient, clips it once (not once per example) to `max_grad_norm`
-- LoRA `A`/`B` and the classifier head clipped as two independent groups,
mirroring the paper's separate treatment of `A` and `B` -- adds one Gaussian
noise draw calibrated to `noise_multiplier * max_grad_norm` per group, then
takes a plain SGD step (`dp_lr`, `dp_momentum` -- 0 by default, matching the
paper's plain-SGD update rule). `max_grad_norm` and `dp_lr` default to that
paper's own validated BERT-base hyperparameters (their Table 2); RoBERTa-base
is comparable in scale. Only Opacus's standalone `RDPAccountant` is used (for
reporting `epsilon`), kept per-`Client` for its whole lifetime so spend
composes correctly across every round it's actually selected for -- not the
rest of Opacus's per-example machinery (no `GradSampleModule`, no
`PrivacyEngine.make_private`).

This protects each client's entire per-step update as the unit of privacy,
rather than each individual training example the way Opacus's per-example
DP-SGD does -- a weaker guarantee, but one that doesn't depend on a large
batch size to keep noise-to-signal usable (the noise here is added once to
the already batch-averaged gradient, not summed per-example and divided by
batch size after the fact). Two earlier attempts at wiring up Opacus's
per-example pipeline directly for this codebase (full per-example clipping,
then per-layer clipping, then several rounds of learning-rate/optimizer
tuning) all failed to produce any learning under DP, capping out at exactly
the majority-class baseline; this simpler, actually-published mechanism was
adopted instead of further tuning that pipeline blind.

Plain SGD, not AdamW, is used for the DP path regardless: DP noise adds a
constant bias to Adam's second-moment estimate, which specifically
miscalibrates its adaptive step size for small, low-variance parameters like
a rank-8 LoRA adapter (a documented failure mode of DP-Adam generally, not
specific to this codebase -- see e.g. "DP-AdamBC", arXiv:2312.14334), and
Algorithm 1 in the reference paper uses plain SGD for exactly this reason.

This uses `noise_multiplier` as the primary privacy knob rather than solving
for a target `epsilon` up front, because in a federated setting a client
doesn't know in advance how many future rounds it will be selected for, so
there is no fixed "total steps" to solve a target epsilon against. Instead,
`epsilon` actually spent is *reported* per round (`RoundMetrics.epsilon_spent`,
the max across that round's selected clients) via `RDPAccountant.get_epsilon`.

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
(`max_grad_norm`, `noise_multiplier`, `delta`, `dp_lr`, `dp_momentum`).

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
| `--max-grad-norm`     | gradient clipping norm (once per step, per group) for the DP path |
| `--dp-lr`             | plain SGD learning rate for the DP path                    |
| `--dp-momentum`       | SGD momentum for the DP path (0 matches the reference algorithm) |

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
[^4]: https://doi.org/10.48550/arXiv.2312.17493

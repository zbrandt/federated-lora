# FLoRA

FLoRA (**F**ederated **LoRA**) is a federated machine learning method for
parameter-efficient fine-tuning with low-rank adaptation of large language
models. It uses **FFA-LoRA** (Frozen-A LoRA): the low-rank update
`ΔW = B @ A` keeps `A` frozen, shared, and identical across every client, and
only ever trains `B`.

Freezing `A` fixes two separate problems at once:

**FedAvg's aggregation noise.** Averaging LoRA factors directly across
clients computes `mean(B_k) @ mean(A_k)`, not `mean(B_k @ A_k)`. Expanding
two clients:

```
(1/2)(B1+B2) @ (1/2)(A1+A2) = 1/4 (B1A1 + B1A2 + B2A1 + B2A2)
```

The cross terms `B1A2` and `B2A1` are pure noise — they pair one client's
down-projection with another client's up-projection, and neither corresponds
to any client's actual local update. Once `A` is identical and frozen across
every client, this disappears entirely: `mean(B_k) @ A = mean(B_k @ A)`
exactly, since `A` is a shared constant that factors out of the average. No
stacking or concatenation trick is needed — plain weighted FedAvg on `B` is
already exact.

**DP-SGD's quadratic noise.** If both `A` and `B` are DP-noised, the
reconstructed update is `(B+ηB)(A+ηA) = BA + BηA + ηBA + ηBηA` — noise
enters quadratically (a pure-noise `ηBηA` term, plus two linear-noise cross
terms). With `A` frozen and noise-free, the update is `(B+ηB)A = BA + ηBA`:
noise enters exactly once, linearly. Three earlier attempts at DP-FLoRA with
both `A` and `B` trainable (Opacus per-example DP-SGD with flat clipping,
then per-layer clipping, then a hand-rolled client-level DP-SGD) all left
accuracy pinned at exactly the majority-class baseline regardless of
hyperparameters — this quadratic-noise mechanism is why.

This project can simulate FLoRA on RoBERTa-base[^1] over the GLUE[^2]
benchmark, using LoRA adapters injected into the attention `query` and
`value` projections.

## FFA-LoRA mechanics (`lora_layer.py`)

`HeterogeneousDPLoRALayer` wraps each target `nn.Linear` (query/value, per
transformer layer):

- **`A_max`** — an orthogonal-init, non-persistent *buffer* (not a
  parameter) of shape `(r_max, in_features)`, seeded from `config.seed`.
  Because this simulator keeps one shared physical model that every client
  trains in turn (see `client_computation` in `server/computation.py`),
  "shared across clients" is automatic — there's exactly one `A_max` per
  wrapped layer, never one per client.
- **`B`** — a real `nn.Linear(r_max, out_features, bias=False)`, always
  full `r_max` width, the *only* trainable/noised part of the adapter. It
  has to stay a genuine `nn.Linear` (not a manually-sliced `nn.Parameter`)
  for Opacus's automatic per-example gradient hooks to work at all — those
  hooks only fire for known module types.

**Rank heterogeneity** (`client_ranks` in `Config`): a client assigned rank
`r_i <= r_max` only projects through the first `r_i` rows of `A_max`; the
projection is then zero-padded back to `r_max` width before being fed into
`B`. Since the padding is exactly 0 for every example, the *true* gradient
for `B`'s columns `>= r_i` is exactly 0 too — without ever reshaping `B`
itself (which would break Opacus's hooks and require a differently-shaped
optimizer per client).

**Important subtlety**: that padding trick alone is *not* enough once DP
noise is involved. Opacus adds noise to the *entire* gradient tensor
unconditionally, including the structurally-zero padded columns — left
alone, a low-rank client's "inactive" columns would come back as pure noise,
not zero. `client.py`'s `local_update` explicitly zeros
`B.weight[:, r_i:] = 0` after local training finishes, on every path (DP or
not), before the state is sent to the server.

## Per-client differential privacy (`client.py`)

Setting `client_epsilons[i]` in `Config` enables DP-SGD for client `i`.
Training is Opacus per-example DP-SGD scoped *only* to `B` and the
classifier head (never `A`, which isn't a parameter at all) — flat clipping
across both combined to `max_grad_norm`, **not** per-layer clipping: Opacus's
`DPPerLayerOptimizer` collapses a list of per-layer clip norms into a single
*aggregate* L2 value and uses that much larger number to scale the noise
added to every parameter, silently inflating the injected noise (a real bug
hit and reverted in an earlier iteration of this codebase). Each client's
target `epsilon_i` is converted to a noise multiplier `sigma_i` via Opacus's
`get_noise_multiplier`, calibrated against the *expected* total local steps
across the whole run (`rounds * client_sample_rate * local_steps`) — a
client's actual future participation count isn't known in advance, so this
is an approximation; a client selected more or less than expected will
over/under-spend its nominal budget somewhat. Each client keeps one
`PrivacyEngine` for its whole lifetime so epsilon composes correctly across
every round it's actually selected for.

## Setup

All hyperparameters live in the `Config` dataclass in `config.py`: a
federated setup (`num_clients`, `client_sample_rate`, `partition_strategy`,
`dirichlet_alpha`, `rounds`, `local_steps`, `batch_size`, `lr`), a LoRA setup
(`lora_rank` — this is `r_max`, `lora_alpha`, `lora_dropout`,
`target_modules`, `train_classifier_head`), rank heterogeneity
(`client_ranks: list[int] | None`, exactly `num_clients` entries or `None`
for homogeneous `r_max`), and DP (`client_epsilons: list[float | None] |
None`, exactly `num_clients` entries — per-entry `None` disables DP for that
specific client — or `None` to disable DP entirely; plus `delta`,
`max_grad_norm`, `dp_lr`). `client_ranks`/`client_epsilons` are populated
directly in `config.py`, not exposed on the command line.

A subset of the rest is exposed on the command line; the rest are edited in
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
| `--max-grad-norm`     | per-example clipping norm (flat, across `B` + classifier head) for the DP path |
| `--dp-lr`             | AdamW learning rate for the DP path                        |

## Usage

Run FLoRA on a task and seed (homogeneous rank, no DP unless `client_ranks`/
`client_epsilons` are set in `config.py`); the run auto-saves to
`results/<method>_<task>_seed<seed>.json`:

````bash
python main.py run --method flora --task sst2 --seed 42
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
`num_clients = 4`, `train_split = "train[:1%]"`), run once, and restore the
defaults — accuracy will sit near chance, but a clean per-round log confirms
the local update, FedAvg aggregation, and (if enabled) DP-SGD are all wired
correctly. Recommended order for validating any new configuration: (1)
homogeneous rank, no DP, (2) heterogeneous `client_ranks`, still no DP, (3)
homogeneous rank with DP enabled, (4) both together. If accuracy ever comes
back pinned at exactly the majority-class baseline regardless of
hyperparameters — the exact failure signature every previous DP-FLoRA
attempt hit — that's a strong signal of a bug elsewhere (aggregation, model
loading, eval) rather than a DP-mechanism problem, since freezing `A` was
specifically meant to rule that mechanism out.

## Ablations (`sweep.py`)

`sweep.py` at the repo root runs controlled ablations over rank and DP
epsilon and plots the resulting accuracy trend:

- `rank` — DP held fixed (or off), `lora_rank` (r_max) varied between runs.
- `dp` — rank held fixed, the shared per-client epsilon varied between runs
  (including a `none` no-DP baseline point).
- `grid` — the full rank x epsilon interaction: every combination, not one
  axis at a time. Where `rank`/`dp` can only show the two independent main
  effects, `grid` can reveal whether the accuracy-maximizing rank actually
  *shifts* as epsilon changes (the DP noise-amplification prediction is
  that it should move to smaller ranks as epsilon shrinks). It also exposes
  two flags the other two modes don't need:
  - `--epochs N` sets `Config.local_epochs`, switching local training from a
    fixed step count (`local_steps`, the same for every client) to `N` full
    passes over each selected client's *own* shard — which varies in size
    under non-iid partitioning, so this is not just a renamed `local_steps`.
  - `--max-physical-batch-size N` wraps DP training in Opacus's
    `BatchMemoryManager`, splitting a large logical `--batch-size` (bigger
    batches improve DP privacy amplification) into GPU-memory-safe physical
    sub-batches that are gradient-accumulated before the actual clip+noise+
    step fires once per logical batch. Only meaningful under DP with
    `--epochs` set; ignored otherwise. If wide ranks still OOM, lower this
    further (e.g. 16 → 8) before lowering `--batch-size` itself.

Every sweep point is cached to `results/sweep/<tag>.json` (the tag encodes
rank/epsilon/rounds/num_clients/batch_size/epochs/seed, so runs at different
settings can never silently collide) and skipped on a re-run unless
`--force` is passed — safe to relaunch after a timeout. See `sweep.py`'s own
module docstring (`python sweep.py --help` / `python sweep.py <mode>
--help`) for full usage.

## Roadmap

- **LA-LoRA's smoothing filter.** The optional Gaussian low-pass filter from
  LA-LoRA's roadmap is specific to that method's alternating-update noise
  profile and isn't applicable here as-is.
- **Rank-aware normalized aggregation.** `aggregate()` currently averages
  literal zero-padded columns in with everyone else's, which dilutes
  higher-rank columns proportionally to how many selected clients that
  round have lower rank. A rank-aware normalized average (dividing each
  column by the weight of only the clients whose rank covers it) would
  remove that dilution but isn't implemented.

## References

[^1]: https://doi.org/10.48550/arXiv.2106.09685
[^2]: https://doi.org/10.48550/arXiv.1804.07461

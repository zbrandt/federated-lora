from __future__ import annotations

import argparse
from dataclasses import dataclass, field


GLUE_TASKS = {
    "sst2": (("sentence", None),          2, "validation"),
    "qnli": (("question", "sentence"),    2, "validation"),
    "qqp":  (("question1", "question2"),  2, "validation"),
    "mnli": (("premise", "hypothesis"),   3, "validation_matched")
}


@dataclass(slots=True)
class Config:
    method: str = "flora"               # label only, names the output file and the plot legend
    model_name: str = "roberta-base"    # https://huggingface.co/FacebookAI/roberta-base
    dataset_name: str = "nyu-mll/glue"  # https://huggingface.co/datasets/nyu-mll/glue
    dataset_task: str = "sst2"          # sentences from movie reviews and human annotations of their sentiment
    train_split: str = "train"

    # Federated setup for language understanding
    num_clients: int = 20
    client_sample_rate: float = 0.2
    partition_strategy: str = "noniid"
    dirichlet_alpha: float = 0.8
    rounds: int = 100
    local_steps: int = 20
    local_epochs: int | None = None     # if set, overrides local_steps: train `local_epochs` full passes over
                                         # each selected client's own local shard per round instead of a fixed
                                         # step count. Client shard sizes vary under non-iid partitioning, so
                                         # (unlike local_steps) this gives each client a different number of
                                         # actual optimizer steps per round -- see client.py.
    batch_size: int = 128               # matches the FLoRA paper's effective batch size (arXiv:2409.05976 Appendix
                                         # A.2: "the batch size is 128 and the micro batch size is 16" -- they
                                         # accumulate 8 micro-batches of 16 to reach it; RoBERTa-base is small
                                         # enough that a real batch of 128 is used directly here instead).
    max_length: int = 128
    seed: int = 42
    lr: float = 3e-4                    # AdamW learning rate for the plain (non-DP) path

    # LoRA setup: FFA-LoRA (Frozen-A LoRA, see lora_layer.py). `lora_rank` is
    # r_max -- the shared, frozen `A_max`'s rank. Individual clients may be
    # assigned a smaller active rank via `client_ranks` below; `A` is never
    # trained or noised regardless, only `B` is.
    lora_rank: int = 8                  # r_max
    lora_alpha: int = 8
    lora_dropout: float = 0.1
    target_modules: tuple[str, ...] = ("query", "value")
    train_classifier_head: bool = True

    # Rank heterogeneity: client i trains only its first `client_ranks[i]`
    # rows of the shared A_max (see lora_layer.py). None => every client
    # uses the full r_max (homogeneous). If set, must have exactly
    # `num_clients` entries, each in [1, lora_rank] -- validated in build().
    client_ranks: list[int] | None = None

    # Per-client differential privacy (optional; None => DP disabled
    # entirely, plain FedAvg-LoRA). Each client i with client_epsilons[i] set
    # trains under Opacus per-example DP-SGD, scoped only to B (never A) and
    # the classifier head -- flat clipping (NOT per-layer: Opacus's
    # DPPerLayerOptimizer collapses per-layer clip norms into one aggregate
    # L2 value for noise scaling, which silently inflates the injected noise
    # -- a real bug hit and reverted in an earlier iteration of this file).
    # None => DP disabled. If set, must have exactly `num_clients` entries;
    # per-entry `None` disables DP for that specific client while others
    # keep it enabled.
    client_epsilons: list[float | None] | None = field(default_factory=lambda: [8.0] * 20)   # must match num_clients; None per-entry = that client stays non-private
    delta: float = 1e-5                 # per-example clipping norm (C), flat across  DP accounting target delta
    max_grad_norm: float = 1.0          #B + classifier head
    dp_lr: float = 1e-3                 # AdamW learning rate for the DP path (larger than `lr` to help
                                         # push through the injected noise)
    max_physical_batch_size: int | None = None
                                         # Opacus's BatchMemoryManager: split each DP logical batch (batch_size)
                                         # into physical sub-batches of at most this size, gradient-accumulating
                                         # up to the full logical batch before the actual clip+noise+step. Lets
                                         # batch_size be large -- bigger batches give better privacy amplification
                                         # per Abadi et al. -- without needing enough GPU memory to hold
                                         # per-example gradients for the whole logical batch at once. Only takes
                                         # effect under DP (target_epsilon is not None) with local_epochs set;
                                         # ignored otherwise -- see client.py.

    results_dir: str = "results"        # runs auto-save in this directory as <method>_<task>_seed<seed>.json
    output: str | None = None           # explicit path override, ignores results_dir naming when set

    @property
    def text_fields(self) -> tuple[str, str | None]:
        return GLUE_TASKS[self.dataset_task][0]

    @property
    def num_labels(self) -> int:
        return GLUE_TASKS[self.dataset_task][1]

    @property
    def eval_split(self) -> str:
        return GLUE_TASKS[self.dataset_task][2]

    @property
    def use_dp(self) -> bool:
        """True whenever any client has a target epsilon configured."""
        return self.client_epsilons is not None

    @classmethod
    def from_argv(cls, argv: list[str] | None = None) -> "Config":
        defaults = cls()
        parser = argparse.ArgumentParser(prog="flora")
        parser.add_argument("--method", default=defaults.method,
                            help="run label used for the output filename and plot legend")
        parser.add_argument("--task", choices=list(GLUE_TASKS), default=defaults.dataset_task)
        parser.add_argument("--seed", type=int, default=defaults.seed,
                            help="random number generator seed for mean and standard deviation bands")
        parser.add_argument("--results-dir", default=defaults.results_dir,
                            help="directory where the run output is written to")
        parser.add_argument("--max-grad-norm", type=float, default=defaults.max_grad_norm,
                            help="per-example gradient clipping norm (flat, across B + classifier head) for the DP path")
        parser.add_argument("--dp-lr", type=float, default=defaults.dp_lr,
                            help="AdamW learning rate used for the DP path")
        args = parser.parse_args(argv)
        return cls(
            method=args.method,
            dataset_task=args.task,
            seed=args.seed,
            results_dir=args.results_dir,
            max_grad_norm=args.max_grad_norm,
            dp_lr=args.dp_lr,
        )

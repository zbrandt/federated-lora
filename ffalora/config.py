"""
Sets intitial parameters for the FFA-LoRA run, including model, dataset, and federated learning hyperparameters. Also includes LoRA-specific parameters such as rank, alpha, and dropout. Additionally, it handles differential privacy settings for clients if enabled.
"""

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
    method: str = "ffalora"               # label only, names the output file and the plot legend
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
    local_epochs: int | None = None     # if set, overrides local_steps and runs this many epochs per round instead of a fixed number of steps
    batch_size: int = 128               # matches the FLoRA paper's effective batch size
    max_length: int = 128
    seed: int = 42
    lr: float = 3e-4                    # AdamW learning rate for the plain (non-DP) path

    # LoRA hyperparameters
    lora_rank: int = 8                  # r_max
    lora_alpha: int = 8
    lora_dropout: float = 0.1
    target_modules: tuple[str, ...] = ("query", "value")
    train_classifier_head: bool = True

    # Per-client LoRA rank (optional; None => all clients use lora_rank). Each client i with client_ranks[i] set uses that rank for its B matrix, while A is always frozen at lora_rank. If set, must have exactly `num_clients` entries.
    client_ranks: list[int] | None = None

   # Per-client target epsilon for differential privacy (optional; None => that client is non-private). If set, must have exactly `num_clients` entries. DP is decided per client (client.py: `if target_epsilon is not None`) -- a client with its own entry set to None always takes the plain, non-DP path regardless of what any other client is set to.
    client_epsilons: list[float | None] | None = field(default_factory=lambda: [8.0] * 20)
    delta: float = 1e-5                 # target delta for DP, used to calibrate the noise multiplier. Should be set to 1/(total number of training examples) or smaller. See https://arxiv.org/abs/2006.14799 for discussion of delta and its effect on privacy guarantees.
    max_grad_norm: float = 1.0          # per-example gradient clipping norm (flat, across B + classifier head) for the DP path
    dp_lr: float = 1e-3                 # AdamW learning rate for the DP path
    max_physical_batch_size: int | None = None
                                         # maximum physical batch size for the DP path. If set, overrides the default of min(batch_size, 128) and is used to determine the number of microbatches per physical batch. The effective batch size is still `batch_size`, but the physical batch size may be smaller to fit in GPU memory. If not set, defaults to min(batch_size, 128).

    results_dir: str = "results"        # directory where the run output is written to. The output filename is constructed as {results_dir}/{method}_{dataset_task}_seed{seed}.json unless overridden by `output`.
    output: str | None = None           # optional override for the output filename. If set, the output is written to this path instead of the default constructed path.

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
        parser = argparse.ArgumentParser(prog="ffalora")
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

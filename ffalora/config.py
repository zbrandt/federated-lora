"""
All the settings for one FFA-LoRA run: which model and dataset to use, the
federated learning setup (how many clients, how many rounds), LoRA settings,
and optional differential privacy settings.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field


# Which text columns to read, how many labels, and which split to evaluate on, per GLUE task.
GLUE_TASKS = {
    "sst2": (("sentence", None),          2, "validation"),
    "qnli": (("question", "sentence"),    2, "validation"),
    "qqp":  (("question1", "question2"),  2, "validation"),
    "mnli": (("premise", "hypothesis"),   3, "validation_matched")
}


# All the tunable settings for a run, with sensible defaults.
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
    local_epochs: int | None = None     # if set, train this many local epochs per round instead of a fixed step count
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

    # Per-client LoRA rank (optional; None => every client uses lora_rank)
    client_ranks: list[int] | None = None

    # Per-client DP epsilon (optional; None => no client uses DP). A client with its own entry set
    # to None always skips DP, regardless of what other clients are set to.
    client_epsilons: list[float | None] | None = field(default_factory=lambda: [8.0] * 20)
    delta: float = 1e-5                 # target delta for DP; should be <= 1/(number of training examples)
    max_grad_norm: float = 1.0          # per-example gradient clipping norm for the DP path
    dp_lr: float = 1e-3                 # AdamW learning rate for the DP path
    max_physical_batch_size: int | None = None
                                         # caps GPU memory use under DP; effective batch size stays `batch_size`

    results_dir: str = "results"        # where run output is written: {results_dir}/{method}_{dataset_task}_seed{seed}.json
    output: str | None = None           # optional override for the output filename

    # Which two text columns this task's examples come from.
    @property
    def text_fields(self) -> tuple[str, str | None]:
        return GLUE_TASKS[self.dataset_task][0]

    # How many classes this task has.
    @property
    def num_labels(self) -> int:
        return GLUE_TASKS[self.dataset_task][1]

    # Which dataset split to evaluate on.
    @property
    def eval_split(self) -> str:
        return GLUE_TASKS[self.dataset_task][2]

    # True whenever any client has a target epsilon configured.
    @property
    def use_dp(self) -> bool:
        return self.client_epsilons is not None

    # Build a Config from command-line arguments, falling back to the defaults above.
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

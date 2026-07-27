from __future__ import annotations

import argparse
from dataclasses import dataclass


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
    batch_size: int = 16
    max_length: int = 128
    seed: int = 42
    lr: float = 3e-4                    # single learning rate: FLoRA trains A and B jointly, not alternately

    # LoRA setup for language understanding
    lora_rank: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.1
    target_modules: tuple[str, ...] = ("query", "value")
    train_classifier_head: bool = True

    # Differential privacy (optional; None => plain FLoRA). These defaults are
    # deliberately conservative starting points, not tuned values -- DP-SGD's
    # usable range for max_grad_norm/dp_lr/noise_multiplier depends heavily on
    # the model, task, and batch size, so expect to sweep them.
    max_grad_norm: float = 0.1          # per-example gradient clipping norm (flat: one combined norm across every
                                         # trainable tensor, both LoRA A/B and the classifier head) for Opacus DP-SGD
    noise_multiplier: float | None = None  # set > 0 to enable DP-FLoRA
    delta: float = 1e-5                 # only used to report accumulated epsilon spend, not as a solver target
    dp_lr: float = 0.01                 # SGD learning rate for the DP path (see client.py for why DP uses SGD,
                                         # not AdamW -- Adam's adaptive step size defeats DP-SGD's clipping/noise
                                         # calibration). Start small: unlike Adam's step, SGD's step scales
                                         # directly with the clipped+noised gradient, so too large a value here
                                         # (combined with `dp_momentum`'s amplification) will diverge outright.
    dp_momentum: float = 0.5            # SGD momentum for the DP path; averages the noised gradient across steps
                                         # within a round. Kept modest (rather than the usual 0.9) since momentum
                                         # amplifies the effective step size by roughly 1/(1 - dp_momentum), and
                                         # DP-SGD's per-step noise is already sizeable relative to the true signal.

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
        """True whenever DP-FLoRA's Opacus pipeline should be engaged."""
        return self.noise_multiplier is not None

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
        parser.add_argument("--noise-multiplier", type=float, default=defaults.noise_multiplier,
                            help="Gaussian noise multiplier for DP-SGD; unset disables DP (plain FLoRA)")
        parser.add_argument("--max-grad-norm", type=float, default=defaults.max_grad_norm,
                            help="per-example gradient clipping norm (flat, across LoRA A/B and the classifier head) under DP-SGD")
        parser.add_argument("--dp-lr", type=float, default=defaults.dp_lr,
                            help="SGD learning rate used for the DP path (DP-SGD uses SGD, not AdamW)")
        parser.add_argument("--dp-momentum", type=float, default=defaults.dp_momentum,
                            help="SGD momentum used for the DP path")
        args = parser.parse_args(argv)
        return cls(
            method=args.method,
            dataset_task=args.task,
            seed=args.seed,
            results_dir=args.results_dir,
            noise_multiplier=args.noise_multiplier,
            max_grad_norm=args.max_grad_norm,
            dp_lr=args.dp_lr,
            dp_momentum=args.dp_momentum,
        )

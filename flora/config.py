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
    batch_size: int = 128               # matches the FLoRA paper's effective batch size (arXiv:2409.05976 Appendix
                                         # A.2: "the batch size is 128 and the micro batch size is 16" -- they
                                         # accumulate 8 micro-batches of 16 to reach it; RoBERTa-base is small
                                         # enough that a real batch of 128 is used directly here instead). A larger
                                         # batch also gives a less noisy mean gradient for the DP path to clip/noise.
    max_length: int = 128
    seed: int = 42
    lr: float = 3e-4                    # single learning rate: FLoRA trains A and B jointly, not alternately

    # LoRA setup for language understanding
    lora_rank: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.1
    target_modules: tuple[str, ...] = ("query", "value")
    train_classifier_head: bool = True

    # Differential privacy (optional; None => plain FLoRA). Client-level DP-SGD
    # following Liu et al., "Differentially Private Low-Rank Adaptation of
    # Large Language Model Using Federated Learning" (arXiv:2312.17493,
    # Algorithm 1) -- clip the ordinary (already batch-averaged) gradient once
    # per step to `max_grad_norm`, add one Gaussian noise draw calibrated to
    # `noise_multiplier * max_grad_norm`, then take a plain SGD step. See
    # client.py's `_local_update_dp` for why this (not Opacus's per-example
    # DP-SGD) is what's implemented. `max_grad_norm` and `dp_lr` below default
    # to that paper's own validated BERT-base hyperparameters (their Table 2:
    # clipping bound C=10, learning rate 5e-4); `noise_multiplier` is the
    # direct analogue of their noise scale sigma and is left for the caller to
    # set, same as the privacy/utility sweeps in that paper's own tables.
    max_grad_norm: float = 10.0         # gradient clipping norm (once per step, per group -- LoRA A/B and the
                                         # classifier head clipped independently -- not per-example) for DP-SGD
    noise_multiplier: float | None = None  # set > 0 to enable DP-FLoRA (this is "sigma" in the DP-LoRA paper)
    delta: float = 1e-5                 # only used to report accumulated epsilon spend, not as a solver target
    dp_lr: float = 5e-4                 # plain SGD learning rate for the DP path
    dp_momentum: float = 0.0            # SGD momentum for the DP path; 0 matches the reference algorithm (plain
                                         # SGD, no momentum term) -- adaptive optimizers like Adam are a known bad
                                         # fit for DP-SGD (DP noise biases Adam's second-moment estimate; see
                                         # "DP-AdamBC", arXiv:2312.14334), and momentum has the same amplification
                                         # risk that made an earlier (higher-momentum) attempt at this diverge.

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
        """True whenever DP-FLoRA's client-level DP-SGD path should be engaged."""
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
                            help="gradient clipping norm (once per step, per group) for the DP path")
        parser.add_argument("--dp-lr", type=float, default=defaults.dp_lr,
                            help="plain SGD learning rate used for the DP path")
        parser.add_argument("--dp-momentum", type=float, default=defaults.dp_momentum,
                            help="SGD momentum used for the DP path (0 matches the reference algorithm)")
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

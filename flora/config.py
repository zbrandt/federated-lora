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

    # Differential privacy (optional; None => plain FLoRA)
    max_grad_norm: float = 1.0          # per-example gradient clipping norm for the classifier head under Opacus DP-SGD
    lora_max_grad_norm: float = 0.1     # separate (smaller) per-example clipping norm for the LoRA A/B parameters --
                                         # their raw gradients are naturally far smaller than the classifier head's
                                         # (attenuated by the frozen encoder), so Opacus's default "flat" clipping,
                                         # which computes ONE combined norm across every trainable tensor and applies
                                         # that single clip factor to all of them, lets the head's much larger
                                         # per-example norm dictate the factor for the adapter too -- crushing its
                                         # already-tiny gradient before noise is even added. Per-layer clipping
                                         # (see client.py) clips each parameter to its own norm instead.
    noise_multiplier: float | None = None  # set > 0 to enable DP-FLoRA
    delta: float = 1e-5                 # only used to report accumulated epsilon spend, not as a solver target
    dp_lr: float = 0.5                  # SGD learning rate for the DP path (see client.py for why DP uses SGD,
                                         # not AdamW -- Adam's adaptive step size defeats DP-SGD's clipping/noise
                                         # calibration). SGD needs a much larger LR than Adam for the same
                                         # progress, since its step scales directly with the (small) clipped
                                         # gradient instead of Adam's ~lr-normalized-regardless-of-magnitude step.
    dp_momentum: float = 0.9            # SGD momentum for the DP path; averages the noised gradient across
                                         # steps within a round, similar to Adam's first moment but without
                                         # Adam's problematic (noise-biased) second-moment normalization

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
                            help="per-example gradient clipping norm for the classifier head under DP-SGD")
        parser.add_argument("--lora-max-grad-norm", type=float, default=defaults.lora_max_grad_norm,
                            help="per-example gradient clipping norm for the LoRA A/B parameters under DP-SGD")
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
            lora_max_grad_norm=args.lora_max_grad_norm,
            dp_lr=args.dp_lr,
            dp_momentum=args.dp_momentum,
        )

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
    method: str = "flora"
    model_name: str = "roberta-base"
    dataset_name: str = "nyu-mll/glue"
    dataset_task: str = "sst2"
    train_split: str = "train"

    # Federated setup for language understanding
    num_clients: int = 20
    client_sample_rate: float = 0.2
    partition_strategy: str = "noniid"
    dirichlet_alpha: float = 0.8
    rounds: int = 100
    local_steps: int = 20
    local_epochs: int | None = None
    batch_size: int = 128
    max_length: int = 128
    seed: int = 42
    lr: float = 3e-4

    # LoRA setup for language understanding
    lora_rank: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.1
    target_modules: tuple[str, ...] = ("query", "value")
    train_classifier_head: bool = True

    client_epsilons: list[float | None] | None = field(default_factory=lambda: [8.0] * 20)
    delta: float = 1e-5
    max_grad_norm: float = 1.0
    dp_lr: float = 1e-3
    max_physical_batch_size: int | None = None

    results_dir: str = "results"
    output: str | None = None

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
        return self.client_epsilons is not None

    @classmethod
    def from_argv(cls, argv: list[str] | None = None) -> "Config":
        defaults = cls()
        parser = argparse.ArgumentParser(prog="flora")
        parser.add_argument("--method", default=defaults.method)
        parser.add_argument("--task", choices=list(GLUE_TASKS), default=defaults.dataset_task)
        parser.add_argument("--seed", type=int, default=defaults.seed)
        parser.add_argument("--results-dir", default=defaults.results_dir)
        parser.add_argument("--max-grad-norm", type=float, default=defaults.max_grad_norm)
        parser.add_argument("--dp-lr", type=float, default=defaults.dp_lr)
        args = parser.parse_args(argv)
        return cls(
            method=args.method,
            dataset_task=args.task,
            seed=args.seed,
            results_dir=args.results_dir,
            max_grad_norm=args.max_grad_norm,
            dp_lr=args.dp_lr,
        )

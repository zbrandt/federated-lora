from __future__ import annotations

import argparse
from dataclasses import dataclass


GLUE_TASKS = {
    "sst2": (("sentence", None),          2, "validation"),
    "qnli": (("question", "sentence"),    2, "validation"),
    "qqp":  (("question1", "question2"),  2, "validation"),
    "mnli": (("premise", "hypothesis"),   3, "validation_matched")
}

# TODO: figure out how dataclass works
@dataclass(slots=True)
class Config:
    model_name: str = "roberta-base"    # https://huggingface.co/FacebookAI/roberta-base
    dataset_name: str = "nyu-mll/glue"  # https://huggingface.co/datasets/nyu-mll/glue
    dataset_task: str = "sst2"          # sentences from movie reviews and human annotations of their sentiment
    train_split: str = "train"          # TODO: figure out what this is

    # Federated setup for language understanding
    num_clients: int = 20
    client_sample_rate: float = 0.2
    partition_strategy: str = "noniid"
    dirichlet_alpha: float = 0.8 
    rounds: int = 100
    local_steps: int = 20
    batch_size: int = 16
    max_length: int = 128   # TODO: figure out what this is
    seed: int = 42
    lr_a: float = 3e-4
    lr_b: float = 3e-4
    lr_head: float = 3e-4

    # LoRA setup for language understanding
    lora_rank: int = 8
    lora_alpha: int = 8                                     # TODO: figure out what this is
    lora_dropout: float = 0.1                               # TODO: figure out what this is
    target_modules: tuple[str, ...] = ("query", "value")    # TODO: figure out what this is
    train_classifier_head: bool = True                      # TODO: figure out how original paper trained classifier head

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

    @classmethod
    def from_argv(cls, argv: list[str] | None = None) -> "Config":
        parser = argparse.ArgumentParser(prog="la_lora")
        parser.add_argument("--task", choices=list(GLUE_TASKS), default="sst2")
        args = parser.parse_args(argv)
        return cls(dataset_task=args.task)

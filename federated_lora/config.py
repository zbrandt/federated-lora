from __future__ import annotations

import argparse
from dataclasses import dataclass, field

from peft import TaskType

GLUE_TASKS = {
	'sst2': (('sentence', None), 2, 'validation'),
	'qnli': (('question', 'sentence'), 2, 'validation'),
	'qqp': (('question1', 'question2'), 2, 'validation'),
	'mnli': (('premise', 'hypothesis'), 3, 'validation_matched'),
}


@dataclass
class PrivacyConfig:
	dp: bool = False
	clip_norm: float = 1.0  # TODO
	target_epsilon: float | None = None  # TODO
	target_delta: float = 1e-5  # TODO
	noise_multiplier: float | None = None  # TODO
	smoothing: bool = True  # TODO


@dataclass(slots=True)
class Config:
	method: str = 'lalora'
	model_name: str = 'roberta-base'
	dataset_name: str = 'nyu-mll/glue'
	dataset_task: str = 'sst2'  # sentences from movie reviews and human annotations of their sentiment
	train_split: str = 'train[:1%]'  # TODO

	# Federated setup for language understanding
	num_clients: int = 4
	client_sample_rate: float = 0.2
	partition_strategy: str = 'noniid'
	dirichlet_alpha: float = 0.8
	rounds: int = 2
	local_steps: int = 20
	batch_size: int = 16
	max_length: int = 128  # TODO
	seed: int = 42
	lr_a: float = 3e-4
	lr_b: float = 3e-4
	lr_head: float = 3e-4

	# LoRA setup for language understanding
	task_type: TaskType = TaskType.SEQ_CLS
	lora_rank: int = 8
	target_modules: tuple[str, ...] = (
		'query',
		'value',
	)  # names of the modules to apply adapter to
	lora_alpha: int = 8  # alpha parameter for LoRA scaling
	lora_dropout: float = 0.1  # dropout probability for LoRA layers
	modules_to_save: tuple[str, ...] = (
		'classifier'  # modules apart from the LoRA layers to be trained and saved
	)
	train_classifier_head: bool = False

	privacy: PrivacyConfig = field(default_factory=PrivacyConfig)

	results_dir: str = 'results'  # runs auto-save in this directory as <method>_<task>_seed<seed>.json
	output: str | None = (
		None  # explicit path override, ignores results_dir naming when set
	)

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
	def from_argv(cls, argv: list[str] | None = None) -> Config:
		defaults = cls()
		parser = argparse.ArgumentParser(prog='la_lora')
		parser.add_argument(
			'--method',
			default=defaults.method,
			help='run label used for the output filename and plot legend',
		)
		parser.add_argument(
			'--task', choices=list(GLUE_TASKS), default=defaults.dataset_task
		)
		parser.add_argument(
			'--seed',
			type=int,
			default=defaults.seed,
			help='random number generator seed for mean and standard deviation bands',
		)
		parser.add_argument(
			'--results-dir',
			default=defaults.results_dir,
			help='directory where the run output is written to',
		)
		args = parser.parse_args(argv)
		return cls(
			method=args.method,
			dataset_task=args.task,
			seed=args.seed,
			results_dir=args.results_dir,
		)

from __future__ import annotations

import argparse
from dataclasses import dataclass, field


# task -> (hf_dataset_id, image_field, label_field, num_labels, eval_split)
IMAGE_TASKS = {
	'cifar100': ('uoft-cs/cifar100', 'img', 'fine_label', 100, 'test'),
}

# TODO: consider separate configuration
@dataclass
class PrivacyConfig:
	clip_norm: float = 1.0  # per-example L2 clipping threshold C
	target_epsilon: float | None = 3.0  # set to None for a non-private run
	target_delta: float = 1e-5
	noise_multiplier: float | None = None  # filled in by build()
	smoothing: bool = True  # apply the low-pass Gaussian filter to LoRA grads


@dataclass(slots=True)
class Config:
	method: str = 'lalora'
	model: str = 'google/vit-base-patch16-224' # use ViT instead of the paper's Swin
	task: str = 'cifar100'
	train_split: str = 'train'

	# set up hyperparameters for image classification 
	num_clients: int = 8
	client_sample_rate: float = 0.5
	partition_strategy: str = 'noniid'
	dirichlet_alpha: float = 0.1
	global_rounds: int = 25  # TODO: paper uses 100
	local_steps: int = 20
	batch_size: int = 16
	seed: int = 42

	# setup hyperparameters for LoRA fine-tuning
	lora_rank: int = 16
	lora_alpha: int = 16
	target_modules: tuple[str, ...] = ('query', 'value')
	lora_dropout: float = 0.0

	# set up hyperparameters for optimization
	lr_a: float = 0.1
	lr_b: float = 0.1
	lr_head: float = 0.1
	lr_decay: float = 0.99  # multiplies the lr each global round

	privacy: PrivacyConfig = field(default_factory=PrivacyConfig)

	results_dir: str = 'results'
	output: str | None = None

	@property
	def dataset_id(self) -> str:
		return IMAGE_TASKS[self.task][0]

	@property
	def image_field(self) -> str:
		return IMAGE_TASKS[self.task][1]

	@property
	def label_field(self) -> str:
		return IMAGE_TASKS[self.task][2]

	@property
	def num_labels(self) -> int:
		return IMAGE_TASKS[self.task][3]

	@property
	def eval_split(self) -> str:
		return IMAGE_TASKS[self.task][4]

	@classmethod
	def from_argv(cls, argv: list[str] | None = None) -> Config:
		defaults = cls()
		parser = argparse.ArgumentParser(prog='la_lora')
		parser.add_argument(
			'--method', default=defaults.method
		)
		parser.add_argument(
			'--task', choices=list(IMAGE_TASKS), default=defaults.task
		)
		parser.add_argument(
			'--seed',
			type=int, default=defaults.seed
		)
		parser.add_argument(
			'--results-dir', default=defaults.results_dir
		)
		args = parser.parse_args(argv)
		return cls(
			method=args.method,
			task=args.task,
			seed=args.seed,
			results_dir=args.results_dir,
		)

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

# TODO
IMAGE_TASKS = {
	'cifar100': ('uoft-cs/cifar100', 'img', 'fine_label', 100, 'test'),
}


@dataclass
class PrivacyConfig:
	clip_norm: float = 1.0  # per-sample L2 clipping norm
	target_epsilon: float = 3.0  # the target privacy loss budget
	target_delta: float = 1e-5  # TODO
	noise_multiplier: float = (
		0.0  # the Gaussian noise multiplier for differential privacy
	)
	smoothing: bool = True


@dataclass(slots=True)
class Config:
	method: str = 'lalora'
	model: str = 'google/vit-base-patch16-224-in21k'
	task: str = 'cifar100'

	# set up hyperparameters for image classification
	num_clients: int = 8
	client_sample_rate: float = 0.5
	dirichlet_alpha: float = (
		0.1  # the parameter of the Dirichlet distribution.
	)
	global_rounds: int = 20  # the number of communication rounds
	local_steps: int = 20  # the number of local update steps per round
	batch_size: int = 16
	num_workers: int = 8
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
	def dataset(self) -> str:
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
		parser.add_argument('--method', default=defaults.method)
		parser.add_argument(
			'--task', choices=list(IMAGE_TASKS), default=defaults.task
		)
		parser.add_argument('--seed', type=int, default=defaults.seed)
		parser.add_argument('--results-dir', default=defaults.results_dir)
		args = parser.parse_args(argv)
		return cls(
			method=args.method,
			task=args.task,
			seed=args.seed,
			results_dir=args.results_dir,
		)

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Config:
	task: str
	method: str
	output_dir: str
	seed: int = 42

	dataset: str = 'uoft-cs/cifar100'
	num_clients: int = 3
	dirichlet_alpha: float = 0.5

	model: str = 'google/vit-base-patch16-224-in21k'
	num_labels: int = 100  # number of labels to use in the last layer added to the model, typically for a classification task
	lora_rank: int = 8
	lora_alpha: int = 8
	target_modules: tuple[str, ...] = ('query', 'value')
	lora_dropout: float = 0.05

	global_rounds: int = 4  # the number of communication rounds
	local_steps: int = 20  # the number of local update steps per round
	batch_size: int = 16
	num_workers: int = 8

	clip_norm: float = 1.0  # per-sample L2 clipping norm
	target_epsilon: float = 3.0
	target_delta: float = 1e-5

	# set up hyperparameters for optimization
	lr_a: float = 0.1
	lr_b: float = 0.1
	lr_head: float = 0.1
	lr_decay: float = 0.99  # multiplies the lr each global round

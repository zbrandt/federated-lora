from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Config:
	task: str
	method: str
	output_dir: str = 'results/'
	seed: int = 42

	dataset: str = 'uoft-cs/cifar100'
	num_clients: int = 8
	client_sample_rate: float = 0.5
	dirichlet_alpha: float = 0.1

	model: str = 'microsoft/swin-base-patch4-window7-224-in22k'
	num_labels: int = 100  # number of labels to use in the last layer added to the model, typically for a classification task
	lora_rank: int = 16
	lora_alpha: int = 16
	target_modules: tuple[str, ...] = ('query', 'value')
	lora_dropout: float = 0.05

	global_rounds: int = 100  # the number of communication rounds
	local_steps: int = 20  # the number of local update steps per round
	batch_size: int = 16
	num_workers: int = 8

	max_grad_norm: float = 4.0  # per-example L2-norm clipping
	target_epsilon: float = 3.0
	target_delta: float = 1e-5

	lr_a: float = 0.20
	lr_b: float = 0.20
	lr_head: float = 0.20
	lr_decay: float = 0.99

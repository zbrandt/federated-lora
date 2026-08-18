from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Config:
	task: str
	method: str
	output_dir: str = 'results/'
	seed: int = 42

	dataset: str = 'uoft-cs/cifar100'
	num_clients: int = 20
	dirichlet_alpha: float = 0.5

	model: str = 'google/vit-base-patch16-224-in21k'
	num_labels: int = 100  # number of labels to use in the last layer added to the model, typically for a classification task
	# i am keeping this as a fallback
	lora_rank: int = 8
	client_ranks: tuple[int, ...] | None=None
	lora_alpha: int = 8
	target_modules: tuple[str, ...] = ('query', 'value')
	lora_dropout: float = 0.05

	global_rounds: int = 100  # the number of communication rounds
	local_steps: int = 20  # the number of local update steps per round
	batch_size: int = 64
	num_workers: int = 8

	clip_norm: float = 1.0  # per-sample L2 clipping norm
	# target_epsilon: float = 3.0
	# add epsilons 
	target_epsilon: float = 3.0 
	client_epsilons: tuple[float, ...] | None = None 
	target_delta: float = 1e-5

	lr_a: float = 0.25
	lr_b: float = 0.25
	lr_head: float = 0.15

	log_uploads_dir: str | None = None  # if set, dump raw per-client uploads each round for offline aggregation-error analysis

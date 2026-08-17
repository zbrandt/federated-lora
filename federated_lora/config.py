from __future__ import annotations

from dataclasses import dataclass

TASKS: dict[str, dict] = {
	'cifar100': {
		'experiment': 'vision',
		'dataset': 'uoft-cs/cifar100',
		'dataset_config': None,
		'eval_split': 'test',
		'label_field': 'fine_label',
		'text_fields': (),
		'num_labels': 100,
		'model': 'google/vit-base-patch16-224-in21k',
		'target_modules': ('query', 'value'),
	},
	'sst2': {
		'experiment': 'text',
		'dataset': 'nyu-mll/glue',
		'dataset_config': 'sst2',
		'eval_split': 'validation',
		'label_field': 'label',
		'text_fields': ('sentence',),
		'num_labels': 2,
		'model': 'FacebookAI/roberta-base',
		'target_modules': ('query', 'value'),
	},
}


@dataclass(slots=True)
class Config:
	task: str
	method: str
	output_dir: str = 'results/'
	seed: int = 42
	experiment: str = 'vision'

	dataset: str = 'uoft-cs/cifar100'
	dataset_config: str | None = (
		None  # HF dataset config name (e.g. glue 'sst2')
	)
	eval_split: str = 'test'  # split used for evaluation
	label_field: str = 'fine_label'  # label column in the raw dataset
	text_fields: tuple[str, ...] = ()  # input text column(s) for text tasks
	max_length: int = 128  # max token length for text tasks

	num_clients: int = 8
	client_sample_rate: float = 0.5
	dirichlet_alpha: float = 0.1

	model: str = 'google/vit-base-patch16-224-in21k'
	num_labels: int = 100  # number of labels to use in the last layer added to the model, typically for a classification task
	lora_rank: int = 16
	lora_alpha: int = 16
	target_modules: tuple[str, ...] = ('query', 'value')
	lora_dropout: float = 0.05

	global_rounds: int = 100  # the number of communication rounds
	local_steps: int = 20  # the number of local update steps per round
	batch_size: int = 16
	num_workers: int = 8

	# --- differential privacy -------------------------------------------------
	clipping: str = (
		'median'  # 'median' (per-layer) | 'flat' (standard DP-SGD) | 'none'
	)
	max_grad_norm: float = (
		4.0  # per-example L2-norm clipping bound (used by 'flat')
	)
	target_epsilon: float | None = 3.0
	target_delta: float = 1e-5

	lr_a: float = 0.02
	lr_b: float = 0.02
	lr_head: float = 0.02
	lr_decay: float = 0.99

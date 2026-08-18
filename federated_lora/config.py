from __future__ import annotations

from dataclasses import dataclass

TASKS: dict[str, dict] = {
	'cifar100': {
		'experiment': 'vision',
		'dataset': 'uoft-cs/cifar100',
		'eval_split': 'test',
		'label_field': 'fine_label',
		'text_fields': (),
		'num_labels': 100,
		# 'model': 'google/vit-base-patch16-224-in21k',
		'model': 'microsoft/swin-tiny-patch4-window7-224',
		'target_modules': ('query', 'value'),
	},
	'sst2': {
		'experiment': 'text',
		'dataset': 'nyu-mll/glue',
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
	batch_size: int
	target_epsilon: float | None
	clipping: str
	lr_a: float
	lr_b: float
	lr_head: float
	seed: int
	output_dir: str

	experiment: str
	dataset: str
	eval_split: str  # split used for evaluation
	label_field: str  # label column in the raw dataset
	text_fields: tuple[str, ...]  # input text column(s) for text tasks
	num_labels: int  # number of labels to use in the last layer added to the model, typically for a classification task
	model: str

	num_clients: int = 8
	global_rounds: int = 20
	local_steps: int = 20
	client_sample_rate: float = 0.5
	dirichlet_alpha: float = 0.1
	
	lora_rank: int = 16
	lora_alpha: int = 16
	target_modules: tuple[str, ...] = ('query', 'value')
	lora_dropout: float = 0.05

	max_grad_norm: float = 4.0
	target_delta: float = 1e-5
	lr_decay: float = 0.99

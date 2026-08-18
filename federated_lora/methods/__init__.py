from __future__ import annotations

from federated_lora.method import Method
from federated_lora.methods.dp_lora import DPLoRA
from federated_lora.methods.ffa_lora import FFALoRA
from federated_lora.methods.flora import FLoRA
from federated_lora.methods.la_lora import LALoRA
from federated_lora.methods.rolora import RoLoRA

METHODS: dict[str, type[Method]] = {
	'dp_lora': DPLoRA,
	'ffa_lora': FFALoRA,
	'rolora': RoLoRA,
	'la_lora': LALoRA,
	'flora': FLoRA,
}

__all__ = ['Method']

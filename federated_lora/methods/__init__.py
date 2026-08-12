from __future__ import annotations

from federated_lora.method import Method
from federated_lora.methods.dp_lora import DPLoRA
from federated_lora.methods.ffa_lora import FFALoRA
from federated_lora.methods.la_lora import LALoRA
from federated_lora.methods.rolora import RoLoRA
from federated_lora.methods.flora import FLoRA

_METHODS: dict[str, type[Method]] = {
	'dp_lora': DPLoRA,
	'ffa_lora': FFALoRA,
	'rolora': RoLoRA,
	'la_lora': LALoRA,
	'flora': FLoRA,
}


def get_method(name: str) -> Method:
	return _METHODS[name]()


__all__ = ['Method', 'get_method']

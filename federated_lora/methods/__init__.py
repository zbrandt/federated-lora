from __future__ import annotations

from federated_lora.method import Method
from federated_lora.methods.dp_lora import DPLoRA
from federated_lora.methods.ffa_lora import FFALoRA
from federated_lora.methods.lalora import LaLoRA
from federated_lora.methods.rolora import RoLoRA

_METHODS: dict[str, type[Method]] = {
	'lalora': LaLoRA,
	'dp_lora': DPLoRA,
	'ffa_lora': FFALoRA,
	'rolora': RoLoRA,
}


def get_method(name: str) -> Method:
	return _METHODS[name]()


__all__ = ['Method', 'get_method']

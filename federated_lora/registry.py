from __future__ import annotations

from federated_lora.methods.base import Method
from federated_lora.methods.lalora import LaLoRA


_METHODS: dict[str, type[Method]] = {
    'lalora': LaLoRA,
    # 'fedit': FedIT,
    # 'dp_lora': DPLoRA,
    # 'ffa_lora': FFALoRA,
    # 'rolora': RoLoRA,
}

def get_method(name: str) -> method:
    return _METHODS[name]()

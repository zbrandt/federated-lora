from __future__ import annotations

from dataclasses import dataclass

import torch
from datasets import Dataset
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling, PreTrainedModel, PreTrainedTokenizerBase

from config import Config


def configure_trainable_lora_parameters(model: PreTrainedModel, method: str) -> None:
	# FedIT trains both LoRA matrices; FFA trains only B and freezes A.
	for name, parameter in model.named_parameters():
		if "lora_A" in name:
			parameter.requires_grad = method == "fedit"
		elif "lora_B" in name:
			parameter.requires_grad = True
		else:
			parameter.requires_grad = False


def extract_adapter_state(model: PreTrainedModel) -> dict[str, torch.Tensor]:
	"""Send adapter tensors back to the server, not the base GPT-2 weights."""
	return {
		key: tensor.detach().cpu().clone()
		for key, tensor in get_peft_model_state_dict(model).items()
	}


def adapter_upload_bytes(state_dict: dict[str, torch.Tensor], method: str) -> int:
	# Count only the tensors that each method actually communicates.
	if method == "ffa":
		keys = [key for key in state_dict if "lora_B" in key]
	else:
		keys = [key for key in state_dict if "lora_" in key]
	return sum(state_dict[key].numel() * state_dict[key].element_size() for key in keys)


@dataclass(slots=True)
class ClientResult:
	state_dict: dict[str, torch.Tensor]
	n_examples: int
	n_tokens: int
	average_loss: float
	uploaded_bytes: int


class Client:
	def __init__(
		self,
		client_id: int,
		dataset: Dataset,
		tokenizer: PreTrainedTokenizerBase,
		config: Config,
		device: torch.device,
	) -> None:
		self.client_id = client_id
		self.dataset = dataset
		self.config = config
		self.device = device
		self.collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

	def local_update(
		self,
		model: PreTrainedModel,
		adapter_state: dict[str, torch.Tensor],
	) -> ClientResult:
		"""Load the server adapter, run local training, and return the updated adapter."""
		set_peft_model_state_dict(model, adapter_state)
		configure_trainable_lora_parameters(model, self.config.method)
		model.train()

		trainable_parameters = [
			parameter for parameter in model.parameters() if parameter.requires_grad # set in configure_trainable_lora_parameters
		]
		optimizer = AdamW(trainable_parameters, lr=self.config.learning_rate)
		data_loader = DataLoader(
			self.dataset,
			batch_size=self.config.batch_size,
			shuffle=True,
			collate_fn=self.collator,
		)

		total_loss = 0.0
		total_batches = 0
		total_tokens = 0
		for _ in range(self.config.local_epochs):
			# One simple manual epoch loop keeps the training state visible.
			for batch in data_loader:
				batch = {key: value.to(self.device) for key, value in batch.items()}
				total_tokens += int(batch["attention_mask"].sum().item())

				optimizer.zero_grad(set_to_none=True)
				loss = model(**batch).loss
				loss.backward()
				optimizer.step()

				total_loss += float(loss.item())
				total_batches += 1

		updated_state = extract_adapter_state(model)
			# Package the pieces the server needs for aggregation and logging.
		return ClientResult(
			state_dict=updated_state,
			n_examples=len(self.dataset),
			n_tokens=total_tokens,
			average_loss=total_loss / max(1, total_batches),
			uploaded_bytes=adapter_upload_bytes(updated_state, self.config.method),
		)

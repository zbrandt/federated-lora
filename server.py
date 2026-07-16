from __future__ import annotations

from dataclasses import dataclass

import torch
from peft import LoraConfig, TaskType, get_peft_model, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer

from aggregate import FFAAvg, FedITAvg
from client import Client, configure_trainable_lora_parameters, extract_adapter_state
from config import Config
from data import load_datasets
from eval import evaluate_perplexity


@dataclass(slots=True)
class RoundMetrics:
	# One row of logging output per communication round.
	round_index: int
	train_loss: float
	eval_loss: float
	perplexity: float
	uploaded_bytes: int
	cumulative_uploaded_bytes: int


class Server:
	def __init__(self, config: Config) -> None:
		self.config = config
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		# Load the tokenizer once and reuse it for all clients.
		self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
		if self.tokenizer.pad_token is None:
			self.tokenizer.pad_token = self.tokenizer.eos_token

		# Build one shared dataset split for clients plus a held-out eval split.
		train_shards, self.eval_dataset = load_datasets(config, self.tokenizer)

		# Load model, then wrap it with a LoRA adapter.
		dtype = torch.float16 if self.device.type == "cuda" else torch.float32
		base_model = AutoModelForCausalLM.from_pretrained(
			config.model_name,
			torch_dtype=dtype,
		)
		self.model = base_model.to(self.device)

		lora_config = LoraConfig(
			task_type=TaskType.CAUSAL_LM,
			target_modules=list(config.target_modules),
			r=config.lora_rank,
			lora_alpha=config.lora_alpha,
			lora_dropout=config.lora_dropout,
		)
		self.model = get_peft_model(self.model, lora_config)
		configure_trainable_lora_parameters(self.model, config.method) # method definition in client
		self.current_state = extract_adapter_state(self.model) # method definition in client

		# Create one client object per shard so the server loop stays simple.
		self.clients = [
			Client(
				client_id=index,
				dataset=shard,
				tokenizer=self.tokenizer,
				config=config,
				device=self.device,
			)
			for index, shard in enumerate(train_shards)
		]
		self.aggregator = FedITAvg() if config.method == "fedit" else FFAAvg()

	# Each round broadcasts the current adapter, collects updates, averages them,
	# and measures validation perplexity.
	def run(self) -> list[RoundMetrics]:
		history: list[RoundMetrics] = []
		cumulative_uploaded_bytes = 0

		for round_index in range(1, self.config.rounds + 1):
			client_states: list[dict[str, torch.Tensor]] = []
			client_weights: list[float] = []
			train_losses: list[float] = []
			round_uploaded_bytes = 0

			for client in self.clients:
				# Push the latest server adapter before each client update.
				set_peft_model_state_dict(self.model, self.current_state)
				client_result = client.local_update(self.model, self.current_state)
				client_states.append(client_result.state_dict)
				client_weights.append(float(client_result.n_examples))
				train_losses.append(client_result.average_loss)
				round_uploaded_bytes += client_result.uploaded_bytes

			# Aggregate the client adapters into the next server state.
			self.current_state = self.aggregator.aggregate(client_states, client_weights)
			set_peft_model_state_dict(self.model, self.current_state)

			# Evaluate the current global adapter on the shared validation split.
			eval_loss, perplexity = evaluate_perplexity(
				self.model,
				self.eval_dataset,
				self.tokenizer,
				self.config,
				self.device,
			)

			cumulative_uploaded_bytes += round_uploaded_bytes
			history.append(
				RoundMetrics(
					round_index=round_index,
					train_loss=sum(train_losses) / max(1, len(train_losses)),
					eval_loss=eval_loss,
					perplexity=perplexity,
					uploaded_bytes=round_uploaded_bytes,
					cumulative_uploaded_bytes=cumulative_uploaded_bytes,
				)
			)

		return history

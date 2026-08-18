from __future__ import annotations

import gc
import traceback

import torch
from torch.utils.data import DataLoader, Subset

from experiment import build
from federated_lora.config import Config
from federated_lora.model import get_trainable_parameters


TASK = "cifar100"
METHODS = ['dp_lora', 'rolora', 'ffa_lora', 'la_lora']
NUM_CLIENTS = 2
GLOBAL_ROUNDS = 2
LOCAL_STEPS = 2
BATCH_SIZE = 16
EVAL_SUBSET = 128

checks: list[tuple[str, bool, str]] = []

def check(name: str, ok: bool, detail: str = '') -> None:
	checks.append((name, bool(ok), detail))
	print(
		f'  [{"PASS" if ok else "FAIL"}] {name}'
		+ (f' — {detail}' if detail else '')
	)


def run_method(method: str) -> None:
	print(f'\n=== {method} ===')
	config = Config(
		task=TASK,
		method=method,
		num_clients=NUM_CLIENTS,
		global_rounds=GLOBAL_ROUNDS,
		local_steps=LOCAL_STEPS,
		batch_size=BATCH_SIZE,
		output_dir=f'results/smoke/{TASK}',
	)

	server = build(config)
	check(f'{method}: build() completed', True, f'device={server.device.type}')

	trainable = get_trainable_parameters(server.model)
	# total = sum(p.numel() for p in server.model.parameters())
	
	
	backbone_frozen = sum(
		1
		for n, p in server.model.named_parameters()
		if not p.requires_grad and 'lora_' not in n and 'classifier' not in n
	)
	
	# check(
	# 	f'{method}: backbone size ~ViT-Base',
	# 	80e6 < total < 95e6,
	# 	f'{total / 1e6:.1f}M params',
	# )
	
	check(
		f'{method}: LoRA adapters trainable',
		any('lora_' in n for n in trainable),
		f'{sum("lora_" in n for n in trainable)} lora tensors',
	)

	check(
		f'{method}: classifier head trainable',
		any('classifier' in n for n in trainable),
	)

	check(
		f'{method}: backbone frozen',
		backbone_frozen > 0,
		f'{backbone_frozen} frozen params',
	)

	c0 = server.clients[0]
	check(
		f'{method}: Opacus wrapped model (DP)',
		'GradSample' in c0.model.__class__.__name__
		or hasattr(c0.model, '_module'),
		c0.model.__class__.__name__,
	)

	check(
		f'{method}: DP optimizer in place',
		'DP' in c0.optimizer.__class__.__name__,
		c0.optimizer.__class__.__name__,
	)

	lr_before = c0.optimizer.param_groups[0]['lr']

	# shrink the eval set so each round's evaluate() is fast
	old = server.test_dataloader
	server.test_dataloader = DataLoader(
		Subset(old.dataset, list(range(min(EVAL_SUBSET, len(old.dataset))))),
		batch_size=64,
		shuffle=False,
		collate_fn=old.collate_fn,
		num_workers=0,
	)

	try:
		history = server.run()
		ran, err = True, ''
	except Exception:
		history, ran, err = [], False, traceback.format_exc()

	check(
		f'{method}: server.run() completed',
		ran,
		'' if ran else 'traceback below',
	)
	if not ran:
		print('\n' + err)
		return

	check(
		f'{method}: one record per round',
		len(history) == GLOBAL_ROUNDS,
		f'{len(history)} rounds',
	)
	finite = all(
		torch.isfinite(torch.tensor(h['train_loss']))
		and torch.isfinite(torch.tensor(h['test_loss']))
		for h in history
	)
	check(
		f'{method}: metrics finite',
		finite,
		f'last train_loss={history[-1]["train_loss"]:.4f} '
		f'top1={history[-1]["top1_acc"]:.2f}%',
	)

	lr_after = c0.optimizer.param_groups[0]['lr']
	check(
		f'{method}: LR decayed on DP optimizer',
		lr_after < lr_before,
		f'{lr_before:.5f} -> {lr_after:.5f}',
	)

	del server
	gc.collect()


for m in METHODS:
	try:
		run_method(m)
	except Exception:
		print(f'\n[FAIL] {m}: build/setup raised:\n' + traceback.format_exc())
		checks.append((f'{m}: setup', False, 'exception'))

n_fail = sum(1 for _, ok, _ in checks if not ok)
print('\n' + '=' * 60)
print(
	f'{"ALL CHECKS PASSED" if n_fail == 0 else f"{n_fail} CHECK(S) FAILED"} '
	f'({len(checks) - n_fail}/{len(checks)})'
)
print('=' * 60)
raise SystemExit(0 if n_fail == 0 else 1)

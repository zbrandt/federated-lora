"""
Pre-flight checks for the federated LoRA benchmark harness.

Three suites, cheapest first:

- ``audit``  static configuration checks (instant, no torch work). Catches
	registry/table drift and argparse-default footguns before a sweep burns
	GPU hours.
- ``unit``   numerical invariants of the DP machinery and the four methods,
	on a synthetic model whose parameters are named like the PEFT ones
	(fast, no downloads, no GPU).
- ``smoke``  real end-to-end runs through ``experiment.py`` for each
	method/model, with tiny round counts, validating the results JSON
	(slow, downloads data on first use).

Usage
-----
	python test.py                      # audit + unit (seconds)
	python test.py --suite all          # everything
	python test.py --suite smoke --dry-run
	python test.py --suite smoke --models vit --methods la_lora
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from opacus import PrivacyEngine
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parent
PYTHON = sys.executable

RESULTS: list[tuple[str, bool, str]] = []


def quiet():
	"""
	Swallow the per-round progress printing of the code under test.
	"""
	return contextlib.redirect_stdout(io.StringIO())


def check(name: str, ok: bool, detail: str = '') -> bool:
	"""
	Record and print the outcome of a single check.
	"""
	RESULTS.append((name, bool(ok), detail))
	print(
		f'  [{"PASS" if ok else "FAIL"}] {name}'
		+ (f' — {detail}' if detail else ''),
		flush=True,
	)
	return bool(ok)


# ======================================================================
# synthetic model
# ======================================================================
class TinyLoRA(nn.Module):
	"""
	A minimal stand-in for the PEFT-wrapped backbone.

	Parameter names contain ``lora_A``, ``lora_B``, and ``classifier`` so
	that ``set_target_modules`` and every method's freezing logic engage
	exactly as they do on the real model.

	Unlike real LoRA, ``lora_B`` is initialised non-zero: with B = 0 the
	gradient of A is identically zero on the first step, which would make
	the alternating-update checks vacuous.
	"""

	def __init__(self, dim: int = 8, rank: int = 4, num_labels: int = 5):
		super().__init__()
		self.backbone = nn.Linear(dim, dim, bias=False)
		self.lora_A = nn.Linear(dim, rank, bias=False)
		self.lora_B = nn.Linear(rank, dim, bias=False)
		self.classifier = nn.Linear(dim, num_labels, bias=False)

		nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
		nn.init.normal_(self.lora_B.weight, std=0.1)

	def forward(self, pixel_values=None, labels=None):
		hidden = self.backbone(pixel_values) + self.lora_B(
			self.lora_A(pixel_values)
		)
		logits = self.classifier(hidden)

		loss = None
		if labels is not None:
			loss = nn.functional.cross_entropy(logits, labels)

		return SimpleNamespace(loss=loss, logits=logits)


def collate(batch):
	images = torch.stack(
		[torch.as_tensor(item[0], dtype=torch.float32) for item in batch]
	)
	labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
	return {'pixel_values': images, 'labels': labels}


def make_dataset(seed: int, size: int, dim: int = 8, num_labels: int = 5):
	rng = torch.Generator().manual_seed(seed)
	return [
		(
			torch.randn(dim, generator=rng).numpy(),
			int(torch.randint(0, num_labels, (1,), generator=rng)),
		)
		for _ in range(size)
	]


def make_batch(seed: int, size: int = 16, dim: int = 8, num_labels: int = 5):
	rng = torch.Generator().manual_seed(seed)
	return {
		'pixel_values': torch.randn(size, dim, generator=rng),
		'labels': torch.randint(0, num_labels, (size,), generator=rng),
	}


def make_private_client(
	seed: int,
	method_name: str = 'dp_lora',
	clipping: str = 'flat',
	noise_multiplier: float = 0.0,
	num_examples: int = 64,
	batch_size: int = 16,
	max_grad_norm: float = 2.0,
):
	"""
	Build a fresh Opacus-wrapped (model, optimizer, loader, engine) tuple
	with deterministic initialisation.
	"""
	from federated_lora.data import provide_dataloader
	from federated_lora.methods import METHODS

	torch.manual_seed(seed)
	model = TinyLoRA()
	METHODS[method_name]().set_target_modules(model)

	dataloader = provide_dataloader(
		make_dataset(seed + 1, num_examples), batch_size, collate, False
	)
	optimizer = SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)

	engine = PrivacyEngine(accountant='rdp')
	model, optimizer, dataloader = engine.make_private(
		module=model,
		optimizer=optimizer,
		data_loader=dataloader,
		noise_multiplier=noise_multiplier,
		max_grad_norm=max_grad_norm,
	)
	optimizer.clipping_strategy = clipping

	return model, optimizer, dataloader, engine


def updated_parameters(before: dict, model: nn.Module) -> set[str]:
	"""
	Names of parameters whose values changed relative to ``before``.
	"""
	return {
		name
		for name, p in model.named_parameters()
		if not torch.equal(before[name], p)
	}


def snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
	return {n: p.detach().clone() for n, p in model.named_parameters()}


# ======================================================================
# audit suite
# ======================================================================
def audit_suite() -> None:
	"""
	Static consistency checks across the registry, the sweep tables, and
	the argparse defaults. These guard the footguns that previously cost
	whole sweeps: a stale method registry, a missing noise multiplier, and
	silently wrong round counts.
	"""
	print('\n=== audit ===')

	import experiment
	import sweep
	from federated_lora.methods import METHODS

	check(
		'registry and sweep agree on the method list',
		set(METHODS) == set(sweep.METHODS),
		f'registry={sorted(METHODS)}, sweep={sorted(sweep.METHODS)}',
	)

	missing_lr = [
		f'eps{e}/{m}'
		for e in sweep.EPSILONS
		for m in sweep.METHODS
		if m not in sweep.LR.get(e, {})
	]
	check(
		'LR table covers every (epsilon, method) in the sweep',
		not missing_lr,
		f'missing: {missing_lr}' if missing_lr else '',
	)

	missing_sigma = [
		f'{t}/eps{float(e)}'
		for t in sweep.TASKS
		for e in sweep.EPSILONS
		if float(e) not in experiment.NOISE_MULTIPLIERS.get(t, {})
	]
	check(
		'paper noise multipliers cover every (task, epsilon) for swin',
		not missing_sigma,
		f'missing: {missing_sigma}' if missing_sigma else '',
	)

	check(
		'model and task tables cover the sweep matrix',
		set(sweep.MODELS) <= set(experiment.MODELS)
		and set(sweep.TASKS) <= set(experiment.DATASETS)
		and set(sweep.TASKS) <= set(experiment.LABELS),
		f'models={sorted(experiment.MODELS)}, '
		f'tasks={sorted(experiment.DATASETS)}',
	)

	# the argparse defaults are what an omitted flag actually produces
	argv = sys.argv
	sys.argv = [
		'experiment.py',
		'--task',
		'cifar100',
		'--model',
		'vit',
		'--method',
		'dp_lora',
	]
	try:
		defaults = experiment.parse_args()
	finally:
		sys.argv = argv

	check(
		'default global_rounds is the paper protocol (100)',
		defaults.global_rounds == 100,
		f'got {defaults.global_rounds}',
	)
	check(
		'default local_steps is the paper protocol (20)',
		defaults.local_steps == 20,
		f'got {defaults.local_steps}',
	)
	check(
		'default num_clients / sample_rate / alpha as intended',
		defaults.num_clients == 8
		and defaults.sample_rate == 0.5
		and defaults.dirichlet_alpha == 0.1,
		f'clients={defaults.num_clients}, '
		f'sample_rate={defaults.sample_rate}, '
		f'alpha={defaults.dirichlet_alpha}',
	)

	# sweep must launch the whole matrix, with an all-string argv
	launched: list[list] = []

	class _Completed:
		returncode = 0

	real_run = sweep.subprocess.run
	sweep.subprocess.run = lambda cmd, **kw: (
		launched.append(list(cmd)),
		_Completed(),
	)[1]
	try:
		with quiet():
			sweep.sweep(
				task='cifar100',
				model='vit',
				methods=sweep.METHODS,
				epsilons=sweep.EPSILONS,
				seeds=[42, 43],
			)
	finally:
		sweep.subprocess.run = real_run

	expected = len(sweep.METHODS) * len(sweep.EPSILONS) * 2
	check(
		'sweep launches every (seed, method, epsilon) combination',
		len(launched) == expected,
		f'{len(launched)}/{expected} launched',
	)
	check(
		'sweep passes only strings to subprocess',
		all(isinstance(arg, str) for cmd in launched for arg in cmd),
		'a float or Path in argv raises TypeError at launch',
	)
	check(
		'sweep forwards --seed to experiment.py',
		all('--seed' in cmd for cmd in launched),
		'otherwise every run silently uses the default seed',
	)


# ======================================================================
# unit suite
# ======================================================================
def unit_privacy() -> None:
	"""
	The DP primitives: flat clipping against stock Opacus, the noise
	scale, per-layer median clipping, and the strategy guard.
	"""
	print('\n--- privacy ---')

	from federated_lora.methods import METHODS
	from federated_lora.privacy import (
		add_noise,
		clip_and_accumulate,
		privatize,
		zero_grad,
	)

	batch = make_batch(7)

	def one_step(stock: bool):
		model, optimizer, _, _ = make_private_client(seed=3)
		model(**batch).loss.backward()
		if stock:
			optimizer.step()
		else:
			params = privatize(model, optimizer)
			optimizer.original_optimizer.step()
			zero_grad(params)
		return snapshot(model)

	stock, unrolled = one_step(True), one_step(False)
	diff = max((stock[k] - unrolled[k]).abs().max().item() for k in stock)
	check(
		'flat clipping matches stock Opacus exactly',
		diff < 1e-6,
		f'max parameter difference {diff:.2e}',
	)

	# the noise must scale with sigma, not with any hardcoded constant
	torch.manual_seed(0)
	sigma, threshold = 5.0, 2.0
	param = nn.Parameter(torch.zeros(4000))
	draws = []
	for _ in range(50):
		param.summed_grad = torch.zeros(4000)
		add_noise({param: threshold}, noise_multiplier=sigma)
		draws.append(param.summed_grad.clone())
	measured = torch.stack(draws).std().item()
	check(
		'Gaussian noise std is sigma * clipping threshold',
		abs(measured - sigma * threshold) / (sigma * threshold) < 0.05,
		f'expected {sigma * threshold:.3f}, measured {measured:.4f}',
	)

	# per-layer median clipping against a direct reference
	torch.manual_seed(1)
	p1 = nn.Parameter(torch.zeros(6, 3))
	p2 = nn.Parameter(torch.zeros(4))
	lot = 10
	p1.grad_sample = torch.randn(lot, 6, 3) * 3.0
	p2.grad_sample = torch.randn(lot, 4) * 0.5
	thresholds = clip_and_accumulate([p1, p2], 'median', max_grad_norm=99.0)

	ok, detail = True, []
	for p in (p1, p2):
		norms = p.grad_sample.reshape(lot, -1).norm(2, dim=-1)
		median = norms.median().clamp(min=1e-6)
		factor = (median / (norms + 1e-6)).clamp(max=1.0)
		reference = torch.einsum('i,i...->...', factor, p.grad_sample)
		ok &= torch.allclose(p.summed_grad, reference, atol=1e-6)
		ok &= abs(thresholds[p] - float(median)) < 1e-6
		detail.append(
			f'{int((norms > median).sum())}/{lot} clipped to '
			f'{float(median):.3f}'
		)
	check(
		'median clipping clips each layer to its own median',
		ok,
		'; '.join(detail),
	)

	model, optimizer, _, _ = make_private_client(seed=5)
	optimizer.clipping_strategy = 'medain'
	model(**batch).loss.backward()
	try:
		METHODS['dp_lora']().step(
			model=model, optimizer=optimizer, round=1, step=0, batch_size=16
		)
		check('a misspelled clipping strategy raises', False, 'no exception')
	except ValueError as error:
		check('a misspelled clipping strategy raises', True, str(error))


def unit_methods() -> None:
	"""
	Each method's update rule: which factors move, on which schedule, and
	whether a frozen factor leaks into the clipping norm.
	"""
	print('\n--- methods ---')

	from federated_lora.methods import METHODS
	from federated_lora.privacy import privatize

	batch = make_batch(11)

	def one_step(method_name, clipping, round=1, step=0, seed=5):
		model, optimizer, _, engine = make_private_client(
			seed=seed, method_name=method_name, clipping=clipping
		)
		before = snapshot(model)
		model(**batch).loss.backward()
		METHODS[method_name]().step(
			model=model,
			optimizer=optimizer,
			round=round,
			step=step,
			batch_size=16,
		)
		return updated_parameters(before, model), engine

	# every method runs under every clipping strategy
	for method_name in sorted(METHODS):
		for clipping in ('flat', 'median', 'none'):
			try:
				changed, _ = one_step(method_name, clipping)
				ok, detail = bool(changed), f'{len(changed)} tensors updated'
			except Exception as error:
				ok = False
				detail = f'{type(error).__name__}: {error}'
			check(f'{method_name} steps under {clipping} clipping', ok, detail)

	# RoLoRA alternates by communication round
	for clipping in ('flat', 'median'):
		odd, _ = one_step('rolora', clipping, round=1)
		even, _ = one_step('rolora', clipping, round=2)
		check(
			f'rolora ({clipping}) updates B on odd rounds, A on even',
			any('lora_B' in n for n in odd)
			and not any('lora_A' in n for n in odd)
			and any('lora_A' in n for n in even)
			and not any('lora_B' in n for n in even),
			f'odd={sorted(odd)}',
		)

	# LA-LoRA alternates by local step, starting with B
	for clipping in ('flat', 'median'):
		first, _ = one_step('la_lora', clipping, step=0)
		second, _ = one_step('la_lora', clipping, step=1)
		check(
			f'la_lora ({clipping}) updates B first, then A',
			any('lora_B' in n for n in first)
			and not any('lora_A' in n for n in first)
			and any('lora_A' in n for n in second)
			and not any('lora_B' in n for n in second),
			f'step0={sorted(first)}',
		)

	# FFA-LoRA freezes A at the parameter level
	torch.manual_seed(2)
	model = TinyLoRA()
	METHODS['ffa_lora']().set_target_modules(model)
	trainable = {n for n, p in model.named_parameters() if p.requires_grad}
	check(
		'ffa_lora trains only B and the classifier head',
		trainable == {'lora_B.weight', 'classifier.weight'},
		f'{sorted(trainable)}',
	)

	# a frozen factor must not consume clipping budget
	model, optimizer, _, _ = make_private_client(seed=5)
	model(**batch).loss.backward()
	trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
	active = {n: p for n, p in trainable.items() if 'lora_A' not in n}
	norms = torch.stack(
		[
			p.grad_sample.reshape(16, -1).norm(2, dim=-1)
			for p in active.values()
		],
		dim=0,
	).norm(2, dim=0)
	factor = (optimizer.max_grad_norm / (norms + 1e-6)).clamp(max=1.0)
	reference = {
		n: torch.einsum('i,i...->...', factor, p.grad_sample)
		/ optimizer.expected_batch_size
		for n, p in active.items()
	}
	for name, p in trainable.items():
		if 'lora_A' in name:
			p.grad_sample = None
			p.grad = None
	privatize(model, optimizer)
	check(
		'a frozen factor is excluded from the clipping norm',
		all(
			torch.allclose(trainable[n].grad, reference[n], atol=1e-6)
			for n in reference
		),
	)

	# the smoothing kernel, against a numpy reference
	method = METHODS['la_lora']()
	kernel = np.array([1, 4, 6, 4, 1]) / 16.0

	grad_a = torch.randn(4, 9)
	smoothed_a = method.smooth(grad_a, mode='row')
	reference_a = np.stack(
		[
			np.convolve(
				np.pad(row.numpy(), 2, 'reflect'), kernel[::-1], 'valid'
			)
			for row in grad_a
		]
	)
	check(
		'la_lora smooths A along its input axis (row-wise)',
		np.allclose(smoothed_a.numpy(), reference_a, atol=1e-6)
		and smoothed_a.shape == grad_a.shape,
	)

	grad_b = torch.randn(9, 4)
	smoothed_b = method.smooth(grad_b, mode='col')
	reference_b = np.stack(
		[
			np.convolve(
				np.pad(col.numpy(), 2, 'reflect'), kernel[::-1], 'valid'
			)
			for col in grad_b.T.contiguous()
		]
	).T
	check(
		'la_lora smooths B along its output axis (column-wise)',
		np.allclose(smoothed_b.numpy(), reference_b, atol=1e-6)
		and smoothed_b.shape == grad_b.shape,
	)

	# aggregation follows each method's own paper
	uploads = [{'w': torch.tensor([1.0])}, {'w': torch.tensor([3.0])}]
	sizes = [1, 3]
	uniform = {
		name: METHODS[name]().aggregate(uploads, sizes)['w'].item()
		for name in ('rolora', 'la_lora')
	}
	weighted = {
		name: METHODS[name]().aggregate(uploads, sizes)['w'].item()
		for name in ('dp_lora', 'ffa_lora')
	}
	check(
		'rolora and la_lora aggregate uniformly (1/|C|)',
		all(abs(v - 2.0) < 1e-6 for v in uniform.values()),
		f'{uniform}',
	)
	check(
		'dp_lora and ffa_lora aggregate weighted by data size',
		all(abs(v - 2.5) < 1e-6 for v in weighted.values()),
		f'{weighted}',
	)


def unit_plumbing() -> None:
	"""
	Weight transport through the Opacus wrapper, windowed gradients, and
	the Poisson loader's edge cases.
	"""
	print('\n--- plumbing ---')

	from federated_lora.data import cycle
	from federated_lora.methods import METHODS
	from federated_lora.model import (
		dewindow_grad_samples,
		get_trainable_state,
		load_trainable_state,
	)

	# broadcast must survive the '_module.' prefix the wrapper introduces
	torch.manual_seed(4)
	global_model = TinyLoRA()
	METHODS['dp_lora']().set_target_modules(global_model)
	client_model, _, dataloader, _ = make_private_client(seed=4)

	global_state = get_trainable_state(global_model)
	client_state = get_trainable_state(client_model)
	check(
		'global and wrapped-client state dicts share a key space',
		set(global_state) == set(client_state) and global_state,
		f'{len(global_state)} trainable tensors',
	)

	sentinel = {
		k: torch.full_like(v, 3.14)
		for k, v in global_state.items()
		if 'lora_A' in k
	}
	load_trainable_state(client_model, sentinel)
	landed = get_trainable_state(client_model)
	check(
		'broadcast weights land inside the wrapped module',
		all(torch.allclose(landed[k], sentinel[k]) for k in sentinel),
		'a silent no-op here means the global model never trains',
	)

	# windowed per-sample gradients (Swin) collapse to one row per example
	torch.manual_seed(3)
	model = TinyLoRA()
	METHODS['dp_lora']().set_target_modules(model)
	lot, windows = 4, 3
	reference = {}
	for name, p in model.named_parameters():
		if p.requires_grad:
			p.grad_sample = torch.randn(lot * windows, *p.shape)
			reference[name] = p.grad_sample.reshape(
				lot, windows, *p.shape
			).sum(dim=1)
	dewindow_grad_samples(model, batch_size=lot)
	ok = all(
		torch.allclose(p.grad_sample, reference[n], atol=1e-6)
		for n, p in model.named_parameters()
		if p.requires_grad
	)
	param = next(p for p in model.parameters() if p.requires_grad)
	param.grad_sample = torch.randn(lot, *param.shape)
	untouched = param.grad_sample.clone()
	dewindow_grad_samples(model, batch_size=lot)
	check(
		'windowed gradients collapse; per-example ones pass through',
		ok and torch.equal(param.grad_sample, untouched),
	)

	check(
		'make_private preserves the Poisson sampling rate',
		abs(dataloader.sample_rate - 1.0 / (64 // 16)) < 1e-9,
		f'q={dataloader.sample_rate}',
	)

	# empty lots are possible under Poisson sampling; they must not poison
	# the loss, and the client loop must skip them
	torch.manual_seed(0)
	_, _, tiny_loader, _ = make_private_client(
		seed=43, num_examples=400, batch_size=1
	)
	batches = iter(cycle(tiny_loader))
	empty = None
	for _ in range(400):
		candidate = next(batches)
		if candidate['pixel_values'].shape[0] == 0:
			empty = candidate
			break
	if empty is None:
		check(
			'empty Poisson lot reaches the client loop as a dict',
			True,
			'no empty lot drawn in 400 tries (inconclusive)',
		)
	else:
		check(
			'empty Poisson lot reaches the client loop as a dict',
			isinstance(empty, dict),
			f'shapes={[tuple(v.shape) for v in empty.values()]}',
		)
	source = (REPO / 'federated_lora' / 'client.py').read_text()
	check(
		'client.py skips empty lots',
		'batch_size == 0' in source,
		'an empty lot yields a nan loss otherwise',
	)


def unit_loop() -> None:
	"""
	A full two-client, two-round federated loop, including the privacy
	accounting the results JSON reports.
	"""
	print('\n--- federated loop ---')

	from federated_lora.client import Client
	from federated_lora.data import provide_dataloader
	from federated_lora.methods import METHODS
	from federated_lora.model import get_trainable_state
	from federated_lora.server import Server

	def build_server(method_name, clipping, noise_multiplier, rounds=2):
		torch.manual_seed(50)
		global_model = TinyLoRA()
		method = METHODS[method_name]()
		method.set_target_modules(global_model)

		clients = []
		for i in range(2):
			local = copy.deepcopy(global_model)
			dataloader = provide_dataloader(
				make_dataset(100 + i, 48), 16, collate, False
			)
			groups = {'lora_A': [], 'lora_B': [], 'head': []}
			for name, p in local.named_parameters():
				if not p.requires_grad:
					continue
				key = (
					'lora_A'
					if 'lora_A' in name
					else 'lora_B'
					if 'lora_B' in name
					else 'head'
				)
				groups[key].append(p)
			optimizer = SGD(
				[
					{'params': groups['lora_A'], 'lr': 0.1},
					{'params': groups['lora_B'], 'lr': 0.1},
					{'params': groups['head'], 'lr': 0.1},
				]
			)
			engine = PrivacyEngine(accountant='rdp')
			local, optimizer, dataloader = engine.make_private(
				module=local,
				optimizer=optimizer,
				data_loader=dataloader,
				noise_multiplier=noise_multiplier,
				max_grad_norm=2.0,
			)
			optimizer.clipping_strategy = clipping
			clients.append(
				Client(
					id=i,
					model=local,
					dataloader=dataloader,
					optimizer=optimizer,
					scheduler=ExponentialLR(optimizer, gamma=0.99),
					privacy_engine=engine,
					num_examples=48,
					steps=3,
					device=torch.device('cpu'),
				)
			)

		test_dataloader = DataLoader(
			make_dataset(999, 32), batch_size=16, collate_fn=collate
		)
		return global_model, Server(
			model=global_model,
			method=method,
			rounds=rounds,
			clients=clients,
			sample_rate=1.0,
			test_dataloader=test_dataloader,
			delta=1e-5,
			device=torch.device('cpu'),
			seed=0,
		)

	global_model, server = build_server('dp_lora', 'flat', 0.0)
	before = get_trainable_state(global_model)
	with quiet():
		history = server.run()
	after = get_trainable_state(global_model)

	check(
		'the global model changes after aggregation',
		any(not torch.equal(before[k], after[k]) for k in before),
		'empty client uploads would leave it untouched',
	)
	check(
		'one history record per round',
		len(history) == 2,
		f'{len(history)} records',
	)
	check(
		'losses and accuracies are finite',
		all(
			np.isfinite(h['train_loss']) and np.isfinite(h['test_loss'])
			for h in history
		),
		f'train_loss {history[0]["train_loss"]:.3f} -> '
		f'{history[-1]["train_loss"]:.3f}',
	)
	check(
		'learning rates are snapshotted before decay',
		abs(history[0]['lr_A'] - 0.1) < 1e-9
		and abs(history[1]['lr_A'] - 0.099) < 1e-9,
		f'{[h["lr_A"] for h in history]}',
	)
	check(
		'no privacy is spent when sigma is zero',
		all(h['epsilon_spent'] is None for h in history),
		f'{[h["epsilon_spent"] for h in history]}',
	)

	# accounting must advance on both tracks, and identically
	spent = {}
	for method_name, clipping in (
		('dp_lora', 'flat'),
		('ffa_lora', 'median'),
		('rolora', 'median'),
		('la_lora', 'median'),
	):
		_, server = build_server(method_name, clipping, 1.0, rounds=3)
		with quiet():
			rounds_history = server.run()
		spent[f'{method_name}/{clipping}'] = [
			round(h['epsilon_spent'], 4) for h in rounds_history
		]

	first = next(iter(spent.values()))
	check(
		'privacy spend increases monotonically each round',
		all(
			curve == sorted(curve) and curve[0] > 0 for curve in spent.values()
		),
		f'{first}',
	)
	check(
		'both clipping tracks account identically',
		all(curve == first for curve in spent.values()),
		'the stock and unrolled paths must charge the same budget',
	)


def unit_suite() -> None:
	print('\n=== unit ===')
	unit_privacy()
	unit_methods()
	unit_plumbing()
	unit_loop()


# ======================================================================
# smoke suite
# ======================================================================
def smoke_run(
	task: str,
	model: str,
	method: str,
	epsilon: float,
	rounds: int,
	steps: int,
	num_clients: int,
	seed: int,
	dry_run: bool,
) -> None:
	"""
	Run ``experiment.py`` end to end with tiny round counts and validate
	the results JSON it writes.
	"""
	output_dir = REPO / 'results' / 'smoke' / task / model
	path = (
		output_dir / f'{method}_{model}_{task}_eps{epsilon:g}_seed{seed}.json'
	)
	if path.exists():
		path.unlink()

	cmd = [
		PYTHON,
		str(REPO / 'experiment.py'),
		'--task',
		task,
		'--model',
		model,
		'--method',
		method,
		'--epsilon',
		str(epsilon),
		'--global_rounds',
		str(rounds),
		'--local_steps',
		str(steps),
		'--num_clients',
		str(num_clients),
		'--seed',
		str(seed),
		'--output_dir',
		str(output_dir),
	]

	label = f'{task}/{model}/{method}'
	if dry_run:
		check(f'{label}: command', True, ' '.join(cmd[1:]))
		return

	print(f'  running {label} ...', flush=True)
	result = subprocess.run(
		cmd, cwd=REPO, capture_output=True, text=True, check=False
	)

	if result.returncode != 0:
		tail = (result.stdout or '')[-1500:]
		check(f'{label}: experiment.py exits cleanly', False, 'output below')
		print('\n' + tail + '\n')
		return
	check(f'{label}: experiment.py exits cleanly', True)

	if not check(f'{label}: results JSON written', path.exists(), str(path)):
		return

	result_json = json.loads(path.read_text())
	history = result_json.get('history', [])

	check(
		f'{label}: one record per round',
		len(history) == rounds,
		f'{len(history)} records',
	)
	if not history:
		return

	expected_keys = {
		'round_index',
		'train_loss',
		'test_loss',
		'top1_acc',
		'top5_acc',
		'lr_A',
		'lr_B',
		'lr_head',
		'epsilon_spent',
	}
	missing = expected_keys - set(history[0])
	check(
		f'{label}: history schema complete',
		not missing,
		f'missing {sorted(missing)}' if missing else '',
	)
	check(
		f'{label}: config recorded for provenance',
		{'method', 'model', 'task', 'epsilon', 'seed'}
		<= set(result_json.get('config', {})),
	)

	last = history[-1]
	check(
		f'{label}: metrics finite and in range',
		all(np.isfinite(h['train_loss']) for h in history)
		and all(np.isfinite(h['test_loss']) for h in history)
		and 0.0 <= last['top1_acc'] <= 100.0,
		f'train_loss={last["train_loss"]:.4f}, top1={last["top1_acc"]:.2f}%',
	)
	check(
		f'{label}: learning rate decays across rounds',
		history[0]['lr_A'] > last['lr_A'] if rounds > 1 else True,
		f'{history[0]["lr_A"]:.5f} -> {last["lr_A"]:.5f}',
	)

	spent = [h['epsilon_spent'] for h in history]
	check(
		f'{label}: privacy spend tracked and increasing',
		all(e is not None and e > 0 for e in spent) and spent == sorted(spent),
		f'eps {spent[0]:.4f} -> {spent[-1]:.4f} after {rounds} rounds'
		if spent[0] is not None
		else f'{spent}',
	)


def smoke_suite(args: argparse.Namespace) -> None:
	print('\n=== smoke ===')
	print(
		f'  device: '
		f'{torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}',
		flush=True,
	)
	if not torch.cuda.is_available() and not args.dry_run:
		print('  WARNING: no CUDA device — this will be slow', flush=True)

	for model in args.models:
		for method in args.methods:
			smoke_run(
				task=args.task,
				model=model,
				method=method,
				epsilon=args.epsilon,
				rounds=args.rounds,
				steps=args.steps,
				num_clients=args.num_clients,
				seed=args.seed,
				dry_run=args.dry_run,
			)


# ======================================================================
# entry point
# ======================================================================
def main():
	parser = argparse.ArgumentParser(
		description='Pre-flight checks for the federated LoRA harness.'
	)

	parser.add_argument(
		'--suite',
		type=str,
		default='fast',
		choices=['fast', 'audit', 'unit', 'smoke', 'all'],
		help="'fast' (default) runs audit + unit",
	)

	parser.add_argument('--task', type=str, default='cifar100')
	parser.add_argument(
		'--models', type=str, nargs='+', default=['vit', 'swin']
	)
	parser.add_argument(
		'--methods',
		type=str,
		nargs='+',
		default=['dp_lora', 'ffa_lora', 'rolora', 'la_lora'],
	)
	parser.add_argument('--epsilon', type=float, default=3.0)

	parser.add_argument('--rounds', type=int, default=2)
	parser.add_argument('--steps', type=int, default=2)
	parser.add_argument(
		'--num_clients',
		type=int,
		default=8,
		help='kept at the sweep default so the prepared shards are reused',
	)
	parser.add_argument('--seed', type=int, default=42)
	parser.add_argument(
		'--dry-run',
		action='store_true',
		help='print the smoke commands without running them',
	)

	args = parser.parse_args()

	if args.suite in ('fast', 'audit', 'all'):
		audit_suite()
	if args.suite in ('fast', 'unit', 'all'):
		unit_suite()
	if args.suite in ('smoke', 'all'):
		smoke_suite(args)

	failed = [name for name, ok, _ in RESULTS if not ok]
	print('\n' + '=' * 68)
	print(
		f'{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed'
		if not failed
		else f'{len(failed)} of {len(RESULTS)} checks FAILED'
	)
	for name in failed:
		print(f'  FAILED: {name}')
	print('=' * 68)

	raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
	main()

from __future__ import annotations

import torch
import torch.nn as nn


def summarize_diagnostics(step_diags: list[dict]) -> dict:
	"""
	Average per-step privacy diagnostics into one per-round summary.

	Parameters
	----------
	step_diags : list[dict]
		The per-step diagnostics stashed by ``privatize`` (may be shorter than
		``local_steps`` when Poisson sampling yields empty lots).

	Returns
	-------
	dict
		Mean pre-clip gradient norm (mean/median), the clipping strategy, the
		mean clipping threshold over layers and steps, and the per-layer mean
		clipping threshold (keys shortened; ``None`` when clipping is off).
	"""
	if not step_diags:
		return {
			'steps_recorded': 0,
			'grad_norm_mean': None,
			'grad_norm_median': None,
			'clip_strategy': None,
			'clip_threshold_mean': None,
			'clip_thresholds': {},
		}

	def _finite_mean(values: list[float]) -> float | None:
		finite = [v for v in values if v is not None and v != float('inf')]
		return sum(finite) / len(finite) if finite else None

	grad_norm_mean = _finite_mean([d['grad_norm_mean'] for d in step_diags])
	grad_norm_median = _finite_mean(
		[d['grad_norm_median'] for d in step_diags]
	)

	# per-layer thresholds, averaged over steps, with shortened keys
	per_layer: dict[str, list[float]] = {}
	for d in step_diags:
		for name, c in d['clip_thresholds'].items():
			per_layer.setdefault(_short_layer(name), []).append(c)
	clip_thresholds = {k: _finite_mean(v) for k, v in per_layer.items()}

	layer_means = [v for v in clip_thresholds.values() if v is not None]
	clip_threshold_mean = (
		sum(layer_means) / len(layer_means) if layer_means else None
	)

	return {
		'steps_recorded': len(step_diags),
		'grad_norm_mean': grad_norm_mean,
		'grad_norm_median': grad_norm_median,
		'clip_strategy': step_diags[-1]['strategy'],
		'clip_threshold_mean': clip_threshold_mean,
		'clip_thresholds': clip_thresholds,
	}


def _short_layer(name: str) -> str:
	"""Trim the long PEFT parameter path down to a readable layer key."""
	return (
		name.replace('base_model.model.', '')
		.replace('.default.weight', '')
		.replace('.weight', '')
	)


def aggregate_diags(diags: list[dict]) -> dict:
	"""Average the selected clients' per-round diagnostics."""
	diags = [d for d in diags if d and d.get('steps_recorded')]
	if not diags:
		return {
			'grad_norm_mean': None,
			'grad_norm_median': None,
			'clip_strategy': None,
			'clip_threshold_mean': None,
			'clip_thresholds': {},
		}

	def _mean(values: list[float | None]) -> float | None:
		vals = [v for v in values if v is not None]
		return sum(vals) / len(vals) if vals else None

	# per-layer thresholds averaged across the selected clients
	per_layer: dict[str, list[float]] = {}
	for d in diags:
		for name, c in d['clip_thresholds'].items():
			if c is not None:
				per_layer.setdefault(name, []).append(c)

	return {
		'grad_norm_mean': _mean([d['grad_norm_mean'] for d in diags]),
		'grad_norm_median': _mean([d['grad_norm_median'] for d in diags]),
		'clip_strategy': diags[-1]['clip_strategy'],
		'clip_threshold_mean': _mean(
			[d['clip_threshold_mean'] for d in diags]
		),
		'clip_thresholds': {
			name: sum(v) / len(v) for name, v in per_layer.items()
		},
	}


def pre_clip_norm_stats(params: list[nn.Parameter]) -> tuple[float, float]:
	"""
	Summarize the per-example *global* gradient norm before clipping.

	The global norm is the L2 norm over all trainable parameters, computed per
	example, then reduced to its mean and median across the lot. This is a
	strategy-independent measure of gradient scale, useful for spotting the
	high-lr divergence regime in the metrics.

	Parameters
	----------
	params : list[nn.Parameter]
		The list of parameters with per-example gradients.

	Returns
	-------
	tuple[float, float]
		The mean and median of the per-example global gradient norm.
	"""
	per_example = torch.stack(
		[
			p.grad_sample.reshape(len(p.grad_sample), -1).norm(2, dim=-1)
			for p in params
		],
		dim=0,
	).norm(2, dim=0)
	return float(per_example.mean()), float(per_example.median())

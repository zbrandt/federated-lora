#!/usr/bin/env python
"""
analyze_aggregation_error.py — offline aggregation-error analysis.

Reads the raw per-client (lora_A, lora_B, ...) uploads that server.py saves
each round when Config.log_uploads_dir / experiment.py's --log-uploads-dir is
set, and for each requested method computes how far that method's actual
aggregation (via its real Method.aggregate()) deviates from the "ideal"
full-rank average of every client's true delta_W = B_i @ A_i.

This deliberately reuses the real Method.aggregate() implementations in
federated_lora/methods/*.py rather than reimplementing each method's
reconciliation scheme here, so it's testing the exact aggregation code that
runs during training, not a parallel copy that could drift out of sync.

Only rblora, hetlora, and flora make sense to test this way under rank
heterogeneity:
  - ffa_lora/dp_lora/la_lora/rolora aggregate via plain Server.fedavg(),
    which requires every client's tensors to already be the same shape, so
    they'll raise on genuinely ragged per-client ranks.
  - flexlora's aggregate() is a no-op pass-through (it literally returns one
    client's raw upload unchanged) -- its real SVD-based reconciliation only
    happens in redistribute(), which needs each client's live PEFT model to
    read layer.scaling['default'], not just the saved tensors. Computing an
    "error" from its aggregate() alone would silently measure something
    meaningless. Excluded from the default --methods list for that reason;
    including it properly would need extending server.py's capture hook to
    also save each client's per-layer LoRA scaling factor (a cheap float)
    and reimplementing redistribute()'s SVD math in a tensors-only form here.

Usage:
    python analyze_aggregation_error.py --uploads-dir logs/uploads/rankspread_d4 \\
        --methods rblora hetlora flexlora flora --label spread4 \\
        --out results/aggregation_error/spread4.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from federated_lora.methods import get_method

# the lora update is computed in A/B, and to find the implied update that has full rank from each invidvidual client Ai and Bi we can calculate that here
# bc this is pre aggregation
def true_delta(b_i: torch.Tensor, a_i: torch.Tensor) -> torch.Tensor:
    return b_i.float() @ a_i.float()

# takes every true delta and adds them and then agerages them
# this is essentially the benchmark because the question we are asking here is how much aggregation strategies and compression come into play
def ideal_delta(uploads: dict[int, dict[str, torch.Tensor]], a_key: str, b_key: str) -> torch.Tensor:
    deltas = [true_delta(u[b_key], u[a_key]) for u in uploads.values()]
    return torch.stack(deltas, dim=0).mean(dim=0)

# frobenius error is the way of measuing the total size of a matrix. 
# this function will measure the size of the difference between the method's actual aggregated result and the ideal
# then turns it into a percentage of the ideal
def relative_frobenius_error(aggregated: torch.Tensor, ideal: torch.Tensor) -> float:
    denom = torch.linalg.norm(ideal, ord='fro').item()
    if denom < 1e-12:
        return float('nan')
    return torch.linalg.norm(aggregated - ideal, ord='fro').item() / denom

# loads the methods real code, runs it on the saved raw client data, figures out which layers exist, and then for each layer computes the ideal then averages the error across all layers for round
def per_round_error(method_name: str, uploads, num_examples) -> float:
    method = get_method(method_name)
    _, eval_state = method.aggregate(uploads=uploads, num_examples=num_examples)

    reference = next(iter(uploads.values()))
    a_keys = [k for k in reference if 'lora_A' in k]

    errors = []
    for a_key in a_keys:
        b_key = a_key.replace('lora_A', 'lora_B')
        ideal = ideal_delta(uploads, a_key, b_key)
        aggregated = eval_state[b_key].float() @ eval_state[a_key].float()
        errors.append(relative_frobenius_error(aggregated, ideal))

    return sum(errors) / len(errors)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--uploads-dir', type=str, required=True, help='directory of round*.pt files saved via --log-uploads-dir')
    parser.add_argument('--methods', type=str, nargs='+', default=['rblora', 'hetlora', 'flora'])
    parser.add_argument('--label', type=str, default='', help='tag for this run (e.g. the rank-spread level), stamped into every output row')
    parser.add_argument('--out', type=str, required=True, help='output CSV path')
    args = parser.parse_args()

    uploads_dir = Path(args.uploads_dir)
    round_files = sorted(uploads_dir.glob('round*.pt'))
    if not round_files:
        raise SystemExit(f'no round*.pt files found in {uploads_dir}')

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'method', 'round', 'aggregation_error'])
        for round_file in round_files:
            round_num = int(round_file.stem.replace('round', ''))
            saved = torch.load(round_file, weights_only=False, map_location='cpu')
            uploads, num_examples = saved['uploads'], saved['num_examples']

            for method_name in args.methods:
                error = per_round_error(method_name, uploads, num_examples)
                writer.writerow([args.label, method_name, round_num, error])
                print(f'{args.label:10} {method_name:9} round {round_num:4d}  error={error:.4f}', flush=True)

    print(f'\nwrote {out_path}')


if __name__ == '__main__':
    main()

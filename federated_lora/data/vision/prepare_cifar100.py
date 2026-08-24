from __future__ import annotations

from pathlib import Path

import numpy as np
from datasets import load_dataset

from federated_lora.data import partition_dirichlet


def load_datasets(
	num_clients: int,
	dirichlet_alpha: float,
	seed: int,
) -> Path:
	output_dir = (
		Path('data') / f'cifar100_{num_clients}clients_alpha{dirichlet_alpha}'
	)
	if output_dir.is_dir():
		print(f'Data already saved to {output_dir.resolve()}')
		return output_dir

	output_dir.mkdir(parents=True, exist_ok=True)

	dataset = load_dataset('uoft-cs/cifar100')

	train, test = dataset['train'], dataset['test']

	X_train = np.stack([np.array(im) for im in train['img']]).astype(np.uint8)
	y_train = np.array(train['fine_label'], dtype=np.int64)

	X_test = np.stack([np.array(im) for im in test['img']]).astype(np.uint8)
	y_test = np.array(test['fine_label'], dtype=np.int64)

	client_indices = partition_dirichlet(
		labels=y_train.tolist(),
		num_clients=num_clients,
		alpha=dirichlet_alpha,
		seed=seed,
	)

	for c, i in enumerate(client_indices):
		images, labels = X_train[i], y_train[i]
		np.savez_compressed(
			output_dir / f'client_{c}.npz', images=images, labels=labels
		)

		print(f'client_{c}: n={len(labels)}')

	np.savez_compressed(output_dir / 'test.npz', images=X_test, labels=y_test)

	print(f'Data saved to {output_dir.resolve()}')
	return output_dir

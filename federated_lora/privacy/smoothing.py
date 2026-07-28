from __future__ import annotations

import torch
import torch.nn.functional as F

# fixed 5-tap binomial kernel, normalized to preserve the update's scale
_KERNEL = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0

# axis of the low-pass filter for the active matrix:
#   A is (r x n) -> smooth along the input dim n (last axis)
#   B is (m x r) -> smooth along the output dim m (first axis)
AXIS_A = -1
AXIS_B = 0


def smooth(grad: torch.Tensor, axis: int) -> torch.Tensor:
	"""
	Apply the fixed [1,4,6,4,1]/16 binomial low-pass filter along ``axis``.

	Length-preserving (reflect padding), so the returned tensor has the same
	shape as ``grad`` and can be assigned straight back to ``param.grad``.
	"""
	moved = grad.movedim(axis, -1)
	shape = moved.shape
	flat = moved.reshape(-1, shape[-1]).unsqueeze(1)  # (rows, 1, L)

	kernel = _KERNEL.to(device=grad.device, dtype=grad.dtype).view(1, 1, -1)
	padded = F.pad(flat, (2, 2), mode='reflect')
	filtered = F.conv1d(padded, kernel).squeeze(1)

	return filtered.reshape(shape).movedim(-1, axis)
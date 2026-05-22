# Licensed under the MIT License. See LICENSE file for details.
# Copyright (c) 2026 Zhenyu Zhao

import torch


def hermitian(W):
    return 0.5 * (W + W.conj().transpose(-1, -2))


def unitary_from_H(W):
    Hh = hermitian(W)
    A = 1j * Hh
    return torch.matrix_exp(A)


def batch_hermitian_fast(W):
    return 0.5 * (W + torch.einsum('...ji->...ij', W.conj()))


def batch_unitary_cayley(W):
    H = batch_hermitian_fast(W)
    batch_shape = H.shape[:-2]
    n = H.shape[-1]
    I = torch.eye(n, device=H.device, dtype=H.dtype)
    I_batch = I.view(*([1] * len(batch_shape)), n, n).expand(*batch_shape, n, n)

    A = 1j * H
    return torch.linalg.solve(I_batch + A, I_batch - A)
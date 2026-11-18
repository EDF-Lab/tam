# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""Pure mathematical operations for Path Signatures and Free Lie Algebra representations.

Implements truncated graded tensor algebra, Chen's product identity, segment exponentials,
tensor inversion, tensor logarithm, Lyndon word basis extraction, and Hoff's lead-lag transform.
"""

from typing import List, Tuple
import itertools
import torch


def compute_witt_dimension(d: int, k: int) -> int:
    r"""Computes the dimension of the degree-:math:`k` homogeneous Lie component via Witt's formula.

    .. math::

        \beta(d, k) = \frac{1}{k} \sum_{m \mid k} \mu(m) d^{k/m}

    where :math:`\mu(\cdot)` is the Möbius function.

    Args:
        d (int): Alphabet / path dimensionality.
        k (int): Homogeneous degree.

    Returns:
        int: Dimension of the homogeneous Lie component.
    """
    if k <= 0:
        return 0
    if d <= 0:
        return 0

    def _mobius(n: int) -> int:
        if n == 1:
            return 1
        factors = 0
        divisor = 2
        temp = n
        while divisor * divisor <= temp:
            if temp % divisor == 0:
                factors += 1
                temp //= divisor
                if temp % divisor == 0:
                    return 0
            divisor += 1
        if temp > 1:
            factors += 1
        return -1 if factors % 2 == 1 else 1

    total = 0
    for m in range(1, k + 1):
        if k % m == 0:
            total += _mobius(m) * (d ** (k // m))
    return int(total // k)


def get_lyndon_words(d: int, k: int) -> List[Tuple[int, ...]]:
    r"""Generates all Lyndon words of exact length :math:`k` over alphabet :math:`\{0, \dots, d-1\}`.

    A word :math:`w` is a Lyndon word if it is strictly smaller lexicographically
    than all its non-trivial proper suffixes.

    Args:
        d (int): Alphabet size.
        k (int): Word length.

    Returns:
        List[Tuple[int, ...]]: Sorted list of Lyndon words of length :math:`k`.
    """
    if k <= 0 or d <= 0:
        return []

    words = []
    for candidate in itertools.product(range(d), repeat=k):
        cand_list = list(candidate)
        is_lyndon = True
        for i in range(1, k):
            if cand_list[i:] <= cand_list:
                is_lyndon = False
                break
        if is_lyndon:
            words.append(candidate)
    return words


def lyndon_basis_indices(d: int, n: int) -> List[int]:
    r"""Finds flat tensor indices of all Lyndon words up to level :math:`N`.

    Maps each Lyndon word :math:`(i_1, \dots, i_k)` to its global index in the
    truncated tensor vector.

    Args:
        d (int): Path dimensionality.
        n (int): Truncation level.

    Returns:
        List[int]: Global flat tensor indices for the Lyndon basis.
    """
    indices = []
    offset = 1  # Skip level 0 (constant 1)
    for k in range(1, n + 1):
        words = get_lyndon_words(d, k)
        for w in words:
            # Map word tuple to offset + linear index
            local_idx = 0
            for char in w:
                local_idx = local_idx * d + char
            indices.append(offset + local_idx)
        offset += d ** k
    return indices


def segment_exp(delta: torch.Tensor, n: int) -> torch.Tensor:
    r"""Computes the truncated signature of a straight line segment with increment :math:`\delta`.

    .. math::

        S^k(\delta) = \frac{\delta^{\otimes k}}{k!}

    Args:
        delta (torch.Tensor): Increment vector of shape ``(D,)``.
        n (int): Truncation level.

    Returns:
        torch.Tensor: Truncated signature of shape ``( (D^{N+1} - 1) / (D - 1), )``.
    """
    d = delta.shape[0]
    levels = [torch.tensor([1.0], dtype=delta.dtype, device=delta.device)]
    curr = delta.clone()
    levels.append(curr)

    fact = 1.0
    for k in range(2, n + 1):
        fact *= float(k)
        curr = torch.outer(curr, delta).flatten()
        levels.append(curr / fact)

    return torch.cat(levels)


def tensor_product(a: torch.Tensor, b: torch.Tensor, d: int, n: int) -> torch.Tensor:
    r"""Evaluates the truncated tensor product :math:`(a \otimes b)` of two graded tensor series.

    .. math::

        (a \otimes b)^{(k)} = \sum_{j=0}^k a^{(j)} \otimes b^{(k-j)}

    Args:
        a (torch.Tensor): Left signature series.
        b (torch.Tensor): Right signature series.
        d (int): Alphabet / path dimensionality.
        n (int): Truncation level.

    Returns:
        torch.Tensor: Truncated product series up to level :math:`N`.
    """
    offsets = [0]
    for k in range(n + 1):
        offsets.append(offsets[-1] + d ** k)

    out_levels = []
    for k in range(n + 1):
        res_k = None
        for j in range(k + 1):
            aj = a[offsets[j] : offsets[j + 1]]
            bkj = b[offsets[k - j] : offsets[k - j + 1]]
            term = torch.outer(aj, bkj).flatten()
            res_k = term if res_k is None else res_k + term
        out_levels.append(res_k)

    return torch.cat(out_levels)


def tensor_inverse(a: torch.Tensor, d: int, n: int) -> torch.Tensor:
    r"""Computes the truncated tensor inverse :math:`a^{-1}` such that :math:`a \otimes a^{-1} = 1`.

    Args:
        a (torch.Tensor): Group-like signature series with :math:`a^{(0)} = 1`.
        d (int): Alphabet / path dimensionality.
        n (int): Truncation level.

    Returns:
        torch.Tensor: Inverse signature series up to level :math:`N`.
    """
    offsets = [0]
    for k in range(n + 1):
        offsets.append(offsets[-1] + d ** k)

    b_levels = [torch.tensor([1.0], dtype=a.dtype, device=a.device)]
    for k in range(1, n + 1):
        bk = torch.zeros(d ** k, dtype=a.dtype, device=a.device)
        for j in range(1, k + 1):
            aj = a[offsets[j] : offsets[j + 1]]
            b_prev = b_levels[k - j]
            bk = bk - torch.outer(aj, b_prev).flatten()
        b_levels.append(bk)

    return torch.cat(b_levels)


def tensor_log(sig: torch.Tensor, d: int, n: int) -> torch.Tensor:
    r"""Evaluates the truncated tensor logarithm :math:`\log^{\otimes N}(1 + \mathbf{s})`.

    .. math::

        \log^{\otimes N}(S) = \sum_{k=1}^N \frac{(-1)^{k-1}}{k} \mathbf{s}^{\otimes k}

    Args:
        sig (torch.Tensor): Group-like signature series of shape ``( (D^{N+1} - 1) / (D - 1), )``.
        d (int): Alphabet / path dimensionality.
        n (int): Truncation level.

    Returns:
        torch.Tensor: Log-signature series in the free Lie algebra.
    """
    s = sig.clone()
    s[0] = 0.0

    curr_pow = s.clone()
    acc = s.clone()

    for k in range(2, n + 1):
        curr_pow = tensor_product(curr_pow, s, d, n)
        coeff = (1.0 / float(k)) if (k % 2 == 1) else (-1.0 / float(k))
        acc = acc + coeff * curr_pow

    acc[0] = 0.0
    return acc


def lead_lag_transform(x: torch.Tensor) -> torch.Tensor:
    r"""Constructs Hoff's piecewise linear Lead-Lag path for a discrete trajectory.

    For a sequence :math:`\{x_0, \dots, x_{T-1}\} \subset \mathbb{R}^d`, embeds the path
    into :math:`\mathbb{R}^{2d}` on a :math:`(2T - 1)`-step grid:

    .. math::

        X^{\text{lead-lag}}_{2k} = (x_k, x_k), \qquad X^{\text{lead-lag}}_{2k+1} = (x_k, x_{k+1})

    Args:
        x (torch.Tensor): Trajectory of shape ``(T, d)``.

    Returns:
        torch.Tensor: Lead-lag path of shape ``(2*T - 1, 2*d)``.
    """
    t_len, d = x.shape
    if t_len <= 1:
        return torch.cat([x, x], dim=-1)

    out = torch.zeros(2 * t_len - 1, 2 * d, dtype=x.dtype, device=x.device)
    # First point: (x_0, x_0)
    out[0, :d] = x[0]
    out[0, d:] = x[0]

    for k in range(t_len - 1):
        # Step 2k + 1: (x_k, x_{k+1})
        out[2 * k + 1, :d] = x[k]
        out[2 * k + 1, d:] = x[k + 1]

        # Step 2k + 2: (x_{k+1}, x_{k+1})
        out[2 * k + 2, :d] = x[k + 1]
        out[2 * k + 2, d:] = x[k + 1]

    return out

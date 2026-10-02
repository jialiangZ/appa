import torch

from functools import lru_cache
from numpy.polynomial.legendre import leggauss
from torch import Tensor
from typing import Callable, Iterable, Optional, Sequence, Tuple


def gmres(
    A: Callable[[Tensor], Tensor],
    b: Tensor,
    x0: Optional[Tensor] = None,
    iterations: int = 1,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    r"""Solves a linear system :math:`Ax = b` with generalized minimal residual (GMRES) iterations.

    The matrix :math:`A \in \mathbb{R}^{D \times D}` can be non-symmetric non-definite.

    Wikipedia:
        https://wikipedia.org/wiki/Generalized_minimal_residual_method

    Warning:
        This function is optimized for GPU execution. To avoid CPU-GPU communication,
        all iterations are performed regardless of convergence.

    Arguments:
        A: The linear operator :math:`x \mapsto Ax`.
        b: The right-hand side vector :math:`b`, with shape :math:`(*, D)`.
        x0: An initial guess :math:`x_0`, with shape :math:`(*, D)`. If :py:`None`, use
            :math:`x_0 = 0` instead.
        iterations: The number of GMRES iterations :math:`n`.
        dtype: The data type used for intermediate computations. If :py:`None`, use
            :class:`torch.float64` instead.

    Returns:
        The :math:`n`-th iteration :math:`x_n`, with shape :math:`(*, D)`.
    """

    if dtype is None:
        dtype = torch.float64

    epsilon = torch.finfo(dtype).smallest_normal

    if x0 is None:
        r = b
    else:
        r = b - A(x0)

    r = r.to(dtype)

    def normalize(x):
        norm = torch.linalg.vector_norm(x, dim=-1)
        x = x / torch.clip(norm[..., None], min=epsilon)

        return x, norm

    def rotation(a, b):
        c = torch.clip(torch.sqrt(a * a + b * b), min=epsilon)
        return a / c, -b / c

    V = [None for _ in range(iterations + 1)]
    B = [None for _ in range(iterations + 1)]
    H = [[None for _ in range(iterations)] for _ in range(iterations + 1)]
    cs = [None for _ in range(iterations)]
    ss = [None for _ in range(iterations)]

    V[0], B[0] = normalize(r)

    for j in range(iterations):
        v = V[j].to(b)
        w = A(v).to(dtype)

        # Apply Arnoldi iteration to get the j+1-th basis
        for i in range(j + 1):
            H[i][j] = torch.einsum("...i,...i", w, V[i])
            w = w - H[i][j][..., None] * V[i]
        w, w_norm = normalize(w)
        H[j + 1][j] = w_norm
        V[j + 1] = w

        # Apply Givens rotation
        for i in range(j):
            tmp = cs[i] * H[i][j] - ss[i] * H[i + 1][j]
            H[i + 1][j] = cs[i] * H[i + 1][j] + ss[i] * H[i][j]
            H[i][j] = tmp

        cs[j], ss[j] = rotation(H[j][j], H[j + 1][j])
        H[j][j] = cs[j] * H[j][j] - ss[j] * H[j + 1][j]

        # Update residual vector
        B[j + 1] = ss[j] * B[j]
        B[j] = cs[j] * B[j]

        # Fill with zeros
        for i in range(j + 1, iterations + 1):
            H[i][j] = torch.zeros_like(H[j][j])

    V, B, H = V[:-1], B[:-1], H[:-1]

    V = torch.stack(V, dim=-2)
    B = torch.stack(B, dim=-1)
    H = torch.stack([torch.stack(Hi, dim=-1) for Hi in H], dim=-2)

    y = torch.linalg.solve_triangular(
        H + epsilon * torch.eye(iterations, dtype=dtype, device=H.device),
        B.unsqueeze(dim=-1),
        upper=True,
    ).squeeze(dim=-1)

    if x0 is None:
        x = torch.einsum("...ij,...i", V, y)
    else:
        x = x0 + torch.einsum("...ij,...i", V, y)

    return x.to(b)


class GaussLegendre(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        f: Callable[[Tensor], Tensor],
        a: Tensor,
        b: Tensor,
        n: int,
        *phi: Tensor,
    ) -> Tensor:
        ctx.f, ctx.n = f, n
        ctx.save_for_backward(a, b, *phi)

        return GaussLegendre.quadrature(f, a, b, n)

    @staticmethod
    def backward(ctx, grad_area: Tensor) -> Tuple[Tensor, ...]:
        f, n = ctx.f, ctx.n
        a, b, *phi = ctx.saved_tensors

        if ctx.needs_input_grad[1]:
            grad_a = -f(a) * grad_area
        else:
            grad_a = None

        if ctx.needs_input_grad[2]:
            grad_b = f(b) * grad_area
        else:
            grad_b = None

        if phi:
            with torch.enable_grad():
                area = GaussLegendre.quadrature(f, a, b, n)

            grad_phi = torch.autograd.grad(
                area, phi, grad_area, create_graph=True, allow_unused=True
            )
        else:
            grad_phi = ()

        return (None, grad_a, grad_b, None, *grad_phi)

    @staticmethod
    @lru_cache(maxsize=None)
    def nodes(n: int, **kwargs) -> Tuple[Tensor, Tensor]:
        r"""Returns the nodes and weights for a :math:`n`-point Gauss-Legendre
        quadrature over the interval :math:`[0, 1]`.

        See :func:`numpy.polynomial.legendre.leggauss`.
        """

        nodes, weights = leggauss(n)

        nodes = (nodes + 1) / 2
        weights = weights / 2

        kwargs.setdefault("dtype", torch.get_default_dtype())

        return (
            torch.as_tensor(nodes, **kwargs),
            torch.as_tensor(weights, **kwargs),
        )

    @staticmethod
    def quadrature(
        f: Callable[[Tensor], Tensor],
        a: Tensor,
        b: Tensor,
        n: int,
    ) -> Tensor:
        nodes, weights = GaussLegendre.nodes(n, dtype=a.dtype, device=a.device)
        nodes = torch.lerp(
            a[..., None],
            b[..., None],
            nodes,
        ).movedim(-1, 0)

        return (b - a) * torch.tensordot(weights, f(nodes), dims=1)


def gauss_legendre(
    f: Callable[[Tensor], Tensor],
    a: Tensor,
    b: Tensor,
    n: int = 3,
    phi: Iterable[Tensor] = (),
) -> Tensor:
    r"""Estimates the definite integral of a function :math:`f_\phi(x)` from :math:`a`
    to :math:`b` using a :math:`n`-point Gauss-Legendre quadrature.

    .. math:: \int_a^b f_\phi(x) ~ dx \approx (b - a) \sum_{i = 1}^n w_i f_\phi(x_i)

    Wikipedia:
        https://wikipedia.org/wiki/Gauss-Legendre_quadrature

    References:
        | zuko library (François Rozet)
        | https://github.com/probabilists/zuko/

    Arguments:
        f: A univariate function :math:`f_\phi`.
        a: The lower limit :math:`a`.
        b: The upper limit :math:`b`.
        n: The number of points :math:`n` at which the function is evaluated.
        phi: The parameters :math:`\phi` of :math:`f_\phi`.

    Returns:
        The definite integral estimation.

    Example:
        >>> f = lambda x: torch.exp(-x**2)
        >>> a, b = torch.tensor([-0.69, 4.2])
        >>> gauss_legendre(f, a, b, n=16)
        tensor(1.4807)
    """

    return GaussLegendre.apply(f, a, b, n, *phi)


def str_to_ids(ids_str: str) -> Sequence[int]:
    """
    Convert a string of IDs to a list of integers.

    e.g.:
        '1,2,3'    -> [1, 2, 3]
        '1-3'      -> [1, 2, 3]
        '1,2,8-10' -> [1, 2, 8, 9, 10]

    Args:
        id_list_str: Formatted string of IDs.

    Returns:
        The list of IDs. (list of int)

    """
    id_list_str = ids_str.split(",")
    id_list = []

    for id_ in id_list_str:
        if "-" in id_:
            start, end = id_.split("-")
            id_list += list(range(int(start), int(end) + 1))
        else:
            id_list.append(int(id_))

    return id_list

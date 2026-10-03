r"""Tests for the appa.models.nn.graph module."""

import pytest
import torch

from torch.nn.functional import scaled_dot_product_attention

from appa.models.nn.graph import GraphAttention, xfa

# The legacy xformers sparse-attention API is unavailable with xformers >= 0.0.30
xfa_available = xfa is not None and xfa._has_cpp_library

param_combinations = [
    (2, 10, 32, [16, 12], [2, 2], None, 4, [10, 5]),
    (2, 32, 10, [4, 4, 8, 8], [1, 2, 1, 0], 0.1, 2, [15, 7, 12, 5]),
    (2, 10, 24, [16, 18, 16], [0, 1, 0], None, 2, [5, 2, 11]),
]


@pytest.mark.parametrize("B", [(), (1,), (4,), (2, 3)])
@pytest.mark.parametrize("M", [15, 16])
@pytest.mark.parametrize("N", [15, 16])
@pytest.mark.parametrize("C", [16])
@pytest.mark.parametrize("D", [16, 32])
def test_graph_attention(B: tuple[int, ...], M: int, N: int, C: int, D: int):
    q = torch.randn(*B, M, C)
    k = torch.randn(*B, N, C)
    v = torch.randn(*B, N, D)

    mask = torch.rand((M, N)) < 0.5
    edges = torch.nonzero(mask).squeeze()

    y_torch = scaled_dot_product_attention(q, k, v, attn_mask=mask)
    y_graph = GraphAttention.fallback_attention(q, k, v, edges)

    assert torch.allclose(torch.nan_to_num(y_torch), y_graph, atol=1e-6)


@pytest.mark.skipif(
    not xfa_available, reason="legacy xformers.sparse API removed (xformers >= 0.0.30)"
)
@pytest.mark.parametrize("B", [(), (1,), (4,), (2, 3)])
@pytest.mark.parametrize("M", [15, 16])
@pytest.mark.parametrize("N", [15, 16])
@pytest.mark.parametrize("C", [16])
@pytest.mark.parametrize("D", [16, 32])
def test_xfa_attention(B: tuple[int, ...], M: int, N: int, C: int, D: int):
    q = torch.randn(M, C)
    k = torch.randn(*B, N, C)
    v = torch.randn(*B, N, D)

    mask = torch.rand((M, N)) < 0.5
    edges = torch.nonzero(mask).squeeze()

    y_torch = scaled_dot_product_attention(q, k, v, attn_mask=mask)
    y_xfa = GraphAttention.xfa_attention(q, k, v, edges)

    assert torch.allclose(torch.nan_to_num(y_torch), y_xfa, atol=1e-6)

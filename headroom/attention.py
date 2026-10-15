"""Single-head float32 attention benchmark; online softmax is a known technique."""
import math

import torch
import torch.nn.functional as functional


def full_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    scores = (query @ key.T) * (1.0 / math.sqrt(query.shape[1]))
    return torch.softmax(scores, dim=-1) @ value


def library_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return functional.scaled_dot_product_attention(query[None, None], key[None, None], value[None, None])[0, 0]


def tiled_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                    query_tile: int = 64, key_tile: int = 128) -> torch.Tensor:
    """Online softmax over bounded score tiles; never constructs [N, M]."""
    if key.shape[0] != value.shape[0] or query.shape[1] != key.shape[1]:
        raise ValueError("Incompatible attention dimensions")
    if query_tile < 1 or key_tile < 1 or key.shape[0] < 1:
        raise ValueError("Tile sizes and key length must be positive")
    scale = 1.0 / math.sqrt(query.shape[1])
    rows = []
    for q0 in range(0, query.shape[0], query_tile):
        q = query[q0:q0 + query_tile]
        maximum = torch.full((q.shape[0],), -float("inf"), dtype=q.dtype, device=q.device)
        denominator = torch.zeros_like(maximum)
        numerator = torch.zeros((q.shape[0], value.shape[1]), dtype=q.dtype, device=q.device)
        for k0 in range(0, key.shape[0], key_tile):
            k = key[k0:k0 + key_tile]
            v = value[k0:k0 + key_tile]
            scores = (q @ k.T) * scale
            next_maximum = torch.maximum(maximum, scores.max(dim=1).values)
            correction = torch.exp(maximum - next_maximum)
            probabilities = torch.exp(scores - next_maximum[:, None])
            denominator = denominator * correction + probabilities.sum(dim=1)
            numerator = numerator * correction[:, None] + probabilities @ v
            maximum = next_maximum
        rows.append(numerator / denominator[:, None])
    return torch.cat(rows, dim=0)

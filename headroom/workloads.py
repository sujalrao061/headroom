"""Bounded point-cloud workloads with distinct output contracts."""
import torch


def pairwise_distance(query: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Broadcasting materializes [N, M, 3], which is the measured hot spot."""
    delta = query[:, None, :] - reference[None, :, :]
    return torch.sqrt(torch.sum(delta * delta, dim=-1))


def nearest_baseline(query: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return torch.argmin(pairwise_distance(query, reference), dim=1)


def matrix_baseline(query: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return pairwise_distance(query, reference)


def radius_baseline(query: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return torch.sum(pairwise_distance(query, reference) <= 1.0, dim=1)


def cdist_nearest(query: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return torch.argmin(torch.cdist(query, reference), dim=1)


def cdist_matrix(query: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return torch.cdist(query, reference)


def cdist_radius(query: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.cdist(query, reference) <= 1.0, dim=1)


def tiled_nearest(query: torch.Tensor, reference: torch.Tensor, tile: int = 128) -> torch.Tensor:
    """Bound both axes; strict < retains the earliest index on equal distances."""
    rows = []
    for q0 in range(0, query.shape[0], tile):
        q = query[q0:q0 + tile]
        best_distance = torch.full((q.shape[0],), float("inf"), device=q.device, dtype=q.dtype)
        best_index = torch.zeros((q.shape[0],), device=q.device, dtype=torch.long)
        for r0 in range(0, reference.shape[0], tile):
            r = reference[r0:r0 + tile]
            distance = torch.sqrt(torch.sum((q[:, None, :] - r[None, :, :]) ** 2, dim=-1))
            local_distance, local_index = torch.min(distance, dim=1)
            improve = local_distance < best_distance
            best_distance = torch.where(improve, local_distance, best_distance)
            best_index = torch.where(improve, local_index + r0, best_index)
        rows.append(best_index)
    return torch.cat(rows)


def tiled_radius_count(query: torch.Tensor, reference: torch.Tensor, tile: int = 128) -> torch.Tensor:
    """Count matches across bounded tiles without materializing the full matrix."""
    rows = []
    for q0 in range(0, query.shape[0], tile):
        q = query[q0:q0 + tile]
        counts = torch.zeros((q.shape[0],), device=q.device, dtype=torch.long)
        for r0 in range(0, reference.shape[0], tile):
            r = reference[r0:r0 + tile]
            distance = torch.sqrt(torch.sum((q[:, None, :] - r[None, :, :]) ** 2, dim=-1))
            counts += torch.sum(distance <= 1.0, dim=1)
        rows.append(counts)
    return torch.cat(rows)


CASES = {
    "nearest": {
        "name": "Nearest neighbor", "contract": "One reference index per query point",
        "baseline": nearest_baseline, "candidate": tiled_nearest, "strategy": "tile_nearest", "source_line": 8,
        "library": cdist_nearest,
        "code": "def nearest(query, reference):\n    delta = query[:, None, :] - reference[None, :, :]\n    distance = torch.sqrt(torch.sum(delta * delta, dim=-1))\n    return torch.argmin(distance, dim=1)",
    },
    "matrix": {
        "name": "Full distance matrix", "contract": "Every pairwise distance, shape [N, M]",
        "baseline": matrix_baseline, "candidate": tiled_nearest, "strategy": "no_repair", "source_line": 8,
        "library": cdist_matrix,
        "code": "def matrix(query, reference):\n    delta = query[:, None, :] - reference[None, :, :]\n    distance = torch.sqrt(torch.sum(delta * delta, dim=-1))\n    return distance",
    },
    "radius": {
        "name": "Radius count", "contract": "Number of reference points within radius 1.0 for each query, shape [N]",
        "baseline": radius_baseline, "candidate": tiled_radius_count, "strategy": "tile_radius_count", "source_line": 8,
        "library": cdist_radius,
        "code": "def radius_count(query, reference):\n    delta = query[:, None, :] - reference[None, :, :]\n    distance = torch.sqrt(torch.sum(delta * delta, dim=-1))\n    return torch.sum(distance <= 1.0, dim=1)",
    },
}

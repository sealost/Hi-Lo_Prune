"""Architecture-independent hierarchical token selection.

The first stage uses low-level image-patch features to build a diverse
candidate pool. The second stage uses visual-token embeddings to select the
tokens retained by the language model. Candidate tokens not retained by the
second stage form the PA-Fusion set.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class HiLoSelection:
    """Indices produced by hierarchical token selection."""

    keep: torch.Tensor
    fusion: torch.Tensor
    candidates: torch.Tensor


def _greedy_diverse_indices(features: torch.Tensor, count: int) -> torch.Tensor:
    """Select a diverse subset using the greedy diagonal update used by Hi-Lo."""
    if features.ndim != 2:
        raise ValueError(f"features must be rank 2, got shape {tuple(features.shape)}")

    token_count = features.shape[0]
    if token_count == 0:
        raise ValueError("features must contain at least one token")
    if not 1 <= count <= token_count:
        raise ValueError(f"count must be in [1, {token_count}], got {count}")

    normalized = F.normalize(features.float(), p=2, dim=1)
    kernel = normalized @ normalized.T
    kernel_sq = kernel.square()
    residual = torch.diagonal(kernel).clone()
    selected = torch.empty(count, dtype=torch.long, device=features.device)

    for step in range(count):
        index = torch.argmax(residual)
        selected[step] = index
        if step + 1 < count:
            pivot = kernel[index, index]
            if pivot > 1e-10:
                residual.sub_(kernel_sq[:, index] / pivot)
            residual[index] = -torch.inf

    return selected


def hierarchical_token_selection(
    patch_features: torch.Tensor,
    token_features: torch.Tensor,
    *,
    keep_count: int,
    candidate_count: int,
) -> HiLoSelection:
    """Run the coarse-to-fine selection described in Hi-Lo Prune.

    Args:
        patch_features: Low-level patch features with shape [N, D_patch].
        token_features: Projected visual-token features with shape [N, D_model].
        keep_count: Number of tokens retained after pruning.
        candidate_count: Size of the coarse candidate pool. This must be at
            least keep_count and at most N.
    """
    if patch_features.ndim != 2 or token_features.ndim != 2:
        raise ValueError("patch_features and token_features must both be rank 2")
    if patch_features.shape[0] != token_features.shape[0]:
        raise ValueError(
            "patch_features and token_features must describe the same number of tokens"
        )

    token_count = token_features.shape[0]
    if not 1 <= keep_count <= candidate_count <= token_count:
        raise ValueError(
            "expected 1 <= keep_count <= candidate_count <= token_count, "
            f"got {keep_count}, {candidate_count}, {token_count}"
        )

    candidates = _greedy_diverse_indices(patch_features, candidate_count)
    selected_within_candidates = _greedy_diverse_indices(
        token_features[candidates], keep_count
    )
    keep = candidates[selected_within_candidates]

    fusion_mask = torch.ones(candidate_count, dtype=torch.bool, device=candidates.device)
    fusion_mask[selected_within_candidates] = False
    fusion = candidates[fusion_mask]

    return HiLoSelection(
        keep=torch.sort(keep).values,
        fusion=torch.sort(fusion).values,
        candidates=torch.sort(candidates).values,
    )

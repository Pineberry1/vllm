# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


_TOKEN_KEEP = 0
_TOKEN_MERGE = 1
_TOKEN_DROP = 2


@dataclass(frozen=True)
class _Block:
    frame_idx: int
    token_indices: torch.Tensor
    utility: float


def _min_max_norm(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values
    min_value = values.min()
    value_range = values.max() - min_value
    if float(value_range) <= 1e-12:
        return torch.zeros_like(values)
    return (values - min_value) / value_range


def _as_size_tuple(video_size_thw: torch.LongTensor | tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(video_size_thw, torch.Tensor):
        return tuple(map(int, video_size_thw.tolist()))
    return tuple(map(int, video_size_thw))


def _build_blocks(
    *,
    video_embeds: torch.Tensor,
    grid_t: int,
    grid_h: int,
    grid_w: int,
    block_hw: int,
) -> list[_Block]:
    blocks: list[_Block] = []
    for t in range(grid_t):
        frame_offset = t * grid_h * grid_w
        for h0 in range(0, grid_h, block_hw):
            for w0 in range(0, grid_w, block_hw):
                indices = []
                for h in range(h0, min(h0 + block_hw, grid_h)):
                    row_offset = frame_offset + h * grid_w
                    for w in range(w0, min(w0 + block_hw, grid_w)):
                        indices.append(row_offset + w)
                token_indices = torch.tensor(
                    indices, device=video_embeds.device, dtype=torch.long
                )
                block_embeds = video_embeds.index_select(0, token_indices).float()
                mean_embed = block_embeds.mean(dim=0, keepdim=True)
                contrast = torch.linalg.vector_norm(block_embeds - mean_embed, dim=-1)
                uniqueness = 1 - F.cosine_similarity(
                    block_embeds, mean_embed.expand_as(block_embeds), dim=-1
                )
                importance = _min_max_norm(contrast + uniqueness)
                variation = block_embeds.var(dim=0, unbiased=False).mean()
                utility = float(importance.mean() + 0.5 * variation)
                blocks.append(
                    _Block(
                        frame_idx=t,
                        token_indices=token_indices,
                        utility=max(utility, 0.0),
                    )
                )
    return blocks


def _allocate_block_budgets(
    blocks: list[_Block],
    *,
    num_frames: int,
    target_tokens: int,
) -> list[int]:
    budgets = [0 for _ in blocks]
    capacities = [int(block.token_indices.numel()) for block in blocks]

    for frame_idx in range(num_frames):
        frame_block_ids = [
            block_idx
            for block_idx, block in enumerate(blocks)
            if block.frame_idx == frame_idx and capacities[block_idx] > 0
        ]
        best_block_idx = max(
            frame_block_ids,
            key=lambda block_idx: (blocks[block_idx].utility, capacities[block_idx], -block_idx),
        )
        budgets[best_block_idx] = 1

    remaining = target_tokens - num_frames
    while remaining > 0:
        available = [
            block_idx
            for block_idx, capacity in enumerate(capacities)
            if budgets[block_idx] < capacity
        ]
        if not available:
            break

        weights = torch.tensor(
            [blocks[block_idx].utility for block_idx in available],
            dtype=torch.float64,
        )
        if float(weights.sum()) <= 1e-12:
            weights = torch.tensor(
                [capacities[block_idx] - budgets[block_idx] for block_idx in available],
                dtype=torch.float64,
            )

        shares = weights / weights.sum() * remaining
        floors = torch.floor(shares).to(torch.long).tolist()
        progressed = False
        for local_idx, block_idx in enumerate(available):
            grant = min(floors[local_idx], capacities[block_idx] - budgets[block_idx])
            if grant > 0:
                budgets[block_idx] += grant
                remaining -= grant
                progressed = True

        if remaining <= 0:
            break

        remainders = shares - torch.floor(shares)
        order = sorted(
            range(len(available)),
            key=lambda local_idx: (
                float(remainders[local_idx]),
                blocks[available[local_idx]].utility,
                capacities[available[local_idx]] - budgets[available[local_idx]],
                -available[local_idx],
            ),
            reverse=True,
        )
        for local_idx in order:
            block_idx = available[local_idx]
            if budgets[block_idx] >= capacities[block_idx]:
                continue
            budgets[block_idx] += 1
            remaining -= 1
            progressed = True
            if remaining <= 0:
                break

        if not progressed:
            break

    return budgets


def _block_scores(block_embeds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    block_embeds_float = block_embeds.float()
    mean_embed = block_embeds_float.mean(dim=0, keepdim=True)
    contrast = torch.linalg.vector_norm(block_embeds_float - mean_embed, dim=-1)
    uniqueness = 1 - F.cosine_similarity(
        block_embeds_float, mean_embed.expand_as(block_embeds_float), dim=-1
    )
    importance = _min_max_norm(contrast + uniqueness)

    if block_embeds_float.shape[0] == 1:
        local_similarity = torch.ones(
            1, device=block_embeds.device, dtype=block_embeds_float.dtype
        )
    else:
        normalized = F.normalize(block_embeds_float, dim=-1)
        similarity = normalized @ normalized.transpose(0, 1)
        local_similarity = (similarity.sum(dim=-1) - 1) / (
            block_embeds_float.shape[0] - 1
        )
    mergeability = _min_max_norm(local_similarity + (1 - importance))
    return importance, mergeability


def _medoid_local_index(block_embeds: torch.Tensor) -> int:
    block_embeds_float = block_embeds.float()
    if block_embeds_float.shape[0] == 1:
        return 0
    distances = torch.cdist(block_embeds_float, block_embeds_float, p=2).mean(dim=1)
    return int(torch.argmin(distances).item())


def _nearest_output_index(
    *,
    token_index: int,
    output_rep_indices: list[int],
    grid_h: int,
    grid_w: int,
) -> int:
    frame_size = grid_h * grid_w
    token_in_frame = token_index % frame_size
    token_h, token_w = divmod(token_in_frame, grid_w)

    best_output = 0
    best_key: tuple[int, int] | None = None
    for output_idx, rep_index in enumerate(output_rep_indices):
        rep_in_frame = rep_index % frame_size
        rep_h, rep_w = divmod(rep_in_frame, grid_w)
        key = (abs(token_h - rep_h) + abs(token_w - rep_w), abs(token_index - rep_index))
        if best_key is None or key < best_key:
            best_key = key
            best_output = output_idx
    return best_output


def compute_tri_state_folding(
    video_embeds: torch.Tensor,
    video_size_thw: torch.LongTensor | tuple[int, int, int],
    spatial_merge_size: int,
    alpha: float,
    block_t: int = 1,
    block_hw: int = 2,
    edge_prior: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[int], torch.Tensor, torch.Tensor]:
    """Fold video visual tokens with local tri-state KEEP/MERGE/DROP.

    Returns:
        folded_embeds: Folded embeddings with shape (M, D).
        num_tokens_per_frame: Folded token counts per temporal frame.
        token_state: Int8 state over original tokens. Only representative token
            positions are non-DROP so callers can recover folded mRoPE positions.
        source_count: Number of original tokens represented by each folded token.
    """
    del edge_prior  # Reserved for future train-free priors.

    if video_embeds.ndim != 2:
        raise ValueError(f"video_embeds must be 2D, got shape {video_embeds.shape}")
    if block_t != 1:
        raise ValueError("Tri-state merger V0 only supports block_t=1")
    if block_hw <= 0:
        raise ValueError("block_hw must be positive")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")

    grid_t, grid_h_raw, grid_w_raw = _as_size_tuple(video_size_thw)
    grid_h = grid_h_raw // spatial_merge_size
    grid_w = grid_w_raw // spatial_merge_size
    tokens_per_frame = grid_h * grid_w
    num_tokens = int(video_embeds.shape[0])
    expected_tokens = grid_t * tokens_per_frame
    if num_tokens != expected_tokens:
        raise ValueError(
            f"video_embeds has {num_tokens} tokens, expected {expected_tokens} "
            f"for video_size_thw={video_size_thw} and spatial_merge_size={spatial_merge_size}"
        )

    token_state = torch.full(
        (num_tokens,), _TOKEN_DROP, device=video_embeds.device, dtype=torch.int8
    )
    if alpha >= 1.0:
        token_state.zero_()
        source_count = torch.ones(num_tokens, device=video_embeds.device, dtype=torch.long)
        return video_embeds, [tokens_per_frame] * grid_t, token_state, source_count

    target_tokens = min(num_tokens, max(grid_t, math.ceil(alpha * num_tokens)))
    blocks = _build_blocks(
        video_embeds=video_embeds.contiguous(),
        grid_t=grid_t,
        grid_h=grid_h,
        grid_w=grid_w,
        block_hw=block_hw,
    )
    budgets = _allocate_block_budgets(
        blocks, num_frames=grid_t, target_tokens=target_tokens
    )

    output_embeds: list[torch.Tensor] = []
    output_rep_indices: list[int] = []
    output_source_counts: list[int] = []

    for block, budget in zip(blocks, budgets):
        block_indices = block.token_indices
        block_size = int(block_indices.numel())
        if budget <= 0:
            continue

        block_embeds = video_embeds.index_select(0, block_indices).contiguous()
        if budget >= block_size:
            for local_idx in range(block_size):
                rep_index = int(block_indices[local_idx].item())
                token_state[rep_index] = _TOKEN_KEEP
                output_rep_indices.append(rep_index)
                output_embeds.append(block_embeds[local_idx])
                output_source_counts.append(1)
            continue

        importance, mergeability = _block_scores(block_embeds)
        keep_count = min(budget, math.ceil(budget * 0.6))
        keep_order = torch.argsort(importance, descending=True, stable=True)[:keep_count]
        keep_local_indices = set(map(int, keep_order.tolist()))

        for local_idx in sorted(keep_local_indices, key=lambda i: int(block_indices[i])):
            rep_index = int(block_indices[local_idx].item())
            token_state[rep_index] = _TOKEN_KEEP
            output_rep_indices.append(rep_index)
            output_embeds.append(block_embeds[local_idx])
            output_source_counts.append(1)

        merge_count = budget - keep_count
        remaining_local = [
            local_idx for local_idx in range(block_size) if local_idx not in keep_local_indices
        ]
        if merge_count <= 0 or not remaining_local:
            continue

        remaining_tensor = torch.tensor(
            remaining_local, device=video_embeds.device, dtype=torch.long
        )
        merge_order = remaining_tensor[
            torch.argsort(
                mergeability.index_select(0, remaining_tensor),
                descending=True,
                stable=True,
            )
        ].tolist()
        merge_count = min(merge_count, len(merge_order))
        for cluster_idx in range(merge_count):
            cluster_local = merge_order[cluster_idx::merge_count]
            cluster_tensor = torch.tensor(
                cluster_local, device=video_embeds.device, dtype=torch.long
            )
            cluster_embeds = block_embeds.index_select(0, cluster_tensor).contiguous()
            medoid_pos = _medoid_local_index(cluster_embeds)
            rep_local = int(cluster_tensor[medoid_pos].item())
            rep_index = int(block_indices[rep_local].item())
            token_state[rep_index] = _TOKEN_MERGE
            output_rep_indices.append(rep_index)
            output_embeds.append(cluster_embeds.mean(dim=0).to(video_embeds.dtype))
            output_source_counts.append(len(cluster_local))

    # Attach DROP tokens that were not included in a merge cluster to the nearest
    # representative in the same frame so source counts keep covering all inputs.
    outputs_by_frame: list[list[int]] = [[] for _ in range(grid_t)]
    for output_idx, rep_index in enumerate(output_rep_indices):
        outputs_by_frame[rep_index // tokens_per_frame].append(output_idx)

    represented = sum(output_source_counts)
    if represented < num_tokens:
        for token_index in range(num_tokens):
            if token_state[token_index] != _TOKEN_DROP:
                continue
            frame_idx = token_index // tokens_per_frame
            frame_outputs = outputs_by_frame[frame_idx]
            nearest_local = _nearest_output_index(
                token_index=token_index,
                output_rep_indices=[output_rep_indices[i] for i in frame_outputs],
                grid_h=grid_h,
                grid_w=grid_w,
            )
            output_source_counts[frame_outputs[nearest_local]] += 1

    order = sorted(range(len(output_rep_indices)), key=lambda idx: output_rep_indices[idx])
    folded_embeds = torch.stack([output_embeds[idx] for idx in order], dim=0).contiguous()
    sorted_rep_indices = [output_rep_indices[idx] for idx in order]
    source_count = torch.tensor(
        [output_source_counts[idx] for idx in order],
        device=video_embeds.device,
        dtype=torch.long,
    )

    num_tokens_per_frame = [0 for _ in range(grid_t)]
    for rep_index in sorted_rep_indices:
        num_tokens_per_frame[rep_index // tokens_per_frame] += 1

    return folded_embeds, num_tokens_per_frame, token_state, source_count

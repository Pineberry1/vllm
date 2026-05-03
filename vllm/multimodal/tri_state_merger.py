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
    frame_end: int
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
    block_t: int,
    block_hw: int,
    temporal_novelty: torch.Tensor | None = None,
    temporal_residual: torch.Tensor | None = None,
) -> list[_Block]:
    blocks: list[_Block] = []
    for t in range(0, grid_t, block_t):
        t_end = min(t + block_t, grid_t)
        for h0 in range(0, grid_h, block_hw):
            for w0 in range(0, grid_w, block_hw):
                indices = []
                for tt in range(t, t_end):
                    frame_offset = tt * grid_h * grid_w
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
                utility_tensor = importance.mean() + 0.5 * variation
                if temporal_novelty is not None:
                    utility_tensor = utility_tensor + temporal_novelty.index_select(
                        0, token_indices
                    ).mean()
                if temporal_residual is not None:
                    utility_tensor = utility_tensor + 0.5 * temporal_residual.index_select(
                        0, token_indices
                    ).mean()
                utility = float(utility_tensor)
                blocks.append(
                    _Block(
                        frame_idx=t,
                        frame_end=t_end,
                        token_indices=token_indices,
                        utility=max(utility, 0.0),
                    )
                )
    return blocks


def _compute_alpha_init_from_utilities(
    utilities: list[float],
    alpha_min: float = 0.3,
) -> float:
    if not utilities:
        return 1.0
    values = torch.tensor(utilities, dtype=torch.float32)
    if values.numel() == 0:
        return 1.0
    median = float(values.median())
    if median <= 1e-12:
        return alpha_min
    low_utility_frac = float((values < 0.5 * median).float().mean())
    alpha_init = max(alpha_min, 1.0 - low_utility_frac)
    return min(1.0, alpha_init)


def _compute_alpha_init(blocks: list[_Block], alpha_min: float = 0.3) -> float:
    """Content-aware alpha suggestion based on block utility distribution."""
    return _compute_alpha_init_from_utilities(
        [block.utility for block in blocks],
        alpha_min=alpha_min,
    )


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
            if block.frame_idx <= frame_idx < block.frame_end
            and capacities[block_idx] > 0
        ]
        best_block_idx = max(
            frame_block_ids,
            key=lambda block_idx: (blocks[block_idx].utility, capacities[block_idx], -block_idx),
        )
        budgets[best_block_idx] = min(
            capacities[best_block_idx], budgets[best_block_idx] + 1
        )

    remaining = target_tokens - sum(budgets)
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


def _compute_temporal_features(
    video_embeds: torch.Tensor,
    *,
    grid_t: int,
    grid_h: int,
    grid_w: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    embeds = video_embeds.float().view(grid_t, grid_h, grid_w, -1)
    temporal_mean = embeds.mean(dim=0, keepdim=True).expand_as(embeds)

    temporal_novelty = torch.zeros(
        grid_t, grid_h, grid_w, device=video_embeds.device, dtype=torch.float32
    )
    if grid_t > 1:
        prev = embeds[:-1]
        cur = embeds[1:]
        novelty = 1 - F.cosine_similarity(cur, prev, dim=-1)
        temporal_novelty[1:] = novelty
    temporal_novelty = _min_max_norm(temporal_novelty.flatten())

    residual = torch.linalg.vector_norm(embeds - temporal_mean, dim=-1)
    residual = _min_max_norm(residual.flatten())

    background_score = F.cosine_similarity(embeds, temporal_mean, dim=-1)
    background_score = _min_max_norm(background_score.flatten())
    return temporal_novelty, residual, background_score


def _masked_min_max_norm(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values
    masked_min = values.masked_fill(~mask, torch.inf).min(dim=1, keepdim=True).values
    masked_max = values.masked_fill(~mask, -torch.inf).max(dim=1, keepdim=True).values
    value_range = (masked_max - masked_min).clamp_min(1e-12)
    normalized = (values - masked_min) / value_range
    return normalized.masked_fill(~mask, 0)


def _build_block_index_table(
    *,
    grid_t: int,
    grid_h: int,
    grid_w: int,
    block_t: int,
    block_hw: int,
    device: torch.device,
) -> tuple[list[list[int]], list[int], list[int], list[int], torch.Tensor, torch.Tensor]:
    block_indices: list[list[int]] = []
    block_frame_start: list[int] = []
    block_frame_end: list[int] = []

    for t in range(0, grid_t, block_t):
        t_end = min(t + block_t, grid_t)
        for h0 in range(0, grid_h, block_hw):
            for w0 in range(0, grid_w, block_hw):
                indices = []
                for tt in range(t, t_end):
                    frame_offset = tt * grid_h * grid_w
                    for h in range(h0, min(h0 + block_hw, grid_h)):
                        row_offset = frame_offset + h * grid_w
                        for w in range(w0, min(w0 + block_hw, grid_w)):
                            indices.append(row_offset + w)
                block_indices.append(indices)
                block_frame_start.append(t)
                block_frame_end.append(t_end)

    capacities = [len(indices) for indices in block_indices]
    max_capacity = max(capacities, default=1)
    padded_indices_cpu = torch.zeros(
        (len(block_indices), max_capacity), dtype=torch.long
    )
    mask_cpu = torch.zeros((len(block_indices), max_capacity), dtype=torch.bool)
    for block_idx, indices in enumerate(block_indices):
        capacity = len(indices)
        if capacity == 0:
            continue
        padded_indices_cpu[block_idx, :capacity] = torch.tensor(
            indices, dtype=torch.long
        )
        mask_cpu[block_idx, :capacity] = True

    return (
        block_indices,
        block_frame_start,
        block_frame_end,
        capacities,
        padded_indices_cpu.to(device=device, non_blocking=True),
        mask_cpu.to(device=device, non_blocking=True),
    )


def _allocate_block_budgets_from_scores(
    utilities: list[float],
    block_frame_start: list[int],
    block_frame_end: list[int],
    capacities: list[int],
    *,
    num_frames: int,
    target_tokens: int,
) -> list[int]:
    budgets = [0 for _ in capacities]

    for frame_idx in range(num_frames):
        frame_block_ids = [
            block_idx
            for block_idx, capacity in enumerate(capacities)
            if block_frame_start[block_idx] <= frame_idx < block_frame_end[block_idx]
            and capacity > 0
        ]
        best_block_idx = max(
            frame_block_ids,
            key=lambda block_idx: (
                utilities[block_idx],
                capacities[block_idx],
                -block_idx,
            ),
        )
        budgets[best_block_idx] = min(
            capacities[best_block_idx], budgets[best_block_idx] + 1
        )

    remaining = target_tokens - sum(budgets)
    while remaining > 0:
        available = [
            block_idx
            for block_idx, capacity in enumerate(capacities)
            if budgets[block_idx] < capacity
        ]
        if not available:
            break

        weights = [utilities[block_idx] for block_idx in available]
        if sum(weights) <= 1e-12:
            weights = [capacities[block_idx] - budgets[block_idx]
                       for block_idx in available]
        weight_sum = sum(weights)
        if weight_sum <= 1e-12:
            break

        shares = [weight / weight_sum * remaining for weight in weights]
        floors = [math.floor(share) for share in shares]
        progressed = False
        for local_idx, block_idx in enumerate(available):
            grant = min(
                floors[local_idx],
                capacities[block_idx] - budgets[block_idx],
            )
            if grant > 0:
                budgets[block_idx] += grant
                remaining -= grant
                progressed = True

        if remaining <= 0:
            break

        order = sorted(
            range(len(available)),
            key=lambda local_idx: (
                shares[local_idx] - math.floor(shares[local_idx]),
                utilities[available[local_idx]],
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


def _compute_batched_block_scores(
    video_embeds: torch.Tensor,
    padded_indices: torch.Tensor,
    mask: torch.Tensor,
    *,
    temporal_novelty: torch.Tensor | None,
    temporal_residual: torch.Tensor | None,
    background_score: torch.Tensor | None,
) -> tuple[list[float], list[list[float]], list[list[float]]]:
    num_blocks, max_capacity = padded_indices.shape
    block_embeds = video_embeds.index_select(
        0, padded_indices.reshape(-1)
    ).view(num_blocks, max_capacity, -1).float()
    mask_f = mask.unsqueeze(-1).to(block_embeds.dtype)
    counts = mask.sum(dim=1).clamp_min(1).to(block_embeds.dtype)
    mean_embed = (block_embeds * mask_f).sum(dim=1, keepdim=True) / counts.view(
        -1, 1, 1
    )

    contrast = torch.linalg.vector_norm(block_embeds - mean_embed, dim=-1)
    contrast = contrast.masked_fill(~mask, 0)
    uniqueness = 1 - F.cosine_similarity(
        block_embeds, mean_embed.expand_as(block_embeds), dim=-1
    )
    uniqueness = uniqueness.masked_fill(~mask, 0)

    base_importance_features = contrast + uniqueness
    utility_importance = _masked_min_max_norm(base_importance_features, mask)
    variation = (((block_embeds - mean_embed) * mask_f).square().sum(dim=1) /
                 counts.view(-1, 1)).mean(dim=1)
    utility_tensor = (
        (utility_importance * mask).sum(dim=1) / counts
        + 0.5 * variation
    )

    importance_features = base_importance_features
    block_temporal_novelty = None
    block_temporal_residual = None
    block_background_score = None
    if temporal_novelty is not None:
        block_temporal_novelty = temporal_novelty.index_select(
            0, padded_indices.reshape(-1)
        ).view(num_blocks, max_capacity).masked_fill(~mask, 0)
        block_temporal_residual = temporal_residual.index_select(
            0, padded_indices.reshape(-1)
        ).view(num_blocks, max_capacity).masked_fill(~mask, 0)
        block_background_score = background_score.index_select(
            0, padded_indices.reshape(-1)
        ).view(num_blocks, max_capacity).masked_fill(~mask, 0)
        utility_tensor = (
            utility_tensor
            + (block_temporal_novelty * mask).sum(dim=1) / counts
            + 0.5 * (block_temporal_residual * mask).sum(dim=1) / counts
        )
        recency = torch.linspace(
            0.0,
            1.0,
            steps=max_capacity,
            device=video_embeds.device,
            dtype=block_embeds.dtype,
        ).view(1, -1)
        importance_features = (
            importance_features
            + block_temporal_novelty
            + block_temporal_residual
            + recency
        )

    importance = _masked_min_max_norm(importance_features, mask)

    if max_capacity == 1:
        local_similarity = torch.ones_like(importance)
    else:
        normalized = F.normalize(block_embeds, dim=-1)
        similarity = normalized @ normalized.transpose(1, 2)
        eye = torch.eye(
            max_capacity, device=video_embeds.device, dtype=torch.bool
        ).unsqueeze(0)
        valid_pair = mask.unsqueeze(1) & mask.unsqueeze(2) & ~eye
        avg_similarity = similarity.masked_fill(~valid_pair, 0).sum(dim=-1) / (
            counts - 1
        ).clamp_min(1).view(-1, 1)
        max_similarity = similarity.masked_fill(~valid_pair, -1).max(dim=-1).values
        if temporal_novelty is not None:
            local_similarity = max_similarity
        else:
            local_similarity = avg_similarity
        local_similarity = torch.where(
            (counts <= 1).view(-1, 1),
            torch.ones_like(local_similarity),
            local_similarity,
        )
        local_similarity = local_similarity.masked_fill(~mask, 0)

    mergeability_features = local_similarity + (1 - importance)
    if block_temporal_novelty is not None:
        mergeability_features = mergeability_features + (1 - block_temporal_novelty)
    if block_temporal_residual is not None:
        mergeability_features = mergeability_features + (1 - block_temporal_residual)
    if block_background_score is not None:
        mergeability_features = mergeability_features + block_background_score
    mergeability = _masked_min_max_norm(mergeability_features, mask)

    return (
        [max(float(value), 0.0) for value in utility_tensor.detach().cpu().tolist()],
        importance.detach().cpu().tolist(),
        mergeability.detach().cpu().tolist(),
    )


def _position_medoid_local_index_cpu(
    block_indices: list[int],
    cluster_local: list[int],
    *,
    grid_h: int,
    grid_w: int,
) -> int:
    frame_size = grid_h * grid_w
    positions: list[tuple[int, int, int]] = []
    for local_idx in cluster_local:
        token_index = block_indices[local_idx]
        token_t = token_index // frame_size
        token_in_frame = token_index % frame_size
        token_h, token_w = divmod(token_in_frame, grid_w)
        positions.append((token_t, token_h, token_w))

    center_t = sum(pos[0] for pos in positions) / len(positions)
    center_h = sum(pos[1] for pos in positions) / len(positions)
    center_w = sum(pos[2] for pos in positions) / len(positions)
    best_local = cluster_local[0]
    best_key: tuple[float, int] | None = None
    for local_idx, position in zip(cluster_local, positions):
        key = (
            (position[0] - center_t) ** 2
            + (position[1] - center_h) ** 2
            + (position[2] - center_w) ** 2,
            block_indices[local_idx],
        )
        if best_key is None or key < best_key:
            best_key = key
            best_local = local_idx
    return best_local


def _compute_tri_state_folding_fast(
    video_embeds: torch.Tensor,
    *,
    grid_t: int,
    grid_h: int,
    grid_w: int,
    target_tokens: int,
    block_t: int,
    block_hw: int,
) -> tuple[torch.Tensor, list[int], torch.Tensor, torch.Tensor, float]:
    num_tokens = int(video_embeds.shape[0])
    tokens_per_frame = grid_h * grid_w
    (
        block_indices,
        block_frame_start,
        block_frame_end,
        capacities,
        padded_indices,
        mask,
    ) = _build_block_index_table(
        grid_t=grid_t,
        grid_h=grid_h,
        grid_w=grid_w,
        block_t=block_t,
        block_hw=block_hw,
        device=video_embeds.device,
    )

    temporal_novelty = None
    temporal_residual = None
    background_score = None
    if block_t > 1 and grid_t > 1:
        temporal_novelty, temporal_residual, background_score = (
            _compute_temporal_features(
                video_embeds.contiguous(),
                grid_t=grid_t,
                grid_h=grid_h,
                grid_w=grid_w,
            )
        )

    utilities, importance_scores, mergeability_scores = _compute_batched_block_scores(
        video_embeds.contiguous(),
        padded_indices,
        mask,
        temporal_novelty=temporal_novelty,
        temporal_residual=temporal_residual,
        background_score=background_score,
    )
    alpha_init = _compute_alpha_init_from_utilities(utilities)
    budgets = _allocate_block_budgets_from_scores(
        utilities,
        block_frame_start,
        block_frame_end,
        capacities,
        num_frames=grid_t,
        target_tokens=target_tokens,
    )

    token_state_cpu = [_TOKEN_DROP for _ in range(num_tokens)]
    represented = [False for _ in range(num_tokens)]
    output_rep_indices: list[int] = []
    output_source_counts: list[int] = []
    source_token_indices: list[int] = []
    source_output_indices: list[int] = []

    def add_output(
        *,
        rep_index: int,
        state: int,
        source_indices: list[int],
    ) -> None:
        output_idx = len(output_rep_indices)
        output_rep_indices.append(rep_index)
        output_source_counts.append(len(source_indices))
        token_state_cpu[rep_index] = state
        for source_index in source_indices:
            represented[source_index] = True
            source_token_indices.append(source_index)
            source_output_indices.append(output_idx)

    for block_idx, budget in enumerate(budgets):
        indices = block_indices[block_idx]
        block_size = capacities[block_idx]
        if budget <= 0:
            continue

        if budget >= block_size:
            for local_idx, rep_index in enumerate(indices):
                add_output(
                    rep_index=rep_index,
                    state=_TOKEN_KEEP,
                    source_indices=[indices[local_idx]],
                )
            continue

        importance = importance_scores[block_idx]
        mergeability = mergeability_scores[block_idx]
        keep_count = min(budget, math.ceil(budget * 0.6))
        keep_order = sorted(
            range(block_size), key=lambda local_idx: (-importance[local_idx], local_idx)
        )[:keep_count]
        keep_local_indices = set(keep_order)

        for local_idx in sorted(keep_local_indices, key=lambda idx: indices[idx]):
            add_output(
                rep_index=indices[local_idx],
                state=_TOKEN_KEEP,
                source_indices=[indices[local_idx]],
            )

        merge_count = budget - keep_count
        remaining_local = [
            local_idx for local_idx in range(block_size)
            if local_idx not in keep_local_indices
        ]
        if merge_count <= 0 or not remaining_local:
            continue

        merge_order = sorted(
            remaining_local,
            key=lambda local_idx: (-mergeability[local_idx], local_idx),
        )
        merge_count = min(merge_count, len(merge_order))
        for cluster_idx in range(merge_count):
            cluster_local = merge_order[cluster_idx::merge_count]
            if block_t > 1:
                rep_local = _position_medoid_local_index_cpu(
                    indices,
                    cluster_local,
                    grid_h=grid_h,
                    grid_w=grid_w,
                )
            else:
                rep_local = min(
                    cluster_local,
                    key=lambda local_idx: (-importance[local_idx], indices[local_idx]),
                )
            add_output(
                rep_index=indices[rep_local],
                state=_TOKEN_MERGE,
                source_indices=[indices[local_idx] for local_idx in cluster_local],
            )

    if len(output_rep_indices) != target_tokens:
        raise RuntimeError(
            f"tri-state folding selected {len(output_rep_indices)} tokens, "
            f"expected {target_tokens}"
        )

    if not all(represented):
        if block_t == 1:
            outputs_by_frame: list[list[int]] = [[] for _ in range(grid_t)]
            for output_idx, rep_index in enumerate(output_rep_indices):
                outputs_by_frame[rep_index // tokens_per_frame].append(output_idx)

            for token_index, is_represented in enumerate(represented):
                if is_represented:
                    continue
                frame_idx = token_index // tokens_per_frame
                frame_outputs = outputs_by_frame[frame_idx]
                if not frame_outputs:
                    frame_outputs = list(range(len(output_rep_indices)))
                nearest_local = _nearest_output_index(
                    token_index=token_index,
                    output_rep_indices=[
                        output_rep_indices[idx] for idx in frame_outputs
                    ],
                    grid_h=grid_h,
                    grid_w=grid_w,
                )
                output_source_counts[frame_outputs[nearest_local]] += 1
        else:
            outputs_by_temporal_block: dict[int, list[int]] = {}
            for output_idx, rep_index in enumerate(output_rep_indices):
                rep_frame = rep_index // tokens_per_frame
                temporal_block = rep_frame // block_t
                outputs_by_temporal_block.setdefault(temporal_block, []).append(
                    output_idx
                )

            for token_index, is_represented in enumerate(represented):
                if is_represented:
                    continue
                frame_idx = token_index // tokens_per_frame
                temporal_block = frame_idx // block_t
                block_outputs = outputs_by_temporal_block.get(temporal_block, [])
                if not block_outputs:
                    block_outputs = list(range(len(output_rep_indices)))
                nearest_local = _nearest_output_index_temporal(
                    token_index=token_index,
                    output_rep_indices=[
                        output_rep_indices[idx] for idx in block_outputs
                    ],
                    grid_h=grid_h,
                    grid_w=grid_w,
                )
                output_source_counts[block_outputs[nearest_local]] += 1

    source_tokens = torch.tensor(
        source_token_indices, device=video_embeds.device, dtype=torch.long
    )
    source_outputs = torch.tensor(
        source_output_indices, device=video_embeds.device, dtype=torch.long
    )
    output_embeds = torch.zeros(
        (len(output_rep_indices), video_embeds.shape[1]),
        device=video_embeds.device,
        dtype=torch.float32,
    )
    output_embeds.index_add_(
        0,
        source_outputs,
        video_embeds.index_select(0, source_tokens).float(),
    )
    represented_counts = torch.bincount(
        source_outputs, minlength=len(output_rep_indices)
    ).to(device=video_embeds.device, dtype=torch.float32).clamp_min(1)
    output_embeds = (output_embeds / represented_counts.unsqueeze(1)).to(
        video_embeds.dtype
    )

    order = sorted(range(len(output_rep_indices)), key=lambda idx: output_rep_indices[idx])
    order_tensor = torch.tensor(order, device=video_embeds.device, dtype=torch.long)
    folded_embeds = output_embeds.index_select(0, order_tensor).contiguous()
    sorted_rep_indices = [output_rep_indices[idx] for idx in order]
    source_count = torch.tensor(
        [output_source_counts[idx] for idx in order],
        device=video_embeds.device,
        dtype=torch.long,
    )
    token_state = torch.tensor(
        token_state_cpu, device=video_embeds.device, dtype=torch.int8
    )

    num_tokens_per_frame = [0 for _ in range(grid_t)]
    for rep_index in sorted_rep_indices:
        num_tokens_per_frame[rep_index // tokens_per_frame] += 1

    return folded_embeds, num_tokens_per_frame, token_state, source_count, alpha_init

def _block_scores(
    block_embeds: torch.Tensor,
    *,
    temporal_novelty: torch.Tensor | None = None,
    temporal_residual: torch.Tensor | None = None,
    background_score: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    block_embeds_float = block_embeds.float()
    mean_embed = block_embeds_float.mean(dim=0, keepdim=True)
    contrast = torch.linalg.vector_norm(block_embeds_float - mean_embed, dim=-1)
    uniqueness = 1 - F.cosine_similarity(
        block_embeds_float, mean_embed.expand_as(block_embeds_float), dim=-1
    )
    importance_features = contrast + uniqueness
    if temporal_novelty is not None:
        importance_features = importance_features + temporal_novelty
    if temporal_residual is not None:
        importance_features = importance_features + temporal_residual
    if temporal_novelty is not None and temporal_novelty.numel() > 1:
        recency = torch.linspace(
            0.0,
            1.0,
            steps=temporal_novelty.numel(),
            device=block_embeds.device,
            dtype=block_embeds_float.dtype,
        )
        importance_features = importance_features + recency
    importance = _min_max_norm(importance_features)

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
        cross_frame_sim = similarity.masked_fill(
            torch.eye(
                block_embeds_float.shape[0],
                device=block_embeds.device,
                dtype=torch.bool,
            ),
            -1.0,
        ).max(dim=-1).values
        if temporal_novelty is not None:
            local_similarity = cross_frame_sim

    mergeability_features = local_similarity + (1 - importance)
    if temporal_novelty is not None:
        mergeability_features = mergeability_features + (1 - temporal_novelty)
    if temporal_residual is not None:
        mergeability_features = mergeability_features + (1 - temporal_residual)
    if background_score is not None:
        mergeability_features = mergeability_features + background_score
    mergeability = _min_max_norm(mergeability_features)
    return importance, mergeability


def _medoid_local_index(block_embeds: torch.Tensor) -> int:
    block_embeds_float = block_embeds.float()
    if block_embeds_float.shape[0] == 1:
        return 0
    distances = torch.cdist(block_embeds_float, block_embeds_float, p=2).mean(dim=1)
    return int(torch.argmin(distances).item())


def _medoid_local_index_by_position(
    block_indices: torch.Tensor,
    cluster_tensor: torch.Tensor,
    *,
    grid_h: int,
    grid_w: int,
) -> int:
    cluster_indices = block_indices.index_select(0, cluster_tensor)
    frame_size = grid_h * grid_w
    t = torch.div(cluster_indices, frame_size, rounding_mode="floor")
    in_frame = cluster_indices % frame_size
    h = torch.div(in_frame, grid_w, rounding_mode="floor")
    w = in_frame % grid_w
    positions = torch.stack([t, h, w], dim=1).float()
    centroid = positions.mean(dim=0, keepdim=True)
    distances = torch.linalg.vector_norm(positions - centroid, dim=-1)
    return int(cluster_tensor[int(torch.argmin(distances).item())].item())


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


def _nearest_output_index_temporal(
    *,
    token_index: int,
    output_rep_indices: list[int],
    grid_h: int,
    grid_w: int,
) -> int:
    frame_size = grid_h * grid_w
    token_t = token_index // frame_size
    token_in_frame = token_index % frame_size
    token_h, token_w = divmod(token_in_frame, grid_w)

    best_output = 0
    best_key: tuple[int, int, int] | None = None
    for output_idx, rep_index in enumerate(output_rep_indices):
        rep_t = rep_index // frame_size
        rep_in_frame = rep_index % frame_size
        rep_h, rep_w = divmod(rep_in_frame, grid_w)
        key = (
            abs(token_t - rep_t),
            abs(token_h - rep_h) + abs(token_w - rep_w),
            abs(token_index - rep_index),
        )
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
) -> tuple[torch.Tensor, list[int], torch.Tensor, torch.Tensor, float]:
    """Fold video visual tokens with local tri-state KEEP/MERGE/DROP.

    Returns:
        folded_embeds: Folded embeddings with shape (M, D).
        num_tokens_per_frame: Folded token counts per temporal frame.
        token_state: Int8 state over original tokens. Only representative token
            positions are non-DROP so callers can recover folded mRoPE positions.
        source_count: Number of original tokens represented by each folded token.
        alpha_init: Content-aware alpha suggestion for the next window.
    """
    del edge_prior  # Reserved for future train-free priors.

    if video_embeds.ndim != 2:
        raise ValueError(f"video_embeds must be 2D, got shape {video_embeds.shape}")
    if block_t <= 0:
        raise ValueError("block_t must be positive")
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
        return video_embeds, [tokens_per_frame] * grid_t, token_state, source_count, 1.0

    target_tokens = min(num_tokens, max(grid_t, math.ceil(alpha * num_tokens)))
    return _compute_tri_state_folding_fast(
        video_embeds,
        grid_t=grid_t,
        grid_h=grid_h,
        grid_w=grid_w,
        target_tokens=target_tokens,
        block_t=block_t,
        block_hw=block_hw,
    )

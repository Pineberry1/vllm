# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import math
from pathlib import Path

import pytest
import torch


_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "vllm"
    / "multimodal"
    / "tri_state_merger.py"
)
_SPEC = importlib.util.spec_from_file_location("tri_state_merger", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_TRI_STATE_MERGER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_TRI_STATE_MERGER)
compute_tri_state_folding = _TRI_STATE_MERGER.compute_tri_state_folding


def test_alpha_1_is_identity():
    torch.manual_seed(0)
    video_embeds = torch.randn(8, 16)

    folded, tokens_per_frame, token_state, source_count, alpha_init = compute_tri_state_folding(
        video_embeds,
        (2, 4, 4),
        spatial_merge_size=2,
        alpha=1.0,
    )

    assert folded.data_ptr() == video_embeds.data_ptr()
    assert torch.equal(folded, video_embeds)
    assert tokens_per_frame == [4, 4]
    assert torch.equal(token_state, torch.zeros(8, dtype=torch.int8))
    assert torch.equal(source_count, torch.ones(8, dtype=torch.long))
    assert alpha_init == 1.0


@pytest.mark.parametrize("alpha", [0.25, 0.5, 0.75])
def test_token_count_conserved(alpha):
    torch.manual_seed(1)
    video_embeds = torch.randn(12, 8)

    folded, tokens_per_frame, token_state, source_count, alpha_init = compute_tri_state_folding(
        video_embeds,
        (3, 4, 4),
        spatial_merge_size=2,
        alpha=alpha,
    )

    expected_tokens = max(3, math.ceil(alpha * video_embeds.shape[0]))
    assert folded.shape == (expected_tokens, video_embeds.shape[1])
    assert sum(tokens_per_frame) == folded.shape[0]
    assert int(source_count.sum().item()) == video_embeds.shape[0]
    assert int((token_state != 2).sum().item()) == folded.shape[0]


def test_min_one_token_per_frame():
    torch.manual_seed(2)
    video_embeds = torch.randn(32, 8)

    folded, tokens_per_frame, _, source_count, _ = compute_tri_state_folding(
        video_embeds,
        (8, 4, 4),
        spatial_merge_size=2,
        alpha=0.05,
    )

    assert folded.shape[0] == 8
    assert min(tokens_per_frame) >= 1
    assert tokens_per_frame == [1] * 8
    assert int(source_count.sum().item()) == video_embeds.shape[0]


def test_single_image_grid_can_fold_like_one_frame():
    torch.manual_seed(3)
    image_embeds = torch.randn(16, 8)

    folded, tokens_per_frame, token_state, source_count, alpha_init = compute_tri_state_folding(
        image_embeds,
        (1, 8, 8),
        spatial_merge_size=2,
        alpha=0.5,
    )

    assert folded.shape == (8, image_embeds.shape[1])
    assert tokens_per_frame == [8]
    assert int((token_state != 2).sum().item()) == folded.shape[0]
    assert int(source_count.sum().item()) == image_embeds.shape[0]


def test_block_t_1_matches_default_path():
    torch.manual_seed(4)
    video_embeds = torch.randn(64, 16)

    default = compute_tri_state_folding(
        video_embeds,
        (4, 8, 8),
        spatial_merge_size=2,
        alpha=0.5,
        block_hw=2,
    )
    explicit = compute_tri_state_folding(
        video_embeds,
        (4, 8, 8),
        spatial_merge_size=2,
        alpha=0.5,
        block_t=1,
        block_hw=2,
    )

    assert torch.equal(default[0], explicit[0])
    assert default[1] == explicit[1]
    assert torch.equal(default[2], explicit[2])
    assert torch.equal(default[3], explicit[3])


def test_temporal_redundancy_collapses_across_frames():
    torch.manual_seed(5)
    frame = torch.randn(4, 8)
    video_embeds = frame.repeat(4, 1)

    folded, tokens_per_frame, token_state, source_count, alpha_init = compute_tri_state_folding(
        video_embeds,
        (4, 4, 4),
        spatial_merge_size=2,
        alpha=0.5,
        block_t=2,
        block_hw=2,
    )

    assert folded.shape == (8, video_embeds.shape[1])
    assert sum(tokens_per_frame) == folded.shape[0]
    assert int(source_count.sum().item()) == video_embeds.shape[0]
    assert int((token_state != 2).sum().item()) == folded.shape[0]
    assert float(source_count.float().mean().item()) == pytest.approx(2.0)
    assert int(source_count.max().item()) >= 4


def test_temporal_motion_preserves_novel_tokens():
    video_embeds = torch.zeros(16, 8)
    moving_position = 0
    for frame_idx in range(4):
        token_idx = frame_idx * 4 + moving_position
        video_embeds[token_idx] = torch.eye(4, 8)[frame_idx] * 10

    _, _, token_state, source_count, _ = compute_tri_state_folding(
        video_embeds,
        (4, 4, 4),
        spatial_merge_size=2,
        alpha=0.5,
        block_t=2,
        block_hw=2,
    )

    moving_indices = torch.tensor([0, 4, 8, 12])
    assert int((token_state[moving_indices] != 2).sum().item()) >= 3
    assert int(source_count.sum().item()) == video_embeds.shape[0]


def test_temporal_representatives_have_integer_original_positions():
    torch.manual_seed(6)
    video_embeds = torch.randn(24, 8)

    folded, tokens_per_frame, token_state, source_count, alpha_init = compute_tri_state_folding(
        video_embeds,
        (3, 4, 8),
        spatial_merge_size=2,
        alpha=0.5,
        block_t=2,
        block_hw=2,
    )

    rep_indices = (token_state != 2).nonzero(as_tuple=True)[0]
    frame_size = (4 // 2) * (8 // 2)
    assert rep_indices.numel() == folded.shape[0]
    assert rep_indices.dtype == torch.long
    assert int(rep_indices.max().item()) < video_embeds.shape[0]
    assert all(count >= 0 for count in tokens_per_frame)
    assert sum(tokens_per_frame) == folded.shape[0]
    assert int((rep_indices // frame_size).max().item()) < 3
    assert int(source_count.sum().item()) == video_embeds.shape[0]


def test_block_locality():
    video_embeds = torch.zeros(8, 4)
    identical_block = [0, 1, 4, 5]
    orthogonal_block = [2, 3, 6, 7]

    video_embeds[identical_block] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    video_embeds[orthogonal_block] = torch.eye(4)

    folded, tokens_per_frame, token_state, source_count, alpha_init = compute_tri_state_folding(
        video_embeds,
        (1, 4, 8),
        spatial_merge_size=2,
        alpha=0.5,
        block_hw=2,
    )

    identical_reps = int((token_state[identical_block] != 2).sum().item())
    orthogonal_reps = int((token_state[orthogonal_block] != 2).sum().item())

    assert folded.shape[0] == 4
    assert tokens_per_frame == [4]
    assert identical_reps <= 1
    assert orthogonal_reps >= 2
    assert int(source_count.sum().item()) == video_embeds.shape[0]


def test_alpha_init_uniform_blocks():
    # Every block has the same high-contrast pattern, so all utilities are equal
    # and above zero; no block is classified as low utility.
    pattern = torch.eye(4, 8)
    video_embeds = pattern.repeat(4, 1)

    _, _, _, _, alpha_init = compute_tri_state_folding(
        video_embeds,
        (1, 8, 8),
        spatial_merge_size=2,
        alpha=0.5,
        block_hw=2,
    )

    assert alpha_init == pytest.approx(1.0)


def test_alpha_init_sparse_content():
    video_embeds = torch.zeros(48, 8)
    # Four blocks with strong local variation, the rest flat background.
    for block_start in (0, 4, 16, 20):
        video_embeds[block_start:block_start + 4] = torch.eye(4, 8) * 10

    _, _, _, _, alpha_init = compute_tri_state_folding(
        video_embeds,
        (1, 12, 16),
        spatial_merge_size=2,
        alpha=0.5,
        block_hw=2,
    )

    assert 0.3 <= alpha_init <= 1.0
    assert alpha_init == pytest.approx(2.0 / 3.0)


def test_alpha_init_returns_in_range():
    torch.manual_seed(7)
    for _ in range(50):
        video_embeds = torch.randn(24, 8)
        alpha = float(torch.empty(1).uniform_(0.05, 1.0).item())
        _, _, _, _, alpha_init = compute_tri_state_folding(
            video_embeds,
            (3, 4, 8),
            spatial_merge_size=2,
            alpha=alpha,
            block_t=2,
            block_hw=2,
        )
        assert 0.3 <= alpha_init <= 1.0


def test_alpha_one_returns_one():
    torch.manual_seed(8)
    video_embeds = torch.randn(8, 16)

    _, _, _, _, alpha_init = compute_tri_state_folding(
        video_embeds,
        (2, 4, 4),
        spatial_merge_size=2,
        alpha=1.0,
    )

    assert alpha_init == 1.0

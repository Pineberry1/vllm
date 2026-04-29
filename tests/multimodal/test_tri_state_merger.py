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

    folded, tokens_per_frame, token_state, source_count = compute_tri_state_folding(
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


@pytest.mark.parametrize("alpha", [0.25, 0.5, 0.75])
def test_token_count_conserved(alpha):
    torch.manual_seed(1)
    video_embeds = torch.randn(12, 8)

    folded, tokens_per_frame, token_state, source_count = compute_tri_state_folding(
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

    folded, tokens_per_frame, _, source_count = compute_tri_state_folding(
        video_embeds,
        (8, 4, 4),
        spatial_merge_size=2,
        alpha=0.05,
    )

    assert folded.shape[0] == 8
    assert min(tokens_per_frame) >= 1
    assert tokens_per_frame == [1] * 8
    assert int(source_count.sum().item()) == video_embeds.shape[0]


def test_block_locality():
    video_embeds = torch.zeros(8, 4)
    identical_block = [0, 1, 4, 5]
    orthogonal_block = [2, 3, 6, 7]

    video_embeds[identical_block] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    video_embeds[orthogonal_block] = torch.eye(4)

    folded, tokens_per_frame, token_state, source_count = compute_tri_state_folding(
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

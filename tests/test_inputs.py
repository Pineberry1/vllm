# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config import ModelConfig, VllmConfig
from vllm.inputs.preprocess import InputPreprocessor

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("model_id", ["facebook/chameleon-7b"])
@pytest.mark.parametrize("prompt", ["", {"prompt_token_ids": []}])
@pytest.mark.skip(
    reason=(
        "Applying huggingface processor on text inputs results in "
        "significant performance regression for multimodal models. "
        "See https://github.com/vllm-project/vllm/issues/26320"
    )
)
def test_preprocessor_always_mm_code_path(model_id, prompt):
    model_config = ModelConfig(model=model_id)
    vllm_config = VllmConfig(model_config=model_config)
    input_preprocessor = InputPreprocessor(vllm_config)

    # HF processor adds sep token
    tokenizer = input_preprocessor.get_tokenizer()
    sep_token_id = tokenizer.vocab[tokenizer.sep_token]

    processed_inputs = input_preprocessor.preprocess(prompt)
    assert sep_token_id in processed_inputs["prompt_token_ids"]


def test_text_prompt_forwards_multi_modal_uuids(monkeypatch):
    input_preprocessor = object.__new__(InputPreprocessor)
    calls = {}

    def fake_process_multimodal(
        prompt,
        multi_modal_data,
        mm_processor_kwargs,
        *,
        tokenization_kwargs=None,
        mm_uuids=None,
    ):
        calls["prompt"] = prompt
        calls["multi_modal_data"] = multi_modal_data
        calls["mm_processor_kwargs"] = mm_processor_kwargs
        calls["tokenization_kwargs"] = tokenization_kwargs
        calls["mm_uuids"] = mm_uuids
        return {"prompt_token_ids": [1]}

    monkeypatch.setattr(
        input_preprocessor,
        "_process_multimodal",
        fake_process_multimodal,
    )

    video_item = object()
    processed_inputs = input_preprocessor._process_text(
        {
            "prompt": "prefix <|vision_start|><|video_pad|><|vision_end|>",
            "multi_modal_data": {"video": [video_item]},
            "multi_modal_uuids": {"video": ["bava_mem:stream-a:decision-1"]},
            "mm_processor_kwargs": {"visual_token_merger_alpha": 1.0},
        }
    )

    assert calls == {
        "prompt": "prefix <|vision_start|><|video_pad|><|vision_end|>",
        "multi_modal_data": {"video": [video_item]},
        "mm_processor_kwargs": {"visual_token_merger_alpha": 1.0},
        "tokenization_kwargs": None,
        "mm_uuids": {"video": ["bava_mem:stream-a:decision-1"]},
    }
    assert processed_inputs["prompt"] == (
        "prefix <|vision_start|><|video_pad|><|vision_end|>"
    )

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from types import SimpleNamespace

from vllm.engine.protocol import StreamingInput
from vllm.entrypoints.serve.online_prefill.api_router import (
    DEFAULT_SYSTEM_PROMPT,
    OnlinePrefillAppendRequest,
    OnlinePrefillCreateRequest,
    OnlinePrefillFrame,
    OnlinePrefillSessionManager,
)


class FakeEngineClient:
    def __init__(self):
        self.input_processor = SimpleNamespace(input_preprocessor=object())
        self.seen_inputs: list[StreamingInput] = []

    async def generate(self, input_stream, sampling_params, request_id):
        async for item in input_stream:
            self.seen_inputs.append(item)
        if False:
            yield None


def _patch_prepare(manager: OnlinePrefillSessionManager):
    calls = []

    def fake_prepare(
        frames,
        *,
        prefix_text: str = "",
        suffix_text: str = "",
        stream_end: bool = False,
    ):
        calls.append(
            {
                "frames": len(frames),
                "prefix_text": prefix_text,
                "suffix_text": suffix_text,
                "stream_end": stream_end,
            }
        )
        return (
            StreamingInput(
                prompt={
                    "frames": len(frames),
                    "prefix_text": prefix_text,
                    "suffix_text": suffix_text,
                },
                online_prefill_enabled=True,
                stream_end=stream_end,
            ),
            len(frames),
        )

    manager._prepare_append_streaming_input = fake_prepare
    return calls


def test_online_prefill_delays_prefix_until_first_append():
    async def run():
        engine = FakeEngineClient()
        manager = OnlinePrefillSessionManager(engine)
        calls = _patch_prepare(manager)

        session = await manager.create_session(
            OnlinePrefillCreateRequest(request_id="req", prompt="Describe it.")
        )
        await asyncio.sleep(0.05)

        assert engine.seen_inputs == []
        assert session.prompt_token_counts == []

        frame = OnlinePrefillFrame(data="x")
        await manager.append(
            "req", OnlinePrefillAppendRequest(frames=[frame], stream_end=False)
        )
        await asyncio.sleep(0.05)

        assert calls[0]["prefix_text"] == (
            f"<|im_start|>system\n{DEFAULT_SYSTEM_PROMPT}<|im_end|>\n"
            "<|im_start|>user\n"
        )
        assert calls[0]["suffix_text"] == ""
        assert calls[0]["stream_end"] is False

        await manager.append(
            "req", OnlinePrefillAppendRequest(frames=[], stream_end=True)
        )
        await session.generation_task

        assert calls[1]["prefix_text"] == ""
        assert calls[1]["suffix_text"] == (
            "Describe it.<|im_end|>\n<|im_start|>assistant\n"
        )
        assert calls[1]["stream_end"] is True
        assert len(engine.seen_inputs) == 2

    asyncio.run(run())


def test_online_prefill_single_append_coalesces_prefix_and_suffix():
    async def run():
        engine = FakeEngineClient()
        manager = OnlinePrefillSessionManager(engine)
        calls = _patch_prepare(manager)

        session = await manager.create_session(
            OnlinePrefillCreateRequest(request_id="req-single", prompt="Describe it.")
        )
        frame = OnlinePrefillFrame(data="x")
        await manager.append(
            "req-single", OnlinePrefillAppendRequest(frames=[frame], stream_end=True)
        )
        await session.generation_task

        assert calls == [
            {
                "frames": 1,
                "prefix_text": (
                    f"<|im_start|>system\n{DEFAULT_SYSTEM_PROMPT}<|im_end|>\n"
                    "<|im_start|>user\n"
                ),
                "suffix_text": (
                    "Describe it.<|im_end|>\n<|im_start|>assistant\n"
                ),
                "stream_end": True,
            }
        ]
        assert len(engine.seen_inputs) == 1
        assert session.prompt_token_counts == [1]

    asyncio.run(run())

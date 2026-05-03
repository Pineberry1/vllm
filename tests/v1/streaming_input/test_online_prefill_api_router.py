# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import base64
import io
from types import SimpleNamespace

import torch

from vllm.engine.protocol import StreamingInput
from vllm.entrypoints.serve.online_prefill.api_router import (
    DEFAULT_SYSTEM_PROMPT,
    OnlinePrefillAppendRequest,
    OnlinePrefillCreateRequest,
    OnlinePrefillFrame,
    OnlinePrefillSessionManager,
    OnlinePrefillVisualMemory,
    QWEN_VL_VIDEO_PLACEHOLDER,
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
        mm_processor_kwargs=None,
        visual_memory=None,
    ):
        calls.append(
            {
                "frames": len(frames),
                "prefix_text": prefix_text,
                "suffix_text": suffix_text,
                "stream_end": stream_end,
                "mm_processor_kwargs": mm_processor_kwargs or {},
                "visual_memory": visual_memory,
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

        assert DEFAULT_SYSTEM_PROMPT == ""
        assert calls[0]["prefix_text"] == "<|im_start|>user\n"
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
                "prefix_text": "<|im_start|>user\n",
                "suffix_text": (
                    "Describe it.<|im_end|>\n<|im_start|>assistant\n"
                ),
                "stream_end": True,
                "mm_processor_kwargs": {},
                "visual_memory": None,
            }
        ]
        assert len(engine.seen_inputs) == 1
        assert session.prompt_token_counts == [1]

    asyncio.run(run())


def test_online_prefill_keeps_explicit_system_prompt():
    async def run():
        engine = FakeEngineClient()
        manager = OnlinePrefillSessionManager(engine)
        calls = _patch_prepare(manager)

        session = await manager.create_session(
            OnlinePrefillCreateRequest(
                request_id="req-explicit-system",
                prompt="Describe it.",
                system_prompt="You are a careful assistant.",
            )
        )
        frame = OnlinePrefillFrame(data="x")
        await manager.append(
            "req-explicit-system",
            OnlinePrefillAppendRequest(frames=[frame], stream_end=True),
        )
        await session.generation_task

        assert calls[0]["prefix_text"] == (
            "<|im_start|>system\nYou are a careful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
        )

    asyncio.run(run())


def test_online_prefill_forwards_visual_token_merger_kwargs():
    async def run():
        engine = FakeEngineClient()
        manager = OnlinePrefillSessionManager(engine)
        calls = _patch_prepare(manager)

        session = await manager.create_session(
            OnlinePrefillCreateRequest(
                request_id="req-alpha",
                prompt="Describe it.",
                visual_token_merger_alpha=0.5,
                visual_token_merger_block_hw=2,
            )
        )
        frame = OnlinePrefillFrame(data="x")
        await manager.append(
            "req-alpha", OnlinePrefillAppendRequest(frames=[frame], stream_end=True)
        )
        await session.generation_task

        assert calls[0]["mm_processor_kwargs"] == {
            "visual_token_merger_alpha": 0.5,
            "visual_token_merger_block_hw": 2,
        }
        assert calls[0]["visual_memory"] is None

    asyncio.run(run())


def test_online_prefill_visual_memory_prefix_sent_once():
    async def run():
        engine = FakeEngineClient()
        manager = OnlinePrefillSessionManager(engine)
        calls = _patch_prepare(manager)
        visual_memory = OnlinePrefillVisualMemory(
            data="placeholder",
            memory_id="bava_mem:stream-a:decision-1",
            text_prefix="Question first.\n",
            num_frames=4,
            tokens_per_frame=2,
        )

        session = await manager.create_session(
            OnlinePrefillCreateRequest(
                request_id="req-memory",
                prompt="Describe it.",
                visual_memory=visual_memory,
            )
        )
        frame = OnlinePrefillFrame(data="x")
        await manager.append(
            "req-memory", OnlinePrefillAppendRequest(frames=[frame], stream_end=False)
        )
        await asyncio.sleep(0.05)
        await manager.append(
            "req-memory", OnlinePrefillAppendRequest(frames=[], stream_end=True)
        )
        await session.generation_task

        assert calls[0]["prefix_text"] == (
            f"<|im_start|>user\nQuestion first.\n{QWEN_VL_VIDEO_PLACEHOLDER}"
        )
        assert calls[0]["visual_memory"] is visual_memory
        assert calls[0]["frames"] == 1
        assert calls[1]["prefix_text"] == ""
        assert calls[1]["frames"] == 0
        assert calls[1]["visual_memory"] is None
        assert calls[1]["suffix_text"] == (
            "Describe it.<|im_end|>\n<|im_start|>assistant\n"
        )
        assert len(calls) == 2

    asyncio.run(run())


def test_online_prefill_visual_memory_only_stream_end_is_single_input():
    async def run():
        engine = FakeEngineClient()
        manager = OnlinePrefillSessionManager(engine)
        calls = _patch_prepare(manager)
        visual_memory = OnlinePrefillVisualMemory(
            data="placeholder",
            memory_id="bava_mem:stream-a:decision-1",
            text_prefix="Question first.\n",
            num_frames=4,
            tokens_per_frame=2,
        )

        session = await manager.create_session(
            OnlinePrefillCreateRequest(
                request_id="req-memory-only",
                prompt="Describe it.",
                visual_memory=visual_memory,
                max_tokens=1,
            )
        )
        await manager.append(
            "req-memory-only",
            OnlinePrefillAppendRequest(frames=[], stream_end=True),
        )
        await session.generation_task

        assert len(calls) == 1
        assert calls[0]["prefix_text"] == (
            f"<|im_start|>user\nQuestion first.\n{QWEN_VL_VIDEO_PLACEHOLDER}"
        )
        assert calls[0]["suffix_text"] == (
            "Describe it.<|im_end|>\n<|im_start|>assistant\n"
        )
        assert calls[0]["stream_end"] is True
        assert calls[0]["frames"] == 0
        assert calls[0]["visual_memory"] is visual_memory

    asyncio.run(run())


def test_online_prefill_exports_visual_memory_asynchronously_and_warms_cache():
    async def run():
        engine = FakeEngineClient()
        manager = OnlinePrefillSessionManager(engine)
        calls = _patch_prepare(manager)
        visual_memory = OnlinePrefillVisualMemory(
            data="placeholder",
            memory_id="bava_mem:stream-a:decision-1",
            text_prefix="Question first.\n",
            num_frames=4,
            tokens_per_frame=2,
        )
        exported = OnlinePrefillVisualMemory(
            data="exported",
            memory_id="bava_mem:exported",
            text_prefix="Warm me.\n",
            num_frames=4,
            tokens_per_frame=2,
        )

        session = await manager.create_session(
            OnlinePrefillCreateRequest(
                request_id="req-export",
                prompt="Describe it.",
                visual_memory=visual_memory,
                export_visual_memory=True,
                warm_visual_memory_prefix_cache=True,
            )
        )
        seen_warmups = []

        async def fake_export_session_visual_memory(_session):
            await asyncio.sleep(0)
            return exported

        async def fake_warm(session_arg, memory_arg):
            seen_warmups.append((session_arg.request_id, memory_arg.memory_id))

        manager._export_session_visual_memory = fake_export_session_visual_memory
        manager._warm_visual_memory_prefix_cache = fake_warm

        frame = OnlinePrefillFrame(data="x")
        await manager.append(
            "req-export", OnlinePrefillAppendRequest(frames=[frame], stream_end=True)
        )
        await session.generation_task
        assert session.visual_memory_export_task is not None
        await asyncio.wait_for(session.visual_memory_export_task, timeout=1.0)

        assert session.exported_visual_memory is exported
        assert session.visual_memory_export_pending is False
        assert seen_warmups == [("req-export", "bava_mem:exported")]
        assert calls[0]["frames"] == 1

    asyncio.run(run())


def test_online_prefill_decodes_visual_memory_as_video_embeds():
    tensor = torch.arange(4 * 3, dtype=torch.float32).view(4, 3)
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    payload = base64.b64encode(buffer.getvalue()).decode("utf-8")

    memory = OnlinePrefillVisualMemory(
        data=payload,
        num_frames=2,
        tokens_per_frame=2,
        timestamps=[0.0, 1.0],
    )
    decoded = OnlinePrefillSessionManager._decode_visual_memory(memory)

    assert torch.equal(decoded["video_embeds"], tensor)
    assert torch.equal(
        decoded["video_grid_thw"],
        torch.tensor([[2, 2, 4]], dtype=torch.long),
    )
    assert decoded["timestamps"] == [[0.0, 1.0]]


def test_online_prefill_visual_memory_preprocess_uses_stable_uuid():
    tensor = torch.arange(4 * 3, dtype=torch.float32).view(4, 3)
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    payload = base64.b64encode(buffer.getvalue()).decode("utf-8")
    memory = OnlinePrefillVisualMemory(
        data=payload,
        memory_id="bava_mem:stream-a:decision-1",
        num_frames=2,
        tokens_per_frame=2,
    )

    engine = FakeEngineClient()
    manager = OnlinePrefillSessionManager(engine)
    manager.input_preprocessor = SimpleNamespace(preprocess=lambda payload: payload)

    prompt, frame_token_sizes = manager._preprocess_prompt(
        [],
        prefix_text=f"Stable prefix {QWEN_VL_VIDEO_PLACEHOLDER}",
        visual_memory=memory,
    )

    assert frame_token_sizes is None
    assert prompt["multi_modal_uuids"] == {
        "video": ["bava_mem:stream-a:decision-1"]
    }
    assert prompt["prompt"] == f"Stable prefix {QWEN_VL_VIDEO_PLACEHOLDER}"
    assert set(prompt["multi_modal_data"]["video"].keys()) == {
        "video_embeds",
        "video_grid_thw",
        "timestamps",
    }


def test_online_prefill_visual_merger_defaults_to_image_stream(monkeypatch):
    engine = FakeEngineClient()
    manager = OnlinePrefillSessionManager(engine)
    calls = []

    manager._decode_frame = lambda frame: object()
    manager._prompt_token_count = lambda prompt: 7

    def fake_preprocess_prompt(
        images,
        *,
        prefix_text: str = "",
        suffix_text: str = "",
        mm_processor_kwargs=None,
        visual_memory=None,
    ):
        calls.append(("image", len(images), mm_processor_kwargs))
        return {"path": "image"}, [7]

    def fake_preprocess_video_prompt(*args, **kwargs):
        raise AssertionError("visual merger should keep image stream by default")

    manager._preprocess_prompt = fake_preprocess_prompt
    manager._preprocess_video_prompt = fake_preprocess_video_prompt

    streaming_input, prompt_token_count = manager._prepare_append_streaming_input(
        [OnlinePrefillFrame(data="x")],
        stream_end=True,
        mm_processor_kwargs={"visual_token_merger_alpha": 0.5},
    )

    assert calls == [("image", 1, {"visual_token_merger_alpha": 0.5})]
    assert streaming_input.prompt == {"path": "image"}
    assert streaming_input.frame_token_sizes == [7]
    assert prompt_token_count == 7


def test_online_prefill_visual_merger_can_use_video_stream(monkeypatch):
    monkeypatch.setenv("VLLM_ONLINE_PREFILL_VISUAL_TOKEN_MERGER_INPUT", "video")
    engine = FakeEngineClient()
    manager = OnlinePrefillSessionManager(engine)
    calls = []

    manager._decode_frame = lambda frame: object()
    manager._prompt_token_count = lambda prompt: 5

    def fake_preprocess_video_prompt(
        images,
        *,
        prefix_text: str = "",
        suffix_text: str = "",
        mm_processor_kwargs,
    ):
        calls.append(("video", len(images), mm_processor_kwargs))
        return {"path": "video"}, [5]

    manager._preprocess_video_prompt = fake_preprocess_video_prompt

    streaming_input, prompt_token_count = manager._prepare_append_streaming_input(
        [OnlinePrefillFrame(data="x")],
        stream_end=True,
        mm_processor_kwargs={"visual_token_merger_alpha": 0.5},
    )

    assert calls == [("video", 1, {"visual_token_merger_alpha": 0.5})]
    assert streaming_input.prompt == {"path": "video"}
    assert streaming_input.frame_token_sizes == [5]
    assert prompt_token_count == 5

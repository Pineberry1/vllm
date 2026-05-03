# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
import base64
import io
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from fastapi import APIRouter, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from PIL import Image

from vllm.engine.protocol import EngineClient, StreamingInput
from vllm.inputs import ProcessorInputs
from vllm.inputs.parse import split_enc_dec_inputs
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams

logger = init_logger(__name__)

router = APIRouter()

QWEN_VL_IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
QWEN_VL_VIDEO_PLACEHOLDER = "<|vision_start|><|video_pad|><|vision_end|>"
DEFAULT_SYSTEM_PROMPT = ""
_STREAM_DONE = object()


def _read_merge_window_ms() -> float:
    # Short coalescing window applied after the first queued append arrives.
    # Lets sequential client POSTs (each carrying one frame) merge into a
    # single StreamingInput so they share one vision-encoder invocation and
    # one engine update, instead of paying per-append preprocess/encoder/
    # scheduler overhead. Set to 0 to disable.
    raw = os.environ.get("VLLM_ONLINE_PREFILL_MERGE_WINDOW_MS", "30")
    try:
        return max(0.0, float(raw)) / 1000.0
    except ValueError:
        return 0.030


_MERGE_WINDOW_SEC = _read_merge_window_ms()


class OnlinePrefillFrame(BaseModel):
    data: str
    mime_type: str = "image/jpeg"
    file_name: str | None = None


class OnlinePrefillVisualMemory(BaseModel):
    data: str
    mime_type: str = "application/x-torch"
    memory_id: str | None = None
    text_prefix: str = ""
    num_frames: int = 8
    tokens_per_frame: int | None = None
    timestamps: list[float] | None = None


class OnlinePrefillCreateRequest(BaseModel):
    request_id: str
    prompt: str
    model: str | None = None
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    visual_token_merger_alpha: float | None = None
    visual_token_merger_block_t: int | None = None
    visual_token_merger_block_hw: int | None = None
    visual_memory: OnlinePrefillVisualMemory | None = None
    export_visual_memory: bool = False
    export_visual_memory_num_frames: int = 8
    export_visual_memory_tokens_per_frame: int = 32
    export_visual_memory_id: str | None = None
    export_visual_memory_text_prefix: str = ""
    warm_visual_memory_prefix_cache: bool = False


class OnlinePrefillAppendRequest(BaseModel):
    frames: list[OnlinePrefillFrame] = Field(default_factory=list)
    stream_end: bool = False
    visual_token_merger_alpha: float | None = None
    visual_token_merger_block_t: int | None = None
    visual_token_merger_block_hw: int | None = None


class OnlinePrefillSessionResponse(BaseModel):
    request_id: str
    status: str
    created_at: float
    updated_at: float
    appended_chunks: int
    appended_frames: int
    stream_end_received: bool
    decode_started: bool
    finished: bool
    output_text: str | None = None
    error: str | None = None
    visual_memory: OnlinePrefillVisualMemory | None = None
    visual_memory_error: str | None = None
    visual_memory_export_pending: bool = False
    visual_memory_prefix_cache_warmup_error: str | None = None


@dataclass
class _OnlinePrefillSession:
    request_id: str
    prompt: str
    sampling_params: SamplingParams
    created_at: float
    updated_at: float
    queue: asyncio.Queue[StreamingInput | _QueuedAppend | object]
    generation_task: asyncio.Task[None]
    prefix_text: str
    suffix_text: str
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    status: str = "created"
    appended_chunks: int = 0
    appended_frames: int = 0
    stream_end_received: bool = False
    decode_started: bool = False
    finished: bool = False
    output_text: str | None = None
    error: str | None = None
    prompt_token_counts: list[int] = field(default_factory=list)
    input_started: bool = False
    mm_processor_kwargs: dict[str, Any] = field(default_factory=dict)
    visual_memory: OnlinePrefillVisualMemory | None = None
    export_visual_memory: bool = False
    export_visual_memory_num_frames: int = 8
    export_visual_memory_tokens_per_frame: int = 32
    export_visual_memory_id: str | None = None
    export_visual_memory_text_prefix: str = ""
    warm_visual_memory_prefix_cache: bool = False
    visual_memory_export_hashes: list[str] = field(default_factory=list)
    visual_memory_export_frame_counts: list[int] = field(default_factory=list)
    exported_visual_memory: OnlinePrefillVisualMemory | None = None
    visual_memory_error: str | None = None
    visual_memory_export_pending: bool = False
    visual_memory_prefix_cache_warmup_error: str | None = None
    visual_memory_export_task: asyncio.Task[None] | None = None


@dataclass
class _QueuedAppend:
    frames: list[OnlinePrefillFrame] = field(default_factory=list)
    stream_end: bool = False
    mm_processor_kwargs: dict[str, Any] = field(default_factory=dict)


class OnlinePrefillSessionManager:
    def __init__(self, engine_client: EngineClient) -> None:
        self.engine_client = engine_client
        self.input_preprocessor = engine_client.input_processor.input_preprocessor
        self._sessions: dict[str, _OnlinePrefillSession] = {}
        self._lock = asyncio.Lock()
        self._supported_tasks: tuple[str, ...] | None = None

    async def _get_supported_tasks(self) -> tuple[str, ...]:
        if self._supported_tasks is None:
            self._supported_tasks = await self.engine_client.get_supported_tasks()
        return self._supported_tasks

    async def create_session(
        self,
        request: OnlinePrefillCreateRequest,
    ) -> _OnlinePrefillSession:
        async with self._lock:
            if request.request_id in self._sessions:
                raise HTTPException(status_code=409, detail="request_id already exists")
            if request.export_visual_memory:
                if request.export_visual_memory_num_frames <= 0:
                    raise HTTPException(
                        status_code=400,
                        detail="export_visual_memory_num_frames must be positive",
                    )
                if request.export_visual_memory_tokens_per_frame <= 0:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "export_visual_memory_tokens_per_frame must be positive"
                        ),
                    )

            sampling_params = SamplingParams(
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
            )

            prefix_text = self._build_prefix_text(
                request.system_prompt,
                visual_memory=request.visual_memory,
            )
            suffix_text = self._build_suffix_text(request.prompt)

            queue: asyncio.Queue[StreamingInput | _QueuedAppend | object] = asyncio.Queue()
            now = time.time()
            session = _OnlinePrefillSession(
                request_id=request.request_id,
                prompt=request.prompt,
                sampling_params=sampling_params,
                created_at=now,
                updated_at=now,
                queue=queue,
                generation_task=asyncio.create_task(asyncio.sleep(0)),
                prefix_text=prefix_text,
                suffix_text=suffix_text,
                system_prompt=request.system_prompt,
                mm_processor_kwargs=self._mm_kwargs_from_request(request),
                visual_memory=request.visual_memory,
                export_visual_memory=request.export_visual_memory,
                export_visual_memory_num_frames=(
                    request.export_visual_memory_num_frames
                ),
                export_visual_memory_tokens_per_frame=(
                    request.export_visual_memory_tokens_per_frame
                ),
                export_visual_memory_id=request.export_visual_memory_id,
                export_visual_memory_text_prefix=(
                    request.export_visual_memory_text_prefix
                ),
                warm_visual_memory_prefix_cache=(
                    request.warm_visual_memory_prefix_cache
                ),
            )
            session.generation_task = asyncio.create_task(self._run_session(session))
            self._sessions[request.request_id] = session
            return session

    async def append(
        self,
        request_id: str,
        append_request: OnlinePrefillAppendRequest,
    ) -> _OnlinePrefillSession:
        session = await self.get_session(request_id)
        if session.finished:
            raise HTTPException(status_code=409, detail="session already finished")
        if session.stream_end_received:
            raise HTTPException(status_code=409, detail="stream_end already received")
        if session.error is not None:
            raise HTTPException(status_code=500, detail=session.error)

        if append_request.frames or append_request.stream_end:
            await session.queue.put(
                _QueuedAppend(
                    frames=list(append_request.frames),
                    stream_end=append_request.stream_end,
                    mm_processor_kwargs=self._merge_mm_kwargs(
                        session.mm_processor_kwargs,
                        self._mm_kwargs_from_request(append_request),
                    ),
                )
            )

        if append_request.frames:
            session.appended_chunks += 1
            session.appended_frames += len(append_request.frames)
            if not append_request.stream_end:
                session.status = "streaming"

        if append_request.stream_end:
            session.stream_end_received = True
            session.status = "waiting_for_decode"

        session.updated_at = time.time()
        return session

    async def get_session(self, request_id: str) -> _OnlinePrefillSession:
        async with self._lock:
            session = self._sessions.get(request_id)
        if session is None:
            raise HTTPException(status_code=404, detail="session not found")
        return session

    async def abort(self, request_id: str) -> None:
        session = await self.get_session(request_id)
        session.finished = True
        session.status = "aborted"
        session.updated_at = time.time()
        try:
            await self.engine_client.abort(request_id)
        finally:
            await session.queue.put(_STREAM_DONE)

    async def _run_session(self, session: _OnlinePrefillSession) -> None:
        async def input_stream():
            while True:
                item = await session.queue.get()
                if item is _STREAM_DONE:
                    return
                if isinstance(item, StreamingInput):
                    yield item
                    continue

                assert isinstance(item, _QueuedAppend)
                queued_frames = list(item.frames)
                stream_end = item.stream_end
                mm_processor_kwargs = dict(item.mm_processor_kwargs)

                # Drain anything already queued synchronously first.
                drained_terminal: Any = None
                while not stream_end:
                    try:
                        next_item = session.queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break

                    if next_item is _STREAM_DONE:
                        drained_terminal = _STREAM_DONE
                        break
                    if isinstance(next_item, StreamingInput):
                        drained_terminal = next_item
                        break

                    assert isinstance(next_item, _QueuedAppend)
                    queued_frames.extend(next_item.frames)
                    stream_end = stream_end or next_item.stream_end
                    if next_item.mm_processor_kwargs:
                        mm_processor_kwargs = self._merge_mm_kwargs(
                            mm_processor_kwargs,
                            next_item.mm_processor_kwargs,
                        )

                # Merge window: wait briefly for sequential client POSTs to
                # arrive so their frames coalesce into one StreamingInput.
                # Client-sequential posting (POST N waits for POST N-1's HTTP
                # response before sending POST N+1) would otherwise deliver
                # items one at a time and defeat the drain above, causing a
                # separate prefill/encoder pass per frame.
                while (
                    drained_terminal is None
                    and not stream_end
                    and _MERGE_WINDOW_SEC > 0
                ):
                    try:
                        next_item = await asyncio.wait_for(
                            session.queue.get(), timeout=_MERGE_WINDOW_SEC
                        )
                    except asyncio.TimeoutError:
                        break

                    if next_item is _STREAM_DONE:
                        drained_terminal = _STREAM_DONE
                        break
                    if isinstance(next_item, StreamingInput):
                        drained_terminal = next_item
                        break

                    assert isinstance(next_item, _QueuedAppend)
                    queued_frames.extend(next_item.frames)
                    stream_end = stream_end or next_item.stream_end
                    if next_item.mm_processor_kwargs:
                        mm_processor_kwargs = self._merge_mm_kwargs(
                            mm_processor_kwargs,
                            next_item.mm_processor_kwargs,
                        )

                prefix_text = session.prefix_text if not session.input_started else ""
                suffix_text = session.suffix_text if stream_end else ""
                visual_memory = (
                    session.visual_memory if not session.input_started else None
                )

                if visual_memory is not None and not queued_frames and stream_end:
                    (
                        streaming_input,
                        prompt_token_count,
                    ) = await asyncio.to_thread(
                        self._prepare_append_streaming_input,
                        [],
                        prefix_text=prefix_text,
                        suffix_text=suffix_text,
                        stream_end=True,
                        mm_processor_kwargs=mm_processor_kwargs,
                        visual_memory=visual_memory,
                    )
                    session.prompt_token_counts.append(prompt_token_count)
                    session.input_started = True
                    yield streaming_input
                    prefix_text = ""
                    suffix_text = ""
                    visual_memory = None

                if queued_frames or prefix_text or suffix_text or visual_memory:
                    (
                        streaming_input,
                        prompt_token_count,
                    ) = await asyncio.to_thread(
                        self._prepare_append_streaming_input,
                        queued_frames,
                        prefix_text=prefix_text,
                        suffix_text=suffix_text,
                        stream_end=stream_end,
                        mm_processor_kwargs=mm_processor_kwargs,
                        visual_memory=visual_memory,
                    )
                    session.prompt_token_counts.append(prompt_token_count)
                    if session.export_visual_memory and queued_frames:
                        export_hashes, export_frame_counts = (
                            self._export_metadata_from_prompt(
                                streaming_input.prompt,
                                frame_count=len(queued_frames),
                                mm_processor_kwargs=mm_processor_kwargs,
                            )
                        )
                        session.visual_memory_export_hashes.extend(export_hashes)
                        session.visual_memory_export_frame_counts.extend(
                            export_frame_counts
                        )
                    session.input_started = True
                    yield streaming_input
                if stream_end:
                    return
                if drained_terminal is _STREAM_DONE:
                    return
                if isinstance(drained_terminal, StreamingInput):
                    yield drained_terminal

        try:
            async for output in self.engine_client.generate(
                input_stream(),
                session.sampling_params,
                session.request_id,
            ):
                self._update_session_from_output(session, output)
            if session.status != "aborted":
                session.status = "finished"
                session.finished = True
                session.updated_at = time.time()
                if session.export_visual_memory and session.error is None:
                    self._schedule_visual_memory_export(session)
        except Exception as exc:
            logger.exception(
                "Online prefill session %s failed", session.request_id, exc_info=exc
            )
            session.error = str(exc)
            session.status = "error"
            session.finished = True
            session.updated_at = time.time()

    def _update_session_from_output(
        self,
        session: _OnlinePrefillSession,
        output: RequestOutput,
    ) -> None:
        session.updated_at = time.time()
        if session.stream_end_received:
            session.decode_started = True
            session.status = "decoding"
        elif session.status != "waiting_for_decode":
            session.status = "streaming"
        if output.outputs:
            first_output = output.outputs[0]
            output_text = first_output.text or ""
            if output_text:
                if session.output_text and not output_text.startswith(session.output_text):
                    session.output_text += output_text
                else:
                    session.output_text = output_text
        if output.outputs and logger.isEnabledFor(10):  # logging.DEBUG
            logger.debug(
                "online_prefill api_output request_id=%s output_finished=%s text_preview=%r",
                session.request_id,
                output.finished,
                ((output.outputs[0].text or "")[:80] if output.outputs else ""),
            )
        if output.finished:
            logger.info(
                "online_prefill api_finished request_id=%s output_text_present=%s",
                session.request_id,
                bool(session.output_text),
            )
            session.finished = True
            session.status = "finished"

    def _schedule_visual_memory_export(
        self,
        session: _OnlinePrefillSession,
    ) -> None:
        if session.visual_memory_export_task is not None:
            return
        session.visual_memory_export_pending = True

        async def runner() -> None:
            try:
                exported = await self._export_session_visual_memory(session)
                if exported is not None and session.warm_visual_memory_prefix_cache:
                    try:
                        await self._warm_visual_memory_prefix_cache(session, exported)
                    except Exception as exc:
                        logger.exception(
                            "Online prefill session %s visual memory warmup failed",
                            session.request_id,
                            exc_info=exc,
                        )
                        session.visual_memory_prefix_cache_warmup_error = str(exc)
                session.exported_visual_memory = exported
            except Exception as exc:
                logger.exception(
                    "Online prefill session %s visual memory export failed",
                    session.request_id,
                    exc_info=exc,
                )
                session.visual_memory_error = str(exc)
            finally:
                session.visual_memory_export_pending = False
                session.updated_at = time.time()

        session.visual_memory_export_task = asyncio.create_task(runner())

    @staticmethod
    def _build_prefix_text(
        system_prompt: str,
        *,
        visual_memory: OnlinePrefillVisualMemory | None = None,
    ) -> str:
        visual_memory_text = ""
        if visual_memory is not None:
            visual_memory_text = (
                f"{visual_memory.text_prefix}{QWEN_VL_VIDEO_PLACEHOLDER}"
            )
        if system_prompt:
            return (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{visual_memory_text}"
            )
        return f"<|im_start|>user\n{visual_memory_text}"

    @staticmethod
    def _build_suffix_text(prompt: str) -> str:
        return f"{prompt}<|im_end|>\n<|im_start|>assistant\n"

    def _preprocess_text_prompt(self, prompt: str) -> ProcessorInputs:
        return self.input_preprocessor.preprocess({"prompt": prompt})

    async def _warm_visual_memory_prefix_cache(
        self,
        session: _OnlinePrefillSession,
        visual_memory: OnlinePrefillVisualMemory,
    ) -> None:
        prefix_text = self._build_prefix_text(
            session.system_prompt,
            visual_memory=visual_memory,
        )
        (
            streaming_input,
            _,
        ) = await asyncio.to_thread(
            self._prepare_append_streaming_input,
            [],
            prefix_text=prefix_text,
            suffix_text="",
            stream_end=True,
            mm_processor_kwargs=session.mm_processor_kwargs,
            visual_memory=visual_memory,
        )

        async def input_stream():
            yield streaming_input

        warmup_request_id = f"{session.request_id}-visual-memory-warmup"
        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=1,
        )
        async for _ in self.engine_client.generate(
            input_stream(),
            sampling_params,
            warmup_request_id,
        ):
            pass

    @staticmethod
    def _mm_kwargs_from_request(request: Any) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        alpha = getattr(request, "visual_token_merger_alpha", None)
        if alpha is None:
            raw_alpha = os.environ.get("VLLM_ONLINE_PREFILL_VISUAL_TOKEN_MERGER_ALPHA")
            if raw_alpha not in (None, ""):
                try:
                    alpha = float(raw_alpha)
                except ValueError:
                    alpha = None
        if alpha is not None:
            kwargs["visual_token_merger_alpha"] = float(alpha)

        block_t = getattr(request, "visual_token_merger_block_t", None)
        if block_t is None:
            raw_block_t = os.environ.get("VLLM_ONLINE_PREFILL_VISUAL_TOKEN_MERGER_BLOCK_T")
            if raw_block_t not in (None, ""):
                try:
                    block_t = int(raw_block_t)
                except ValueError:
                    block_t = None
        if block_t is not None:
            kwargs["visual_token_merger_block_t"] = int(block_t)

        block_hw = getattr(request, "visual_token_merger_block_hw", None)
        if block_hw is None:
            raw_block_hw = os.environ.get("VLLM_ONLINE_PREFILL_VISUAL_TOKEN_MERGER_BLOCK_HW")
            if raw_block_hw not in (None, ""):
                try:
                    block_hw = int(raw_block_hw)
                except ValueError:
                    block_hw = None
        if block_hw is not None:
            kwargs["visual_token_merger_block_hw"] = int(block_hw)
        return kwargs

    @staticmethod
    def _merge_mm_kwargs(
        base: dict[str, Any],
        override: dict[str, Any],
    ) -> dict[str, Any]:
        merged = dict(base)
        merged.update(override)
        return merged

    @staticmethod
    def _visual_token_merger_enabled(mm_processor_kwargs: dict[str, Any]) -> bool:
        alpha = mm_processor_kwargs.get("visual_token_merger_alpha")
        try:
            return alpha is not None and float(alpha) < 0.999
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _visual_token_merger_uses_video_stream() -> bool:
        raw = os.environ.get("VLLM_ONLINE_PREFILL_VISUAL_TOKEN_MERGER_INPUT", "image")
        return raw.strip().lower() in {"1", "true", "yes", "video", "videos"}

    def _prepare_append_streaming_input(
        self,
        frames: list[OnlinePrefillFrame],
        *,
        prefix_text: str = "",
        suffix_text: str = "",
        stream_end: bool = False,
        mm_processor_kwargs: dict[str, Any] | None = None,
        visual_memory: OnlinePrefillVisualMemory | None = None,
    ) -> tuple[StreamingInput, int]:
        images = [self._decode_frame(frame) for frame in frames]
        mm_processor_kwargs = dict(mm_processor_kwargs or {})
        if (
            images
            and self._visual_token_merger_enabled(mm_processor_kwargs)
            and self._visual_token_merger_uses_video_stream()
        ):
            if visual_memory is not None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "visual_memory prefix cannot be combined with "
                        "online-prefill video stream input mode"
                    ),
                )
            prompt, frame_token_sizes = self._preprocess_video_prompt(
                images,
                prefix_text=prefix_text,
                suffix_text=suffix_text,
                mm_processor_kwargs=mm_processor_kwargs,
            )
        else:
            prompt, frame_token_sizes = self._preprocess_prompt(
                images,
                prefix_text=prefix_text,
                suffix_text=suffix_text,
                mm_processor_kwargs=mm_processor_kwargs,
                visual_memory=visual_memory,
            )
        return (
            StreamingInput(
                prompt=prompt,
                online_prefill_enabled=True,
                stream_end=stream_end,
                frame_token_sizes=frame_token_sizes,
            ),
            self._prompt_token_count(prompt),
        )

    def _preprocess_prompt(
        self,
        images: list[Image.Image],
        *,
        prefix_text: str = "",
        suffix_text: str = "",
        mm_processor_kwargs: dict[str, Any] | None = None,
        visual_memory: OnlinePrefillVisualMemory | None = None,
    ) -> tuple[ProcessorInputs, list[int] | None]:
        raw_prompt = f"{prefix_text}{QWEN_VL_IMAGE_PLACEHOLDER * len(images)}{suffix_text}"
        preprocess_input: dict[str, Any] = {"prompt": raw_prompt}
        multi_modal_data: dict[str, Any] = {}
        if visual_memory is not None:
            multi_modal_data["video"] = self._decode_visual_memory(visual_memory)
        if images:
            multi_modal_data["image"] = images
        if multi_modal_data:
            preprocess_input["multi_modal_data"] = multi_modal_data
        if visual_memory is not None and visual_memory.memory_id:
            preprocess_input["multi_modal_uuids"] = {
                "video": [visual_memory.memory_id]
            }
        if mm_processor_kwargs:
            preprocess_input["mm_processor_kwargs"] = dict(mm_processor_kwargs)
        prompt = self.input_preprocessor.preprocess(preprocess_input)
        if not images:
            return prompt, None
        _, decoder_inputs = split_enc_dec_inputs(prompt)
        if decoder_inputs["type"] != "multimodal":
            raise HTTPException(
                status_code=500,
                detail="expected multimodal decoder inputs for frame append",
            )

        prompt_token_ids = decoder_inputs["prompt_token_ids"]
        total_tokens = len(prompt_token_ids)
        mm_placeholders = decoder_inputs.get("mm_placeholders", {})
        image_placeholders = sorted(
            mm_placeholders.get("image", ()), key=lambda x: x.offset
        )

        if image_placeholders and len(image_placeholders) == len(images):
            # Derive frame spans from placeholder offsets. Any non-placeholder
            # tokens (e.g. BOS, wrappers, or suffix text) are absorbed into the
            # nearest image span so that sum(spans) == total_tokens exactly.
            boundaries = [p.offset for p in image_placeholders[1:]]
            boundaries.append(total_tokens)
            frame_token_sizes: list[int] = []
            prev_end = 0
            for end in boundaries:
                frame_token_sizes.append(end - prev_end)
                prev_end = end
            return prompt, frame_token_sizes

        if len(images) == 1:
            return prompt, [total_tokens]

        # Last-resort fallback: we cannot recover per-frame boundaries from the
        # decoder inputs. Return None so the scheduler falls back to
        # chunk-size-only boundaries (non-frame-aligned but safe).
        logger.warning(
            "online_prefill: processor did not surface per-image placeholders "
            "for %d images; falling back to non-frame-aligned chunking",
            len(images),
        )
        return prompt, None

    def _preprocess_video_prompt(
        self,
        images: list[Image.Image],
        *,
        prefix_text: str = "",
        suffix_text: str = "",
        mm_processor_kwargs: dict[str, Any],
    ) -> tuple[ProcessorInputs, list[int] | None]:
        raw_prompt = f"{prefix_text}{QWEN_VL_VIDEO_PLACEHOLDER}{suffix_text}"
        video, metadata = self._images_to_video_item(images)
        processor_kwargs = dict(mm_processor_kwargs)
        processor_kwargs.setdefault("do_sample_frames", False)
        preprocess_input: dict[str, Any] = {
            "prompt": raw_prompt,
            "multi_modal_data": {"video": [(video, metadata)]},
            "mm_processor_kwargs": processor_kwargs,
        }
        prompt = self.input_preprocessor.preprocess(preprocess_input)
        return prompt, [self._prompt_token_count(prompt)]

    @staticmethod
    def _images_to_video_item(images: list[Image.Image]) -> tuple[np.ndarray, dict[str, Any]]:
        if not images:
            raise ValueError("expected at least one frame for online-prefill video chunk")
        rgb_frames = [np.asarray(img.convert("RGB")) for img in images]
        if len(rgb_frames) == 1:
            # Qwen3-VL video processor requires at least the temporal factor
            # (2) frames. Duplicate a singleton online chunk instead of
            # failing the streaming session.
            rgb_frames.append(rgb_frames[0].copy())
        video = np.stack(rgb_frames, axis=0)
        fps = float(max(1, min(30, len(rgb_frames))))
        metadata = {
            "total_num_frames": int(video.shape[0]),
            "fps": fps,
            "width": int(video.shape[2]),
            "height": int(video.shape[1]),
            "duration": float(video.shape[0]) / fps,
            "video_backend": "online_prefill",
            "frames_indices": list(range(int(video.shape[0]))),
            "do_sample_frames": False,
        }
        return video, metadata

    @staticmethod
    def _extract_mm_hashes(prompt: ProcessorInputs, modality: str) -> list[str]:
        try:
            _, decoder_inputs = split_enc_dec_inputs(prompt)
        except (KeyError, TypeError, ValueError):
            return []
        if decoder_inputs.get("type") != "multimodal":
            return []
        mm_hashes = decoder_inputs.get("mm_hashes") or {}
        hashes = mm_hashes.get(modality) or []
        return [str(item) for item in hashes if item is not None]

    def _export_metadata_from_prompt(
        self,
        prompt: ProcessorInputs,
        *,
        frame_count: int,
        mm_processor_kwargs: dict[str, Any],
    ) -> tuple[list[str], list[int]]:
        if frame_count <= 0:
            return [], []
        if (
            self._visual_token_merger_enabled(mm_processor_kwargs)
            and self._visual_token_merger_uses_video_stream()
        ):
            hashes = self._extract_mm_hashes(prompt, "video")
            return hashes, ([frame_count] if hashes else [])
        hashes = self._extract_mm_hashes(prompt, "image")
        return hashes, [1 for _ in hashes]

    @staticmethod
    def _decode_visual_memory(
        visual_memory: OnlinePrefillVisualMemory,
    ) -> dict[str, torch.Tensor | list[list[float]]]:
        payload = visual_memory.data
        if payload.startswith("data:"):
            _, payload = payload.split(",", 1)
        try:
            raw = base64.b64decode(payload, validate=True)
            with torch.sparse.check_sparse_tensor_invariants():
                embeddings = torch.load(
                    io.BytesIO(raw),
                    map_location="cpu",
                    weights_only=True,
                )
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"invalid visual_memory tensor payload: {exc}",
            ) from exc

        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.to_dense() if embeddings.is_sparse else embeddings
        else:
            raise HTTPException(
                status_code=400,
                detail="visual_memory payload must decode to a torch.Tensor",
            )
        if embeddings.ndim != 2:
            raise HTTPException(
                status_code=400,
                detail=(
                    "visual_memory tensor must be 2D "
                    f"(tokens, hidden_size), got shape {tuple(embeddings.shape)}"
                ),
            )

        num_frames = int(visual_memory.num_frames)
        if num_frames <= 0:
            raise HTTPException(
                status_code=400,
                detail="visual_memory.num_frames must be positive",
            )
        total_tokens = int(embeddings.shape[0])
        if visual_memory.tokens_per_frame is None:
            if total_tokens % num_frames != 0:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "visual_memory.tokens_per_frame is required when token "
                        "count is not divisible by num_frames"
                    ),
                )
            tokens_per_frame = total_tokens // num_frames
        else:
            tokens_per_frame = int(visual_memory.tokens_per_frame)

        if tokens_per_frame <= 0:
            raise HTTPException(
                status_code=400,
                detail="visual_memory.tokens_per_frame must be positive",
            )
        if tokens_per_frame * num_frames != total_tokens:
            raise HTTPException(
                status_code=400,
                detail=(
                    "visual_memory tensor length must equal "
                    "num_frames * tokens_per_frame"
                ),
            )

        timestamps = visual_memory.timestamps
        if timestamps is None:
            timestamps = [float(i) for i in range(num_frames)]
        if len(timestamps) != num_frames:
            raise HTTPException(
                status_code=400,
                detail="visual_memory.timestamps length must equal num_frames",
            )

        # Synthetic Qwen3-VL video grid: after spatial_merge_size=2, this
        # becomes exactly `tokens_per_frame` video tokens per memory frame.
        video_grid_thw = torch.tensor(
            [[num_frames, 2, 2 * tokens_per_frame]], dtype=torch.long
        )
        return {
            "video_embeds": embeddings.contiguous(),
            "video_grid_thw": video_grid_thw,
            "timestamps": [list(map(float, timestamps))],
        }

    async def _export_session_visual_memory(
        self,
        session: _OnlinePrefillSession,
    ) -> OnlinePrefillVisualMemory | None:
        if not session.visual_memory_export_hashes:
            return None

        exported = await self.engine_client.export_visual_memory_cache(
            session.visual_memory_export_hashes
        )
        tensors = self._flatten_exported_visual_tensors(exported)
        frames = self._materialize_export_frames(
            tensors,
            session.visual_memory_export_frame_counts,
        )
        if not frames:
            raise RuntimeError("no cached visual tensors were available for export")

        memory_tensor = self._build_visual_memory_tensor(
            frames,
            num_frames=session.export_visual_memory_num_frames,
            tokens_per_frame=session.export_visual_memory_tokens_per_frame,
        )
        payload = io.BytesIO()
        torch.save(memory_tensor.contiguous(), payload)
        memory_id = session.export_visual_memory_id or (
            f"bava_mem:{session.request_id}:{session.appended_frames}:"
            f"{session.export_visual_memory_num_frames}x"
            f"{session.export_visual_memory_tokens_per_frame}"
        )
        return OnlinePrefillVisualMemory(
            data=base64.b64encode(payload.getvalue()).decode(),
            memory_id=memory_id,
            text_prefix=session.export_visual_memory_text_prefix,
            num_frames=session.export_visual_memory_num_frames,
            tokens_per_frame=session.export_visual_memory_tokens_per_frame,
            timestamps=[
                float(i) for i in range(session.export_visual_memory_num_frames)
            ],
        )

    @classmethod
    def _materialize_export_frames(
        cls,
        tensors: list[torch.Tensor | None],
        frame_counts: list[int],
    ) -> list[torch.Tensor]:
        frames: list[torch.Tensor] = []
        for idx, tensor in enumerate(tensors):
            if tensor is None:
                continue
            item = cls._strip_visual_position_channels(tensor.detach().cpu())
            if item.ndim != 2 or item.shape[0] <= 0:
                continue
            frame_count = frame_counts[idx] if idx < len(frame_counts) else 1
            if frame_count > 1 and item.shape[0] % frame_count == 0:
                tokens_per_input_frame = item.shape[0] // frame_count
                for frame_idx in range(frame_count):
                    start = frame_idx * tokens_per_input_frame
                    end = start + tokens_per_input_frame
                    frames.append(item[start:end])
            else:
                frames.append(item)
        return frames

    @staticmethod
    def _flatten_exported_visual_tensors(value: Any) -> list[torch.Tensor | None]:
        tensors: list[torch.Tensor | None] = []

        def visit(item: Any) -> None:
            if item is None:
                tensors.append(None)
            elif isinstance(item, torch.Tensor):
                tensors.append(item)
            elif isinstance(item, (bytes, bytearray, memoryview)):
                tensor = torch.load(
                    io.BytesIO(bytes(item)),
                    map_location="cpu",
                    weights_only=True,
                )
                if isinstance(tensor, torch.Tensor):
                    tensors.append(tensor)
            elif isinstance(item, (list, tuple)):
                for child in item:
                    visit(child)

        visit(value)
        return tensors

    @staticmethod
    def _strip_visual_position_channels(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 2 and tensor.shape[-1] > 5:
            candidate_width = tensor.shape[-1] - 5
            if candidate_width > 0 and candidate_width % 4096 == 0:
                return tensor[:, :candidate_width]
        return tensor

    @classmethod
    def _build_visual_memory_tensor(
        cls,
        frames: list[torch.Tensor],
        *,
        num_frames: int,
        tokens_per_frame: int,
    ) -> torch.Tensor:
        if num_frames <= 0 or tokens_per_frame <= 0:
            raise ValueError("num_frames and tokens_per_frame must be positive")
        hidden_size = int(frames[0].shape[-1])
        selected_frames: list[torch.Tensor] = []
        frame_indices = cls._even_indices(len(frames), num_frames)
        for frame_idx in frame_indices:
            frame = frames[frame_idx]
            if frame.shape[-1] != hidden_size:
                raise ValueError(
                    "cannot export visual memory from mixed hidden sizes: "
                    f"{hidden_size} and {frame.shape[-1]}"
                )
            selected_frames.append(cls._resample_frame_tokens(frame, tokens_per_frame))
        return torch.cat(selected_frames, dim=0).contiguous()

    @staticmethod
    def _even_indices(source_count: int, target_count: int) -> list[int]:
        if source_count <= 0:
            raise ValueError("source_count must be positive")
        if target_count == 1:
            return [source_count - 1]
        return [
            int(round(float(i) * float(source_count - 1) / float(target_count - 1)))
            for i in range(target_count)
        ]

    @classmethod
    def _resample_frame_tokens(
        cls,
        frame: torch.Tensor,
        target_tokens: int,
    ) -> torch.Tensor:
        if frame.shape[0] <= 0:
            raise ValueError("cannot resample an empty visual frame")
        indices = cls._even_indices(int(frame.shape[0]), target_tokens)
        return frame.index_select(0, torch.tensor(indices, dtype=torch.long))

    @staticmethod
    def _decode_frame(frame: OnlinePrefillFrame) -> Image.Image:
        payload = frame.data
        if payload.startswith("data:"):
            _, payload = payload.split(",", 1)
        image_bytes = base64.b64decode(payload)
        image = Image.open(io.BytesIO(image_bytes))
        return image.convert("RGB")

    @staticmethod
    def _prompt_token_count(prompt: ProcessorInputs) -> int:
        _, decoder_inputs = split_enc_dec_inputs(prompt)
        if decoder_inputs["type"] == "embeds":
            return int(decoder_inputs["prompt_embeds"].shape[0])
        return len(decoder_inputs["prompt_token_ids"])


def _get_manager(raw_request: Request) -> OnlinePrefillSessionManager:
    if raw_request.app.state.engine_client is None:
        raise HTTPException(status_code=503, detail="engine is not available")

    if not raw_request.app.state.vllm_config.scheduler_config.enable_online_prefill:
        raise HTTPException(
            status_code=400,
            detail="server was not started with --enable-online-prefill",
        )

    manager = getattr(raw_request.app.state, "online_prefill_session_manager", None)
    if manager is None:
        manager = OnlinePrefillSessionManager(raw_request.app.state.engine_client)
        raw_request.app.state.online_prefill_session_manager = manager
    return manager


def _to_response(session: _OnlinePrefillSession) -> OnlinePrefillSessionResponse:
    return OnlinePrefillSessionResponse(
        request_id=session.request_id,
        status=session.status,
        created_at=session.created_at,
        updated_at=session.updated_at,
        appended_chunks=session.appended_chunks,
        appended_frames=session.appended_frames,
        stream_end_received=session.stream_end_received,
        decode_started=session.decode_started,
        finished=session.finished,
        output_text=session.output_text,
        error=session.error,
        visual_memory=session.exported_visual_memory,
        visual_memory_error=session.visual_memory_error,
        visual_memory_export_pending=session.visual_memory_export_pending,
        visual_memory_prefix_cache_warmup_error=(
            session.visual_memory_prefix_cache_warmup_error
        ),
    )


@router.post("/v1/online_prefill/sessions", response_model=OnlinePrefillSessionResponse)
async def create_online_prefill_session(
    request: OnlinePrefillCreateRequest, raw_request: Request
):
    manager = _get_manager(raw_request)
    session = await manager.create_session(request)
    return _to_response(session)


@router.post(
    "/v1/online_prefill/sessions/{request_id}/append",
    response_model=OnlinePrefillSessionResponse,
)
async def append_online_prefill_session(
    request_id: str,
    request: OnlinePrefillAppendRequest,
    raw_request: Request,
):
    manager = _get_manager(raw_request)
    session = await manager.append(request_id, request)
    return _to_response(session)


@router.get(
    "/v1/online_prefill/sessions/{request_id}",
    response_model=OnlinePrefillSessionResponse,
)
async def get_online_prefill_session(request_id: str, raw_request: Request):
    manager = _get_manager(raw_request)
    session = await manager.get_session(request_id)
    return _to_response(session)


@router.get(
    "/v1/online_prefill/sessions/{request_id}/result",
    response_model=OnlinePrefillSessionResponse,
)
async def get_online_prefill_result(request_id: str, raw_request: Request):
    manager = _get_manager(raw_request)
    session = await manager.get_session(request_id)
    return _to_response(session)


@router.delete("/v1/online_prefill/sessions/{request_id}")
async def delete_online_prefill_session(request_id: str, raw_request: Request):
    manager = _get_manager(raw_request)
    await manager.abort(request_id)
    return {"request_id": request_id, "status": "aborted"}


def attach_router(app: FastAPI) -> None:
    app.include_router(router)

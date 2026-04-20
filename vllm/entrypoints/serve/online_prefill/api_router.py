# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
import base64
import io
import time
from dataclasses import dataclass, field
from typing import Any

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
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
_STREAM_DONE = object()


class OnlinePrefillFrame(BaseModel):
    data: str
    mime_type: str = "image/jpeg"
    file_name: str | None = None


class OnlinePrefillCreateRequest(BaseModel):
    request_id: str
    prompt: str
    model: str | None = None
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0


class OnlinePrefillAppendRequest(BaseModel):
    frames: list[OnlinePrefillFrame] = Field(default_factory=list)
    stream_end: bool = False


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


@dataclass
class _OnlinePrefillSession:
    request_id: str
    prompt: str
    sampling_params: SamplingParams
    created_at: float
    updated_at: float
    queue: asyncio.Queue[StreamingInput | _QueuedAppend | object]
    generation_task: asyncio.Task[None]
    prefix_prompt: ProcessorInputs
    suffix_prompt: ProcessorInputs
    status: str = "created"
    appended_chunks: int = 0
    appended_frames: int = 0
    stream_end_received: bool = False
    decode_started: bool = False
    finished: bool = False
    output_text: str | None = None
    error: str | None = None
    prompt_token_counts: list[int] = field(default_factory=list)


@dataclass
class _QueuedAppend:
    frames: list[OnlinePrefillFrame] = field(default_factory=list)
    stream_end: bool = False


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

            sampling_params = SamplingParams(
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
            )

            prefix_prompt = self._preprocess_text_prompt(
                self._build_prefix_text(request.system_prompt)
            )
            suffix_prompt = self._preprocess_text_prompt(
                self._build_suffix_text(request.prompt)
            )

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
                prefix_prompt=prefix_prompt,
                suffix_prompt=suffix_prompt,
            )
            session.generation_task = asyncio.create_task(self._run_session(session))
            self._sessions[request.request_id] = session

            await queue.put(
                StreamingInput(
                    prompt=prefix_prompt,
                    sampling_params=sampling_params,
                    online_prefill_enabled=True,
                )
            )
            session.status = "streaming"
            session.prompt_token_counts.append(self._prompt_token_count(prefix_prompt))
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
                )
            )

        if append_request.frames:
            session.appended_chunks += 1
            session.appended_frames += len(append_request.frames)

        if append_request.stream_end:
            session.stream_end_received = True
            await session.queue.put(_STREAM_DONE)
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
                if item.frames:
                    streaming_input, prompt_token_count = await asyncio.to_thread(
                        self._prepare_append_streaming_input,
                        item.frames,
                    )
                    session.prompt_token_counts.append(prompt_token_count)
                    yield streaming_input
                if item.stream_end:
                    session.prompt_token_counts.append(
                        self._prompt_token_count(session.suffix_prompt)
                    )
                    yield StreamingInput(
                        prompt=session.suffix_prompt,
                        online_prefill_enabled=True,
                        stream_end=True,
                    )

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
        if output.finished:
            session.finished = True
            session.status = "finished"

    @staticmethod
    def _build_prefix_text(system_prompt: str) -> str:
        if system_prompt:
            return (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                "<|im_start|>user\n"
            )
        return "<|im_start|>user\n"

    @staticmethod
    def _build_suffix_text(prompt: str) -> str:
        return f"{prompt}<|im_end|>\n<|im_start|>assistant\n"

    def _preprocess_text_prompt(self, prompt: str) -> ProcessorInputs:
        return self.input_preprocessor.preprocess({"prompt": prompt})

    def _prepare_append_streaming_input(
        self,
        frames: list[OnlinePrefillFrame],
    ) -> tuple[StreamingInput, int]:
        images = [self._decode_frame(frame) for frame in frames]
        prompt, frame_token_sizes = self._preprocess_frame_prompt(images)
        return (
            StreamingInput(
                prompt=prompt,
                online_prefill_enabled=True,
                frame_token_sizes=frame_token_sizes,
            ),
            self._prompt_token_count(prompt),
        )

    def _preprocess_frame_prompt(
        self, images: list[Image.Image]
    ) -> tuple[ProcessorInputs, list[int] | None]:
        raw_prompt = QWEN_VL_IMAGE_PLACEHOLDER * len(images)
        prompt = self.input_preprocessor.preprocess(
            {
                "prompt": raw_prompt,
                "multi_modal_data": {"image": images},
            }
        )
        _, decoder_inputs = split_enc_dec_inputs(prompt)
        if decoder_inputs["type"] != "multimodal":
            raise HTTPException(
                status_code=500,
                detail="expected multimodal decoder inputs for frame append",
            )

        prompt_token_ids = decoder_inputs["prompt_token_ids"]
        if len(images) == 1:
            return prompt, [len(prompt_token_ids)]
        return prompt, None

    def _compute_frame_token_sizes(self, images: list[Image.Image]) -> list[int]:
        frame_token_sizes: list[int] = []
        previous_total = 0
        for image_count in range(1, len(images) + 1):
            prefix_prompt = self.input_preprocessor.preprocess(
                {
                    "prompt": QWEN_VL_IMAGE_PLACEHOLDER * image_count,
                    "multi_modal_data": {"image": images[:image_count]},
                }
            )
            _, decoder_inputs = split_enc_dec_inputs(prefix_prompt)
            if decoder_inputs["type"] != "multimodal":
                raise HTTPException(
                    status_code=500,
                    detail="expected multimodal decoder inputs for frame append",
                )

            current_total = len(decoder_inputs["prompt_token_ids"])
            frame_token_sizes.append(current_total - previous_total)
            previous_total = current_total
        return frame_token_sizes

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

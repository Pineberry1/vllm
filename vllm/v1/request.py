# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.v1.engine import (
    EngineCoreEvent,
    EngineCoreEventType,
    EngineCoreRequest,
    FinishReason,
)
from vllm.v1.structured_output.request import StructuredOutputRequest
from vllm.v1.utils import ConstantList

if TYPE_CHECKING:
    from vllm.lora.request import LoRARequest
    from vllm.v1.core.kv_cache_utils import BlockHash


@dataclass
class StreamingUpdate:
    """Lightweight data for streaming session continuation.

    Contains only the fields needed to update an existing streaming session
    with new input data.
    """

    mm_features: list[MultiModalFeatureSpec] | None
    prompt_token_ids: list[int] | None
    max_tokens: int
    arrival_time: float
    sampling_params: SamplingParams | None
    online_prefill_enabled: bool = False
    stream_end: bool = False
    frame_token_sizes: list[int] | None = None

    @classmethod
    def from_request(cls, request: "Request") -> "StreamingUpdate | None":
        if not request.resumable:
            return None
        return cls(
            mm_features=request.mm_features,
            prompt_token_ids=request.prompt_token_ids,
            max_tokens=request.max_tokens,
            arrival_time=request.arrival_time,
            sampling_params=request.sampling_params,
            online_prefill_enabled=request.is_online_prefill_request,
            stream_end=request.online_stream_ended,
            frame_token_sizes=request.frame_token_sizes,
        )


class Request:
    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int] | None,
        sampling_params: SamplingParams | None,
        pooling_params: PoolingParams | None,
        client_index: int = 0,
        arrival_time: float | None = None,
        prompt_embeds: torch.Tensor | None = None,
        mm_features: list[MultiModalFeatureSpec] | None = None,
        lora_request: "LoRARequest | None" = None,
        cache_salt: str | None = None,
        priority: int = 0,
        trace_headers: Mapping[str, str] | None = None,
        block_hasher: Callable[["Request"], list["BlockHash"]] | None = None,
        resumable: bool = False,
        reasoning_ended: bool | None = None,
        online_prefill_enabled: bool = False,
        stream_end: bool = False,
        frame_token_sizes: list[int] | None = None,
    ) -> None:
        self.request_id = request_id
        self.client_index = client_index
        self.priority = priority
        self.sampling_params = sampling_params
        self.pooling_params = pooling_params
        self.lora_request = lora_request
        self.structured_output_request = StructuredOutputRequest.from_sampling_params(
            sampling_params
        )
        if self.structured_output_request is not None:
            self.structured_output_request.reasoning_ended = reasoning_ended
        self.arrival_time = arrival_time if arrival_time is not None else time.time()

        self.status = RequestStatus.WAITING
        self.events: list[EngineCoreEvent] = []
        self.stop_reason: int | str | None = None

        # P/D: Connector-specific KV transfer parameters.
        self.kv_transfer_params: dict[str, Any] | None = None

        if pooling_params is not None:
            # Pooling models.
            self.max_tokens = 1
        elif sampling_params is not None:
            # Generative models.
            assert sampling_params.max_tokens is not None
            self.max_tokens = sampling_params.max_tokens
            if self.structured_output_request is not None:
                self.status = RequestStatus.WAITING_FOR_FSM

            if sampling_params.extra_args is not None:
                self.kv_transfer_params = sampling_params.extra_args.get(
                    "kv_transfer_params"
                )
        else:
            raise ValueError("sampling_params and pooling_params can't both be unset")

        self.prompt_token_ids = prompt_token_ids
        self.prompt_embeds = prompt_embeds
        # Cache per-block prompt-embed hashes to avoid rehashing the same
        # tensor slices when generating extra keys.
        self._prompt_embeds_per_block_hashes: dict[tuple[int, int], bytes] = {}
        self.num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
            prompt_token_ids, prompt_embeds
        )
        self._output_token_ids: list[int] = []
        self._all_token_ids: list[int] = (
            self.prompt_token_ids.copy()
            if self.prompt_token_ids is not None
            else [0] * self.num_prompt_tokens
        )

        # Used in async scheduling.
        self.num_output_placeholders = 0
        # Used in forced preemption (reset_prefix_cache) with async scheduling.
        self.discard_latest_async_tokens = False

        self.spec_token_ids: list[int] = []
        self.num_computed_tokens = 0
        self.cache_salt: str | None = cache_salt

        # Multi-modal related
        self.mm_features = mm_features or []

        # Read-only views
        # Prevent directly appending to these lists since
        # they should also be updated simultaneously.
        self.output_token_ids = ConstantList(self._output_token_ids)
        self.all_token_ids = ConstantList(self._all_token_ids)
        # trace_headers
        self.trace_headers = trace_headers
        # State
        # The number of tokens with prefix cache hits.
        self.num_cached_tokens = -1

        # True if this request is scheduled as a non-final prefill chunk.
        self.is_prefill_chunk = False

        # The number of NaNs in logits. A value greater than 0
        # indicates that the output is corrupted
        self.num_nans_in_logits = 0

        # The number of times this request has been preempted by the scheduler.
        self.num_preemptions = 0

        # The number of tokens that have been computed remotely.
        self.num_external_computed_tokens = 0

        self.block_hashes: list[BlockHash] = []
        # Store the block hasher without binding self to avoid creating a
        # reference cycle (Request -> partial -> Request) that prevents
        # immediate garbage collection via reference counting.
        self._block_hasher: Callable[[Request], list[BlockHash]] | None = block_hasher
        self.update_block_hashes()

        self.skip_reading_prefix_cache = self.get_skip_reading_prefix_cache()

        # Used for streaming
        self.resumable = resumable
        # None entry in the queue means finished.
        self.streaming_queue: deque[StreamingUpdate | None] | None = None
        self.frame_token_sizes = frame_token_sizes
        self.is_online_prefill_request = resumable and online_prefill_enabled
        self.online_prefill_chunk_size = 512
        self.online_stream_ended = stream_end
        self.decode_blocked_until_stream_end = self.is_online_prefill_request
        self.num_prompt_tokens_received = self.num_prompt_tokens
        self.num_prompt_tokens_prefilled = 0
        self.pending_stream_flush = False
        self.deferred_output_token_ids: list[int] = []
        self.online_frame_end_positions: list[int] = []
        if self.is_online_prefill_request:
            self._append_online_frame_end_positions(
                base_prompt_tokens=0,
                added_prompt_tokens=self.num_prompt_tokens,
                frame_token_sizes=frame_token_sizes,
            )
            self.pending_stream_flush = stream_end and self.num_prompt_tokens > 0

    @classmethod
    def from_engine_core_request(
        cls,
        request: EngineCoreRequest,
        block_hasher: Callable[["Request"], list["BlockHash"]] | None,
    ) -> "Request":
        return cls(
            request_id=request.request_id,
            client_index=request.client_index,
            prompt_token_ids=request.prompt_token_ids,
            prompt_embeds=request.prompt_embeds,
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            arrival_time=request.arrival_time,
            lora_request=request.lora_request,
            cache_salt=request.cache_salt,
            priority=request.priority,
            trace_headers=request.trace_headers,
            block_hasher=block_hasher,
            resumable=request.resumable,
            reasoning_ended=request.reasoning_ended,
            online_prefill_enabled=request.online_prefill_enabled,
            stream_end=request.stream_end,
            frame_token_sizes=request.frame_token_sizes,
        )

    def _append_online_frame_end_positions(
        self,
        base_prompt_tokens: int,
        added_prompt_tokens: int,
        frame_token_sizes: list[int] | None,
    ) -> None:
        if not self.is_online_prefill_request or added_prompt_tokens <= 0:
            return

        if frame_token_sizes is None:
            return

        if sum(frame_token_sizes) != added_prompt_tokens:
            raise ValueError(
                "frame_token_sizes must sum to the number of appended prompt tokens"
            )

        running_end = base_prompt_tokens
        for frame_size in frame_token_sizes:
            running_end += frame_size
            self.online_frame_end_positions.append(running_end)

    def rebuild_block_hashes(self) -> None:
        self.block_hashes = []
        self.update_block_hashes()

    def discard_deferred_output_tokens(self) -> None:
        had_output_tokens = bool(self.deferred_output_token_ids or self._output_token_ids)
        if had_output_tokens:
            del self._all_token_ids[self.num_prompt_tokens :]
            self._output_token_ids.clear()
            self.deferred_output_token_ids.clear()

        self.num_output_placeholders = 0
        if self.num_computed_tokens > self.num_tokens:
            self.num_computed_tokens = self.num_tokens
        self.refresh_online_prefill_progress()

        if had_output_tokens:
            self.rebuild_block_hashes()

    def append_prompt_token_ids(
        self,
        new_ids: list[int] | None,
        frame_token_sizes: list[int] | None = None,
    ) -> None:
        if not new_ids:
            return

        if self.prompt_token_ids is None:
            self.prompt_token_ids = []

        base_prompt_tokens = self.num_prompt_tokens
        self.prompt_token_ids.extend(new_ids)
        self._all_token_ids.extend(new_ids)
        self.num_prompt_tokens = len(self.prompt_token_ids)
        self.num_prompt_tokens_received = self.num_prompt_tokens
        self._append_online_frame_end_positions(
            base_prompt_tokens=base_prompt_tokens,
            added_prompt_tokens=len(new_ids),
            frame_token_sizes=frame_token_sizes,
        )
        self.rebuild_block_hashes()

    def mark_stream_end(self) -> None:
        self.online_stream_ended = True
        self.pending_stream_flush = self.get_unprefilled_prompt_len() > 0

    def get_unprefilled_prompt_len(self) -> int:
        return self.num_prompt_tokens_received - self.num_prompt_tokens_prefilled

    def refresh_online_prefill_progress(self) -> None:
        if not self.is_online_prefill_request:
            return

        self.num_prompt_tokens_prefilled = min(
            self.num_computed_tokens, self.num_prompt_tokens_received
        )
        if self.online_stream_ended:
            self.pending_stream_flush = (
                self.num_prompt_tokens_prefilled < self.num_prompt_tokens_received
            )

    def clamp_online_prefill_computed_tokens(self) -> None:
        if not self.is_online_prefill_request or not self.decode_blocked_until_stream_end:
            return
        if self.num_computed_tokens > self.num_tokens:
            self.num_computed_tokens = self.num_tokens
        self.refresh_online_prefill_progress()

    def take_deferred_output_token_ids(self) -> list[int]:
        deferred = self.deferred_output_token_ids
        self.deferred_output_token_ids = []
        return deferred

    def should_wait_for_online_input(self) -> bool:
        return (
            self.is_online_prefill_request
            and self.decode_blocked_until_stream_end
            and not self.online_stream_ended
            and self.get_unprefilled_prompt_len() < self.online_prefill_chunk_size
        )

    def get_online_prefill_schedulable_tokens(
        self, token_budget: int | None = None
    ) -> int:
        if (
            not self.is_online_prefill_request
            or not self.decode_blocked_until_stream_end
        ):
            return 0

        unprefilled = self.get_unprefilled_prompt_len()
        if unprefilled <= 0:
            return 0

        target_end = self.num_prompt_tokens_prefilled
        if self.online_stream_ended:
            self.pending_stream_flush = True
            target_end = self.num_prompt_tokens_received
        else:
            if unprefilled < self.online_prefill_chunk_size:
                return 0
            min_target_end = (
                self.num_prompt_tokens_prefilled + self.online_prefill_chunk_size
            )
            if self.online_frame_end_positions:
                for frame_end in self.online_frame_end_positions:
                    if frame_end >= min_target_end:
                        target_end = frame_end
                        break
                else:
                    return 0
            else:
                target_end = min_target_end

        if token_budget is not None:
            if token_budget <= 0:
                return 0
            max_target_end = self.num_prompt_tokens_prefilled + token_budget
            if target_end > max_target_end:
                # Prefer a whole-frame boundary that fits in this round. If no
                # boundary fits, fall back to the round budget rather than
                # stalling the request forever.
                capped_target_end = self.num_prompt_tokens_prefilled
                for frame_end in self.online_frame_end_positions:
                    if frame_end <= max_target_end:
                        capped_target_end = frame_end
                    else:
                        break
                if capped_target_end > self.num_prompt_tokens_prefilled:
                    target_end = capped_target_end
                else:
                    target_end = max_target_end

        return max(0, target_end - self.num_prompt_tokens_prefilled)

    def append_output_token_ids(
        self,
        token_ids: int | list[int],
    ) -> None:
        if isinstance(token_ids, int):
            self._output_token_ids.append(token_ids)
            self._all_token_ids.append(token_ids)
        else:
            self._output_token_ids.extend(token_ids)
            self._all_token_ids.extend(token_ids)

        self.update_block_hashes()

    def update_block_hashes(self) -> None:
        """Compute block hashes for any new full blocks and append them."""
        if self._block_hasher is not None:
            self.block_hashes.extend(self._block_hasher(self))

    @property
    def use_structured_output(self) -> bool:
        return self.structured_output_request is not None

    @property
    def num_tokens(self) -> int:
        return len(self._all_token_ids)

    @property
    def num_tokens_with_spec(self) -> int:
        return len(self._all_token_ids) + len(self.spec_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self._output_token_ids)

    @property
    def num_encoder_inputs(self) -> int:
        return len(self.mm_features)

    @property
    def has_encoder_inputs(self) -> bool:
        return self.num_encoder_inputs > 0

    def get_skip_reading_prefix_cache(self) -> bool:
        if (
            self.sampling_params is not None
            and self.sampling_params.skip_reading_prefix_cache is not None
        ):
            return self.sampling_params.skip_reading_prefix_cache
        elif (
            self.pooling_params is not None
            and self.pooling_params.skip_reading_prefix_cache is not None
        ):
            return self.pooling_params.skip_reading_prefix_cache
        return False

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

    def get_finished_reason(self) -> FinishReason | None:
        return RequestStatus.get_finished_reason(self.status)

    def get_num_encoder_embeds(self, input_id: int) -> int:
        assert input_id < len(self.mm_features)
        return self.mm_features[input_id].mm_position.get_num_embeds()

    def record_event(
        self,
        event_type: EngineCoreEventType,
        timestamp: float | None = None,
    ) -> None:
        self.events.append(EngineCoreEvent.new_event(event_type, timestamp))

    def take_events(self) -> list[EngineCoreEvent] | None:
        if not self.events:
            return None
        events, self.events = self.events, []
        return events

    def __lt__(self, other: "Request") -> bool:
        """
        Compare two requests based on priority, arrival time, and request ID.
        Used in priority scheduling.
        """
        if self.priority != other.priority:
            return self.priority < other.priority
        if self.arrival_time != other.arrival_time:
            return self.arrival_time < other.arrival_time
        if self.request_id != other.request_id:
            return self.request_id < other.request_id
        return id(self) < id(other)


class RequestStatus(enum.IntEnum):
    """Status of a request."""

    WAITING = enum.auto()
    WAITING_FOR_FSM = enum.auto()
    WAITING_FOR_REMOTE_KVS = enum.auto()
    WAITING_FOR_STREAMING_REQ = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    # Note: anything after PREEMPTED will be considered
    # as a finished status.
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()
    FINISHED_ERROR = enum.auto()
    FINISHED_REPETITION = enum.auto()

    def __str__(self) -> str:
        return self.name

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status > RequestStatus.PREEMPTED

    @staticmethod
    def get_finished_reason(status: "RequestStatus") -> FinishReason | None:
        return _FINISHED_REASON_MAP.get(status)


# Mapping of finished statuses to their finish reasons.
# NOTE: The ignored requests are the requests whose prompt lengths
# are longer than the model's length cap. Therefore, the stop
# reason should also be "length" as in OpenAI API.
_FINISHED_REASON_MAP = {
    RequestStatus.FINISHED_STOPPED: FinishReason.STOP,
    RequestStatus.FINISHED_LENGTH_CAPPED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ABORTED: FinishReason.ABORT,
    RequestStatus.FINISHED_IGNORED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ERROR: FinishReason.ERROR,
    RequestStatus.WAITING_FOR_STREAMING_REQ: FinishReason.STOP,
    RequestStatus.FINISHED_REPETITION: FinishReason.REPETITION,
}

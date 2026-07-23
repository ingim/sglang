# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0
from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import msgspec

if TYPE_CHECKING:
    from pie_plex import AsyncRuntime

    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler import Scheduler

PLEX_API_VERSION = "pie.plex.engine@1"


@dataclass(frozen=True)
class PlexSchedulePlan:
    ranks: dict[str, int]
    token_budgets: dict[str, int]

    def selects(self, request_id: str) -> bool:
        return request_id in self.ranks

    def rank(self, request_id: str) -> int | None:
        return self.ranks.get(request_id)


class AsyncPlexPolicyController:
    def __init__(
        self,
        runtime: AsyncRuntime,
        *,
        model: str,
        target_id: str,
    ) -> None:
        self.runtime = runtime
        self.model = model
        self.target_id = target_id
        self.epoch = 0
        self.dirty = False
        self.feedback_sequence = 0
        self._engine_to_logical: dict[str, str] = {}
        self._logical_to_engine: dict[str, str] = {}
        self._request_metadata: dict[str, dict[str, Any]] = {}
        self._terminal_on_complete: dict[str, bool] = {}
        self._completion_event: dict[str, str] = {}
        self._pending_request_events: deque[dict[str, Any]] = deque()
        self._pending_feedback: deque[dict[str, Any]] = deque()
        self._pending_finishes: deque[str] = deque()
        self._submitted_candidates: dict[int, tuple[str, ...]] = {}
        self._submitted_residents: dict[int, tuple[str, ...]] = {}
        self._seen_schedule_epoch = 0
        self._seen_evict_epoch = 0
        self._resolved_schedule_epoch = 0
        self._resolved_evict_epoch = 0
        self._schedule_plan: tuple[int, PlexSchedulePlan] | None = None
        self._retraction_order: tuple[int, list[int]] | None = None
        self._retraction_ids: tuple[str, ...] = ()

    @classmethod
    def from_policy(
        cls,
        policy: str,
        *,
        model: str,
        target_id: str,
    ) -> AsyncPlexPolicyController:
        try:
            from pie_plex import AsyncRuntime
        except ImportError as error:
            raise ImportError(
                "PLEX policy configured but pie-plex is not installed. "
                "Install SGLang with the 'plex' extra or install pie-plex directly."
            ) from error
        return cls(
            AsyncRuntime(policy, queue_capacity=256),
            model=model,
            target_id=target_id,
        )

    def tracks(self, request_id: str) -> bool:
        return request_id in self._engine_to_logical

    def register_request(self, request: Req) -> None:
        (
            logical_id,
            generation_id,
            metadata,
            terminal,
            completion_event,
        ) = self._request_identity(request)
        previous = self._logical_to_engine.get(logical_id)
        if previous is not None and previous != request.rid:
            raise ValueError(
                f"PLEX logical request {logical_id!r} is already active as "
                f"engine request {previous!r}"
            )
        self._engine_to_logical[request.rid] = logical_id
        self._logical_to_engine[logical_id] = request.rid
        self._request_metadata[logical_id] = metadata
        self._terminal_on_complete[request.rid] = terminal
        self._completion_event[request.rid] = completion_event
        self._pending_request_events.append(
            {
                "op": "create" if generation_id == 0 else "continue",
                "request_id": logical_id,
                "facts": {
                    "generation_id": generation_id,
                    "engine_request_id": request.rid,
                    "arrival_ms": self._arrival_ms(request),
                    "attained_service": request.kv_committed_len,
                },
                "fields": self._request_fields(request, metadata),
            }
        )
        self._invalidate()

    def mark_retracted(self, request: Req) -> None:
        logical_id = self._engine_to_logical.get(request.rid)
        if logical_id is not None:
            self._pending_feedback.append(
                {
                    "event": "preempted",
                    "request_id": logical_id,
                    "facts": {"attained_service": request.kv_committed_len},
                }
            )
        self._invalidate()

    def observe_batch(self, requests: list[Req]) -> None:
        for request in requests:
            if request.finished() and self.tracks(request.rid):
                self.mark_finished(request)

    def mark_finished(self, request: Req) -> None:
        logical_id = self._engine_to_logical.get(request.rid)
        if logical_id is None:
            return
        terminal = self._terminal_on_complete.get(request.rid, True)
        reason = (
            request.finished_reason.to_json()
            if request.finished_reason is not None
            else None
        )
        self._pending_feedback.append(
            {
                "event": self._completion_event.get(
                    request.rid,
                    "finished" if terminal else "generation-finished",
                ),
                "request_id": logical_id,
                "facts": {
                    "reason": reason,
                    "attained_service": request.kv_committed_len,
                    "generated_tokens": len(request.output_ids),
                },
            }
        )
        if terminal:
            self._pending_finishes.append(logical_id)
        self._forget_request(request.rid, preserve_logical_state=not terminal)
        self._invalidate()

    def publish(self, scheduler: Scheduler) -> None:
        if (
            not self.dirty
            and not self._pending_request_events
            and not self._pending_feedback
        ):
            return

        if self.dirty:
            candidates = [
                request
                for request in scheduler.waiting_queue
                if self.tracks(request.rid) and not request.finished()
            ]
            schedule_event = self._schedule_event(
                scheduler,
                candidates,
                list(self._pending_request_events),
            )
            if not self.runtime.try_submit_bytes(
                "schedule",
                self.epoch,
                msgspec.json.encode(schedule_event),
            ):
                return
            self._pending_request_events.clear()
            self._submitted_candidates[self.epoch] = tuple(
                request.rid for request in candidates
            )
            self._trim_submissions(self._submitted_candidates)

            residents = [
                request
                for request in scheduler.running_batch.reqs
                if self.tracks(request.rid) and not request.finished()
            ]
            if residents:
                if self.runtime.try_submit_bytes(
                    "evict",
                    self.epoch,
                    msgspec.json.encode(self._evict_event(scheduler, residents)),
                ):
                    self._submitted_residents[self.epoch] = tuple(
                        request.rid for request in residents
                    )
                    self._trim_submissions(self._submitted_residents)
            self.dirty = False

        if self._pending_feedback:
            self.feedback_sequence += 1
            event = {
                "api_version": PLEX_API_VERSION,
                "hook": "feedback",
                "context": {
                    "delivery_id": (
                        f"sglang:{self.target_id}:{self.feedback_sequence}"
                    ),
                    "records": list(self._pending_feedback),
                    "context": self._hook_context(),
                },
                "request_events": [
                    {"op": "finish", "request_id": request_id}
                    for request_id in self._pending_finishes
                ],
            }
            if self.runtime.try_submit_bytes(
                "feedback",
                self.epoch,
                msgspec.json.encode(event),
            ):
                self._pending_feedback.clear()
                self._pending_finishes.clear()

    def poll_schedule(self) -> PlexSchedulePlan | None:
        if self._resolved_schedule_epoch == self.epoch:
            return (
                self._schedule_plan[1]
                if self._schedule_plan is not None
                and self._schedule_plan[0] == self.epoch
                else None
            )
        result = self.runtime.latest("schedule", self._seen_schedule_epoch)
        if result is not None:
            epoch, outcome = result
            self._seen_schedule_epoch = epoch
            request_ids = self._submitted_candidates.pop(epoch, ())
            if outcome.get("status") == "success":
                ranks: dict[str, int] = {}
                token_budgets: dict[str, int] = {}
                for rank, item in enumerate(
                    outcome.get("decision", {}).get("selected", [])
                ):
                    index = item.get("candidate_index")
                    if (
                        isinstance(index, int)
                        and not isinstance(index, bool)
                        and 0 <= index < len(request_ids)
                    ):
                        request_id = request_ids[index]
                        ranks[request_id] = rank
                        token_budgets[request_id] = (1 << 63) - 1
                self._schedule_plan = (
                    epoch,
                    PlexSchedulePlan(ranks, token_budgets),
                )
            else:
                self._schedule_plan = None
            if epoch == self.epoch:
                self._resolved_schedule_epoch = epoch
        if self._schedule_plan is None or self._schedule_plan[0] != self.epoch:
            return None
        return self._schedule_plan[1]

    def cached_retraction_order(self, requests: list[Req]) -> list[int] | None:
        if self._resolved_evict_epoch == self.epoch:
            if (
                self._retraction_order is None
                or self._retraction_order[0] != self.epoch
                or self._retraction_ids != tuple(request.rid for request in requests)
            ):
                return None
            return list(self._retraction_order[1])
        result = self.runtime.latest("evict", self._seen_evict_epoch)
        if result is not None:
            epoch, outcome = result
            self._seen_evict_epoch = epoch
            request_ids = self._submitted_residents.pop(epoch, ())
            self._retraction_ids = request_ids
            selected_indices = []
            if outcome.get("status") == "success":
                for item in outcome.get("decision", {}).get("selected", []):
                    index = item.get("candidate_index")
                    if (
                        isinstance(index, int)
                        and not isinstance(index, bool)
                        and 0 <= index < len(request_ids)
                    ):
                        selected_indices.append(index)
            selected = set(selected_indices)
            keepers = [
                index for index in range(len(request_ids)) if index not in selected
            ]
            self._retraction_order = (
                epoch,
                [*keepers, *reversed(selected_indices)],
            )
            if epoch == self.epoch:
                self._resolved_evict_epoch = epoch

        if self._retraction_order is None or self._retraction_order[0] != self.epoch:
            return None
        current_ids = tuple(request.rid for request in requests)
        if self._retraction_ids != current_ids:
            return None
        return list(self._retraction_order[1])

    def close(self) -> None:
        self.runtime.shutdown()

    def _invalidate(self) -> None:
        self.epoch += 1
        self.dirty = True
        self._schedule_plan = None
        self._retraction_order = None
        self._resolved_schedule_epoch = 0
        self._resolved_evict_epoch = 0

    def _forget_request(self, request_id: str, *, preserve_logical_state: bool) -> None:
        logical_id = self._engine_to_logical.pop(request_id, None)
        self._terminal_on_complete.pop(request_id, None)
        self._completion_event.pop(request_id, None)
        if logical_id is not None:
            self._logical_to_engine.pop(logical_id, None)
            if not preserve_logical_state:
                self._request_metadata.pop(logical_id, None)

    def _schedule_event(
        self,
        scheduler: Scheduler,
        candidates: list[Req],
        lifecycle_events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        max_tokens = scheduler.max_prefill_tokens
        max_budget = max(
            (self._max_prefill_budget(request, max_tokens) for request in candidates),
            default=0,
        )
        return {
            "api_version": PLEX_API_VERSION,
            "hook": "schedule",
            "context": {
                "runnable": [
                    {
                        "request_id": self._engine_to_logical[request.rid],
                        "facts": self._facts(request),
                        "max_token_budget": self._max_prefill_budget(
                            request, max_tokens
                        ),
                    }
                    for request in candidates
                ],
                "capacity": {
                    "max_selected": min(
                        len(candidates), scheduler.max_running_requests
                    ),
                    "max_total_tokens": max_tokens,
                    "max_token_budget": max_budget,
                },
                "context": self._hook_context(
                    {"mode": "async-indexed", "epoch": self.epoch}
                ),
            },
            "request_events": [
                *lifecycle_events,
                *(
                    {
                        "op": "merge-facts",
                        "request_id": self._engine_to_logical[request.rid],
                        "facts": self._facts(request),
                    }
                    for request in candidates
                ),
            ],
        }

    def _evict_event(
        self, scheduler: Scheduler, residents: list[Req]
    ) -> dict[str, Any]:
        bytes_per_token = self._bytes_per_token(scheduler)
        return {
            "api_version": PLEX_API_VERSION,
            "hook": "evict",
            "context": {
                "resident": [
                    {
                        "id": request.rid,
                        "request_id": self._engine_to_logical[request.rid],
                        "size_bytes": max(
                            self._allocated_tokens(request) * bytes_per_token,
                            bytes_per_token,
                        ),
                        "facts": {
                            **self._facts(request),
                            "reload_cost": request.kv_committed_len,
                        },
                    }
                    for request in residents
                ],
                "bytes_needed": max(bytes_per_token, 1),
                "context": self._hook_context(
                    {"mode": "async-indexed", "epoch": self.epoch}
                ),
            },
            "request_events": [],
        }

    def _target_facts(self, scheduler: Scheduler) -> dict[str, Any]:
        allocator = scheduler.token_to_kv_pool_allocator
        return {
            "queue_depth": len(scheduler.waiting_queue),
            "running_requests": len(scheduler.running_batch.reqs),
            "free_kv_tokens": allocator.available_size(),
            "total_kv_tokens": allocator.size_full,
        }

    def _hook_context(self, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        context = {
            "engine": "sglang",
            "model": self.model,
            "target_id": self.target_id,
            "capabilities": {"queries": []},
        }
        context.update(extra or {})
        return context

    def _facts(self, request: Req) -> dict[str, Any]:
        entered = request.time_stats.wait_queue_entry_time
        if entered <= 0:
            entered = request.time_stats.scheduler_recv_time
        return {
            "engine_request_id": request.rid,
            "attained_service": request.kv_committed_len,
            "generated_tokens": len(request.output_ids),
            "preempted": request.is_retracted,
            "waiting_ms": max(int((time.perf_counter() - entered) * 1000), 0),
            "cached_tokens": request.num_matched_prefix_tokens,
        }

    @staticmethod
    def _max_prefill_budget(request: Req, max_tokens: int) -> int:
        return max(
            min(
                len(request.origin_input_ids)
                + len(request.output_ids)
                - request.num_matched_prefix_tokens,
                max_tokens,
            ),
            0,
        )

    @staticmethod
    def _allocated_tokens(request: Req) -> int:
        if request.kv is not None:
            return request.kv.kv_allocated_len
        return request.kv_committed_len

    @staticmethod
    def _bytes_per_token(scheduler: Scheduler) -> int:
        allocator = scheduler.token_to_kv_pool_allocator
        size = allocator.get_kvcache().get_kv_size_bytes()
        total = sum(size) if isinstance(size, tuple) else size
        return max(int(total / allocator.size_full), 1)

    @staticmethod
    def _trim_submissions(submissions: dict[int, tuple[str, ...]]) -> None:
        while len(submissions) > 256:
            submissions.pop(next(iter(submissions)))

    @staticmethod
    def _request_fields(request: Req, metadata: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "body": {
                "prompt_token_ids": list(request.origin_input_ids),
                "max_tokens": request.sampling_params.max_new_tokens,
                "priority": request.priority,
            },
            "metadata": dict(metadata),
        }

    @staticmethod
    def _arrival_ms(request: Req) -> int:
        created = request.time_stats.created_time
        return int(created * 1000) if created > 0 else 0

    @staticmethod
    def _request_identity(
        request: Req,
    ) -> tuple[str, int, dict[str, Any], bool, str]:
        custom = request.sampling_params.custom_params or {}
        raw = custom.get("plex")
        config: Mapping[str, Any] = {}
        if raw is not None:
            if not isinstance(raw, Mapping):
                raise ValueError(
                    "sampling_params.custom_params['plex'] must be an object"
                )
            config = raw
        logical_id = config.get("logical_request_id", request.rid)
        generation_id = config.get("generation_id", 0)
        terminal = config.get("terminal", True)
        completion_event = config.get(
            "completion_event",
            "finished" if terminal else "generation-finished",
        )
        if not isinstance(logical_id, str) or not logical_id:
            raise ValueError("PLEX logical_request_id must be a non-empty string")
        if (
            not isinstance(generation_id, int)
            or isinstance(generation_id, bool)
            or generation_id < 0
        ):
            raise ValueError("PLEX generation_id must be a non-negative integer")
        if not isinstance(terminal, bool):
            raise ValueError("PLEX terminal must be a boolean")
        if not isinstance(completion_event, str) or not completion_event:
            raise ValueError("PLEX completion_event must be a non-empty string")
        metadata = config.get("metadata")
        if metadata is None:
            metadata = {
                key: value
                for key, value in config.items()
                if key
                not in {
                    "logical_request_id",
                    "generation_id",
                    "terminal",
                    "completion_event",
                }
            }
        if not isinstance(metadata, Mapping):
            raise ValueError("PLEX metadata must be an object")
        json.dumps(metadata)
        return (
            logical_id,
            generation_id,
            dict(metadata),
            terminal,
            completion_event,
        )

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
    from plex import AsyncRuntime

    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler import Scheduler

PLEX_API_VERSION = "plex.plex.engine@2"

DEFAULT_PRINCIPAL = "sglang-default"


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
        self._request_principal: dict[str, str] = {}
        self._request_generation: dict[str, int] = {}
        self._terminal_on_complete: dict[str, bool] = {}
        self._completion_outcome: dict[str, str] = {}
        self._pending_request_events: deque[dict[str, Any]] = deque()
        self._pending_feedback: deque[dict[str, Any]] = deque()
        self._pending_request_cleanup: deque[dict[str, Any]] = deque()
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
            from plex import AsyncRuntime
        except ImportError as error:
            raise ImportError(
                "PLEX policy configured but plex is not installed. "
                "Install SGLang with the 'plex' extra or install plex directly."
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
            principal_id,
            metadata,
            terminal,
            completion_outcome,
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
        self._request_principal[logical_id] = principal_id
        self._request_generation[logical_id] = generation_id
        self._terminal_on_complete[request.rid] = terminal
        self._completion_outcome[request.rid] = completion_outcome
        fields = self._request_fields(request, metadata)
        facts = {
            "generation_id": generation_id,
            "engine_request_id": request.rid,
            "arrival_ms": self._arrival_ms(request),
            "attained_service": request.kv_committed_len,
        }
        if generation_id == 0:
            self._pending_request_events.append(
                {
                    "event": "create-request",
                    "request_id": logical_id,
                    "principal_id": principal_id,
                    "group_id": None,
                    "fields": fields,
                    "facts": facts,
                }
            )
            self._pending_request_events.append(
                {"event": "admit-request", "request_id": logical_id}
            )
        else:
            self._pending_request_events.append(
                {
                    "event": "continue-request",
                    "request_id": logical_id,
                    "fields": fields,
                    "facts": facts,
                }
            )
        self._pending_request_events.append(
            {"event": "activate-request", "request_id": logical_id}
        )
        self._invalidate()

    def mark_retracted(self, request: Req) -> None:
        logical_id = self._engine_to_logical.get(request.rid)
        if logical_id is not None:
            self._pending_feedback.append(
                {
                    "subject": {"kind": "request", "value": logical_id},
                    "outcome": "progress",
                    "facts": {
                        "attained_service": request.kv_committed_len,
                        "preempted": True,
                    },
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
        facts = {
            "reason": reason,
            "attained_service": request.kv_committed_len,
            "generated_tokens": len(request.output_ids),
        }
        if terminal:
            outcome, status = self._terminal_outcome(
                self._completion_outcome.get(request.rid, "auto"),
                self._reason_text(reason),
            )
            self._pending_feedback.append(
                {
                    "subject": {"kind": "request", "value": logical_id},
                    "outcome": outcome,
                    "facts": {"initiator": "host", **facts},
                }
            )
            self._pending_request_cleanup.append(
                {"request_id": logical_id, "status": status}
            )
        else:
            self._pending_feedback.append(
                {
                    "subject": {"kind": "request", "value": logical_id},
                    "outcome": "progress",
                    "facts": {"boundary": True, **facts},
                }
            )
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
                    "cache",
                    self.epoch,
                    msgspec.json.encode(self._cache_event(scheduler, residents)),
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
                "operation": "feedback",
                "context": {
                    "delivery_id": (
                        f"sglang:{self.target_id}:{self.feedback_sequence}"
                    ),
                    "records": list(self._pending_feedback),
                },
                "cleanup": {
                    "requests": list(self._pending_request_cleanup),
                    "groups": [],
                },
            }
            if self.runtime.try_submit_bytes(
                "feedback",
                self.epoch,
                msgspec.json.encode(event),
            ):
                self._pending_feedback.clear()
                self._pending_request_cleanup.clear()

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
            selections = self._plan_body(outcome, "schedule", "selections")
            if selections is not None:
                ranks: dict[str, int] = {}
                token_budgets: dict[str, int] = {}
                for rank, selection in enumerate(selections):
                    parsed = self._single_selection(selection, len(request_ids))
                    if parsed is None:
                        continue
                    index, budget = parsed
                    request_id = request_ids[index]
                    ranks[request_id] = rank
                    token_budgets[request_id] = budget
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
        result = self.runtime.latest("cache", self._seen_evict_epoch)
        if result is not None:
            epoch, outcome = result
            self._seen_evict_epoch = epoch
            request_ids = self._submitted_residents.pop(epoch, ())
            self._retraction_ids = request_ids
            selected_indices = []
            reclaim = self._plan_body(outcome, "cache", "reclaim")
            if reclaim is not None:
                for index in reclaim:
                    if (
                        isinstance(index, int)
                        and not isinstance(index, bool)
                        and 0 <= index < len(request_ids)
                        and index not in selected_indices
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
        self._completion_outcome.pop(request_id, None)
        if logical_id is not None:
            self._logical_to_engine.pop(logical_id, None)
            if not preserve_logical_state:
                self._request_metadata.pop(logical_id, None)
                self._request_principal.pop(logical_id, None)
                self._request_generation.pop(logical_id, None)

    def _schedule_event(
        self,
        scheduler: Scheduler,
        candidates: list[Req],
        lifecycle_events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        max_tokens = scheduler.max_prefill_tokens
        max_selections = min(len(candidates), scheduler.max_running_requests)
        return {
            "api_version": PLEX_API_VERSION,
            "operation": "schedule",
            "context": {
                "meta": self._decision_meta("schedule"),
                "cause": "capacity-changed",
                "runnable": [
                    {
                        "request": self._request_ref(request),
                        "max_token_budget": self._max_prefill_budget(
                            request, max_tokens
                        ),
                        "facts": self._facts(request),
                    }
                    for request in candidates
                ],
                "capacity": {
                    "max_selections": max_selections,
                    "max_requests": max_selections,
                    "max_total_tokens": max_tokens,
                    "facts": self._engine_facts(scheduler),
                },
            },
            "lifecycle": [
                *lifecycle_events,
                *(
                    {
                        "event": "merge-request-facts",
                        "request_id": self._engine_to_logical[request.rid],
                        "facts": self._facts(request),
                    }
                    for request in candidates
                ),
            ],
        }

    def _cache_event(
        self, scheduler: Scheduler, residents: list[Req]
    ) -> dict[str, Any]:
        bytes_per_token = self._bytes_per_token(scheduler)
        resident = [
            {
                "object": {
                    "object_id": f"sglang-kv:{self._engine_to_logical[request.rid]}",
                    "size_bytes": max(
                        self._allocated_tokens(request) * bytes_per_token,
                        bytes_per_token,
                    ),
                    "beneficiaries": [
                        {
                            "kind": "request",
                            "id": self._engine_to_logical[request.rid],
                        }
                    ],
                    "beneficiary_count": 1,
                    "facts": {
                        **self._facts(request),
                        "reload_cost": request.kv_committed_len,
                    },
                },
                "reclaimable": True,
            }
            for request in residents
        ]
        resident_bytes = sum(entry["object"]["size_bytes"] for entry in resident)
        return {
            "api_version": PLEX_API_VERSION,
            "operation": "cache",
            "context": {
                "meta": self._decision_meta("cache"),
                "cause": "pressure",
                "resident": resident,
                "prospective": [],
                "capacity": {
                    "max_bytes": resident_bytes,
                    "fixed_bytes": 0,
                    "facts": self._engine_facts(scheduler),
                },
                "episode": None,
            },
            "lifecycle": [],
        }

    def _engine_facts(self, scheduler: Scheduler) -> dict[str, Any]:
        allocator = scheduler.token_to_kv_pool_allocator
        return {
            "engine": "sglang",
            "model": self.model,
            "target_id": self.target_id,
            "membership_epoch": self.epoch,
            "queue_depth": len(scheduler.waiting_queue),
            "running_requests": len(scheduler.running_batch.reqs),
            "free_kv_tokens": allocator.available_size(),
            "total_kv_tokens": allocator.size_full,
        }

    def _request_ref(self, request: Req) -> dict[str, Any]:
        logical_id = self._engine_to_logical[request.rid]
        return {
            "request_id": logical_id,
            "generation_id": self._request_generation.get(logical_id, 0),
            "group_id": None,
            "principal_id": self._request_principal.get(logical_id, DEFAULT_PRINCIPAL),
        }

    def _decision_meta(self, operation: str) -> dict[str, Any]:
        return {
            "opportunity_id": self._opportunity_id(operation),
            "snapshot": {"id": "host-filled", "revision": 0},
            "attempt": 0,
            "mechanics": [],
        }

    def _opportunity_id(self, operation: str) -> str:
        return f"sglang:{self.target_id}:{operation}:{self.epoch}"

    @staticmethod
    def _plan_body(
        outcome: Mapping[str, Any], operation: str, key: str
    ) -> list[Any] | None:
        if outcome.get("status") != "success" or outcome.get("actions"):
            return None
        plan = outcome.get("plan")
        if not isinstance(plan, Mapping) or plan.get("operation") != operation:
            return None
        body = plan.get("plan")
        if not isinstance(body, Mapping):
            return None
        entries = body.get(key)
        return entries if isinstance(entries, list) else None

    @staticmethod
    def _single_selection(selection: Any, count: int) -> tuple[int, int] | None:
        if not isinstance(selection, Mapping):
            return None
        indices = selection.get("requests")
        budgets = selection.get("token_budgets")
        if (
            not isinstance(indices, list)
            or not isinstance(budgets, list)
            or len(indices) != 1
            or len(budgets) != 1
        ):
            return None
        index = indices[0]
        budget = budgets[0]
        if (
            isinstance(index, int)
            and not isinstance(index, bool)
            and 0 <= index < count
            and isinstance(budget, int)
            and not isinstance(budget, bool)
            and budget > 0
        ):
            return index, budget
        return None

    @staticmethod
    def _terminal_outcome(configured: str, reason: str) -> tuple[str, str]:
        if configured != "auto":
            mapping = {
                "completed": ("completed", "completed"),
                "failed": ("failed", "failed"),
                "cancelled": ("cancelled", "cancelled"),
                "expired": ("expired", "expired"),
            }
            return mapping[configured]
        lower = reason.lower()
        if "abort" in lower or "cancel" in lower:
            return "cancelled", "cancelled"
        if "error" in lower or "fail" in lower:
            return "failed", "failed"
        if "expire" in lower or "timeout" in lower:
            return "expired", "expired"
        return "completed", "completed"

    @staticmethod
    def _reason_text(reason: Any) -> str:
        if isinstance(reason, Mapping):
            return " ".join(
                str(reason[key])
                for key in ("type", "err_type", "message")
                if reason.get(key)
            )
        if reason is None:
            return ""
        return str(reason)

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
    ) -> tuple[str, int, str, dict[str, Any], bool, str]:
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
        principal_id = config.get("principal_id", config.get("tenant", DEFAULT_PRINCIPAL))
        terminal = config.get("terminal", True)
        completion_outcome = config.get("completion_outcome", "auto")
        if not isinstance(logical_id, str) or not logical_id:
            raise ValueError("PLEX logical_request_id must be a non-empty string")
        if (
            not isinstance(generation_id, int)
            or isinstance(generation_id, bool)
            or generation_id < 0
        ):
            raise ValueError("PLEX generation_id must be a non-negative integer")
        if not isinstance(principal_id, str) or not principal_id:
            raise ValueError("PLEX principal_id must be a non-empty string")
        if not isinstance(terminal, bool):
            raise ValueError("PLEX terminal must be a boolean")
        if completion_outcome not in {
            "auto",
            "completed",
            "failed",
            "cancelled",
            "expired",
        }:
            raise ValueError("PLEX completion_outcome is invalid")
        metadata = config.get("metadata")
        if metadata is None:
            metadata = {
                key: value
                for key, value in config.items()
                if key
                not in {
                    "logical_request_id",
                    "generation_id",
                    "principal_id",
                    "tenant",
                    "terminal",
                    "completion_outcome",
                }
            }
        if not isinstance(metadata, Mapping):
            raise ValueError("PLEX metadata must be an object")
        json.dumps(metadata)
        return (
            logical_id,
            generation_id,
            principal_id,
            dict(metadata),
            terminal,
            completion_outcome,
        )

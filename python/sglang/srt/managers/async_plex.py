# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0
"""SGLang's PLEX binding: an `EnginePort` and the scheduler-facing wrapper.

Everything about the v0.7 wire format, request identity, plan freshness and
feedback lives in `plex.engine`, which ships with the contract. What is left
here is the part only SGLang knows: where the scheduler keeps its queues and
what its per-request counters are called.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from plex.engine import CacheCapacity, PolicyController, ScheduleCapacity

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler import Scheduler

DEFAULT_PRINCIPAL = "sglang-default"


class SGLangRequest:
    """One `Req`, answering the questions `plex.engine` asks of a request."""

    __slots__ = ("_request", "_bytes_per_token")

    def __init__(self, request: Req, bytes_per_token: int) -> None:
        self._request = request
        self._bytes_per_token = bytes_per_token

    @property
    def engine_id(self) -> str:
        return self._request.rid

    def plex_config(self) -> dict[str, Any] | None:
        custom = self._request.sampling_params.custom_params or {}
        return custom.get("plex")

    def body(self) -> dict[str, Any]:
        request = self._request
        return {
            "prompt_token_ids": list(request.origin_input_ids),
            "max_tokens": request.sampling_params.max_new_tokens,
            "priority": request.priority,
        }

    def facts(self) -> dict[str, Any]:
        request = self._request
        entered = request.time_stats.wait_queue_entry_time
        if entered <= 0:
            entered = request.time_stats.scheduler_recv_time
        return {
            "attained_service": request.kv_committed_len,
            "generated_tokens": len(request.output_ids),
            "preempted": request.is_retracted,
            "waiting_ms": max(int((time.perf_counter() - entered) * 1000), 0),
            "cached_tokens": request.num_matched_prefix_tokens,
            "arrival_ms": self.arrival_ms(),
        }

    def arrival_ms(self) -> int:
        created = self._request.time_stats.created_time
        return int(created * 1000) if created > 0 else 0

    def token_budget(self) -> int:
        request = self._request
        return max(
            len(request.origin_input_ids)
            + len(request.output_ids)
            - request.num_matched_prefix_tokens,
            0,
        )

    def size_bytes(self) -> int:
        request = self._request
        allocated = (
            request.kv.kv_allocated_len
            if request.kv is not None
            else request.kv_committed_len
        )
        return max(allocated * self._bytes_per_token, self._bytes_per_token)

    def reload_cost(self) -> int:
        return self._request.kv_committed_len

    def finish_reason(self) -> Any:
        reason = self._request.finished_reason
        return reason.to_json() if reason is not None else None


class SGLangEnginePort:
    """The scheduler, seen through the contract's vocabulary."""

    name = "sglang"
    default_principal = DEFAULT_PRINCIPAL
    # SGLang admits one request per selection; batching them would risk a
    # partial admission, and a selection is all-or-none.
    max_requests_per_selection = 1

    def __init__(self, scheduler: Scheduler) -> None:
        self.scheduler = scheduler

    def view(self, request: Req) -> SGLangRequest:
        return SGLangRequest(request, self.bytes_per_token())

    def bytes_per_token(self) -> int:
        allocator = self.scheduler.token_to_kv_pool_allocator
        size = allocator.get_kvcache().get_kv_size_bytes()
        total = sum(size) if isinstance(size, tuple) else size
        return max(int(total / allocator.size_full), 1)

    def candidates(self) -> list[SGLangRequest]:
        bytes_per_token = self.bytes_per_token()
        return [
            SGLangRequest(request, bytes_per_token)
            for request in self.scheduler.waiting_queue
            if not request.finished()
        ]

    def residents(self) -> list[SGLangRequest]:
        bytes_per_token = self.bytes_per_token()
        return [
            SGLangRequest(request, bytes_per_token)
            for request in self.scheduler.running_batch.reqs
        ]

    def capacity(self) -> ScheduleCapacity:
        scheduler = self.scheduler
        selections = min(len(self.candidates()), scheduler.max_running_requests)
        return ScheduleCapacity(
            max_selections=selections,
            max_requests=selections,
            max_total_tokens=scheduler.max_prefill_tokens,
        )

    def cache_capacity(self, residents: list[SGLangRequest]) -> CacheCapacity:
        """Budget the resident total, so declining to retract stays valid.

        SGLang asks the policy for a retraction *ordering* and then retracts as
        much as it needs from the end of it, rather than enacting a fixed set.
        Budgeting less than the residents occupy would reject the ordering that
        keeps everything, which is a legitimate ranking.
        """
        return CacheCapacity(
            max_bytes=sum(max(request.size_bytes(), 1) for request in residents),
            fixed_bytes=0,
        )

    def under_pressure(self) -> bool:
        return True

    def engine_facts(self) -> dict[str, Any]:
        scheduler = self.scheduler
        allocator = scheduler.token_to_kv_pool_allocator
        return {
            "queue_depth": len(scheduler.waiting_queue),
            "running_requests": len(scheduler.running_batch.reqs),
            "free_kv_tokens": allocator.available_size(),
            "total_kv_tokens": allocator.size_full,
        }


class AsyncPlexPolicyController:
    """Scheduler-facing wrapper over `plex.engine.PolicyController`.

    Keeps the call sites in `scheduler.py` unchanged and adapts the two places
    SGLang's shape differs from the contract's: it retracts by reordering a
    batch rather than evicting named objects, and it reports finishes without
    a reason argument.
    """

    def __init__(
        self, controller: PolicyController, port: SGLangEnginePort
    ) -> None:
        self.controller = controller
        self.port = port

    @classmethod
    def from_policy(
        cls,
        policy: str,
        scheduler: Scheduler,
        *,
        model: str,
        target_id: str,
    ) -> AsyncPlexPolicyController:
        port = SGLangEnginePort(scheduler)
        return cls(
            PolicyController.from_policy(
                policy, port, model=model, target_id=target_id
            ),
            port,
        )

    def tracks(self, engine_id: str) -> bool:
        return self.controller.tracks(engine_id)

    def register_request(self, request: Req) -> None:
        self.controller.register_request(self.port.view(request))

    def mark_retracted(self, request: Req) -> None:
        self.controller.mark_preempted(self.port.view(request))

    def observe_batch(self, requests: list[Req]) -> None:
        for request in requests:
            if request.finished() and self.tracks(request.rid):
                self.mark_finished(request)

    def mark_finished(self, request: Req) -> None:
        view = self.port.view(request)
        self.controller.mark_finished(view, view.finish_reason())

    def publish(self, scheduler: Scheduler | None = None) -> None:
        self.controller.publish()

    def poll_schedule(self) -> Any:
        return self.controller.poll_schedule()

    def cached_retraction_order(self, requests: list[Req]) -> list[int] | None:
        """Order `requests` so the ones the policy would drop retract first.

        `retract_decode` retracts from the end, so victims go last, worst-ranked
        of them last of all.
        """
        victims = self.controller.cached_reclaim_victims(
            [request.rid for request in requests]
        )
        if victims is None:
            return None
        selected = set(victims)
        keepers = [index for index in range(len(requests)) if index not in selected]
        return [*keepers, *reversed(victims)]

    def close(self) -> None:
        self.controller.close()

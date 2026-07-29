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

from plex.engine import (
    NO_SIGNALS,
    CacheCapacity,
    PolicyController,
    RequestSignals,
    ScheduleCapacity,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler import Scheduler

DEFAULT_PRINCIPAL = "sglang-default"


class SGLangRequest:
    """One `Req`, answering the questions `plex.engine` asks of a request."""

    __slots__ = ("_bytes_per_token", "_request", "_signals")

    def __init__(
        self,
        request: Req,
        bytes_per_token: int,
        signals: RequestSignals | None = None,
    ) -> None:
        self._request = request
        self._bytes_per_token = bytes_per_token
        self._signals = signals if signals is not None else NO_SIGNALS

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
        waiting_ms = max(int((time.perf_counter() - entered) * 1000), 0)
        running = request.kv_committed_len > 0
        prompt_tokens = len(request.origin_input_ids)
        hit = min(request.num_matched_prefix_tokens, prompt_tokens)
        uncached = prompt_tokens - hit
        return {
            "attained_service": request.kv_committed_len,
            "service_tokens": request.kv_committed_len,
            "generated_tokens": len(request.output_ids),
            "preempted": request.is_retracted,
            "waiting_ms": waiting_ms,
            "call_wait_us": waiting_ms * 1000,
            "current_queue_ms": 0 if running else waiting_ms,
            # SGLang's own name for the prefix hit is
            # `num_matched_prefix_tokens`; `cached_tokens` means what is
            # resident, which is what it has actually committed.
            "cached_tokens": request.kv_committed_len,
            "lpm_hit_tokens": hit,
            "uncached_tokens": uncached,
            "new_prefill_tokens": max(
                uncached - max(request.kv_committed_len - hit, 0), 0
            ),
            "prefix_hit_ratio_ppm": (
                hit * 1_000_000 // prompt_tokens if prompt_tokens else 0
            ),
            # Whether this request's KV is resident, which covers both what it
            # has already committed and what it matched in the prefix cache.
            # Testing only the hit answered `false` for a long-running request
            # whose entire KV is resident but whose prompt shared no prefix.
            "cache_ready": hit > 0 or request.kv_committed_len > 0,
            "prompt_tokens": prompt_tokens,
            "computation_length": prompt_tokens + len(request.output_ids),
            "dispatch_input_tokens": max(
                prompt_tokens - request.kv_committed_len, 0
            ),
            "queue_member": not running,
            "scheduler_state": "running" if running else "waiting",
            "arrival_ms": self.arrival_ms(),
            "arrival_seq": self._signals.arrival_seq,
            "now_ms": int(time.time() * 1000),
        }

    def cache_facts(self) -> dict[str, Any]:
        request = self._request
        return {
            "cached_length": request.kv_committed_len,
            "computation_length": (
                len(request.origin_input_ids) + len(request.output_ids)
            ),
            "last_access_ms": self._signals.last_access_ms,
            "state_kind": "retracted" if request.is_retracted else "resident",
            "tier": "gpu",
            "leaf": True,
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
        self._signals: dict[str, RequestSignals] = {}
        self._arrivals = 0
        self._probed: set[str] = set()
        self._probed_tokens = 0
        self._hit_tokens = 0

    def observe(self, request: Req) -> RequestSignals:
        """Record arrival order; SGLang matches the prefix itself, later."""
        self._arrivals += 1
        signals = RequestSignals(
            arrival_seq=self._arrivals - 1,
            lpm_hit_tokens=request.num_matched_prefix_tokens,
        )
        self._signals[request.rid] = signals
        return signals

    def touch(self, request: Req) -> None:
        signals = self._signals.get(request.rid)
        if signals is None:
            return
        signals.touch()
        # The match happens when SGLang admits the request, after the policy
        # was asked; carry it forward so later decisions see the real hit.
        #
        # The prompt enters the engine-wide denominator on first admission and
        # once only. Counting it inside the "the hit grew" branch below counted
        # only the requests that *had* a hit, so `hit_ratio_ppm` was a hit rate
        # over hits: it could never fall, and a policy thresholding on it saw a
        # cache that always looked warm. Found by replaying one situation
        # through both bindings, where vLLM -- which probes every arrival --
        # reported 300,000 ppm against SGLang's 0 on the same five requests.
        if request.rid not in self._probed:
            self._probed.add(request.rid)
            self._probed_tokens += len(request.origin_input_ids)
        if request.num_matched_prefix_tokens > signals.lpm_hit_tokens:
            self._hit_tokens += (
                request.num_matched_prefix_tokens - signals.lpm_hit_tokens
            )
            signals.lpm_hit_tokens = request.num_matched_prefix_tokens

    def forget(self, request_id: str) -> None:
        self._signals.pop(request_id, None)
        self._probed.discard(request_id)

    def view(self, request: Req) -> SGLangRequest:
        return SGLangRequest(
            request, self.bytes_per_token(), self._signals.get(request.rid)
        )

    def bytes_per_token(self) -> int:
        allocator = self.scheduler.token_to_kv_pool_allocator
        size = allocator.get_kvcache().get_kv_size_bytes()
        total = sum(size) if isinstance(size, tuple) else size
        return max(int(total / allocator.size_full), 1)

    def candidates(self) -> list[SGLangRequest]:
        bytes_per_token = self.bytes_per_token()
        return [
            SGLangRequest(request, bytes_per_token, self._signals.get(request.rid))
            for request in self.scheduler.waiting_queue
            if not request.finished()
        ]

    def residents(self) -> list[SGLangRequest]:
        bytes_per_token = self.bytes_per_token()
        return [
            SGLangRequest(request, bytes_per_token, self._signals.get(request.rid))
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
        free_tokens = allocator.available_size()
        total_tokens = allocator.size_full
        per_token = self.bytes_per_token()
        running = len(scheduler.running_batch.reqs)
        return {
            "queue_depth": len(scheduler.waiting_queue),
            "running_requests": running,
            "batch_size": running,
            "max_batch_size": scheduler.max_running_requests,
            "max_total_tokens": scheduler.max_prefill_tokens,
            "free_kv_tokens": free_tokens,
            "total_kv_tokens": total_tokens,
            "used_kv_ppm": (
                (total_tokens - free_tokens) * 1_000_000 // total_tokens
                if total_tokens
                else 0
            ),
            "memory_capacity": total_tokens * per_token,
            "active_kv_bytes": (total_tokens - free_tokens) * per_token,
            "hit_ratio_ppm": (
                self._hit_tokens * 1_000_000 // self._probed_tokens
                if self._probed_tokens
                else 0
            ),
            "kv_overloaded": total_tokens > 0 and free_tokens * 2 <= total_tokens,
            "now_ms": int(time.time() * 1000),
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
        self.port.observe(request)
        self.controller.register_request(self.port.view(request))

    def mark_retracted(self, request: Req) -> None:
        self.controller.mark_preempted(self.port.view(request))

    def observe_batch(self, requests: list[Req]) -> None:
        for request in requests:
            self.port.touch(request)
            if request.finished() and self.tracks(request.rid):
                self.mark_finished(request)

    def mark_finished(self, request: Req) -> None:
        view = self.port.view(request)
        self.controller.mark_finished(view, view.finish_reason())
        self.port.forget(request.rid)

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

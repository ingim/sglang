"""PLEX v2 observer: the SGLang scheduler, in the contract's vocabulary.

Stage ① of the phased attach in `.wiki/v2/plan.md`: events and facts flowing,
zero influence. The engine schedules exactly as it does now and a policy
watches.

## Two hooks, and one of them is a hook only because arrival order is not
## recoverable afterwards

vLLM has a single funnel every terminal edge passes through (`_free_request`),
so the observer there hooks arrival, terminal and step. SGLang does not:
requests finish through `process_batch_result_decode`,
`process_batch_result_prefill`, the disaggregation paths and the dLLM path,
and hooking each is four call sites that can drift apart as upstream adds a
fifth.

So this derives the terminal edge instead. The observer already walks the
waiting queue and the running batch every step; a request that was there last
step and is not there now has left. One hook, no possibility of missing an
edge upstream adds, and the cost is that the *reason* is unknown — which is
honest, because at the step boundary it is.

Arrival cannot be derived the same way. A request appearing in the queue tells
you it arrived, but not in what order relative to one that arrived and was
scheduled within the same step, and a fairness policy that cannot order
arrivals has no fairness to enforce. Hence the second hook.

## Why this stays small

The facts are derived by `plex-port-vllm`'s sibling on the Rust side, not
here. v0.7's SGLang integration computed them in-engine; v2's shim reads what
SGLang already holds and hands it over. Every entry below is an attribute read
or a subtraction of two — deriving the same quantity in two places is how two
descriptions of one engine start to disagree.

## What attaching this can change

Nothing. There is no branch on policy output, because stage ① has none. Run
the engine with and without it and the outputs must be identical.
"""

from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler import Scheduler


class PlexObserver:
    """One SGLang scheduler, in the contract's vocabulary. Holds no policy."""

    def __init__(self, scheduler: Scheduler, sink: Any, target: str = "sglang-0") -> None:
        self._scheduler = scheduler
        self._sink = sink
        self._target = target
        self._step = 0
        # Arrival order, which nothing in SGLang records.
        self._arrival_seq: dict[str, int] = {}
        self._arrival_ms: dict[str, int] = {}
        # How many eviction candidates to offer per step. Bounded for the
        # same reason as on vLLM: the cost of describing pages must stay
        # proportional to how many the engine is about to touch.
        self._page_budget = int(os.environ.get("SGLANG_PLEX_PAGE_BUDGET", "64"))
        # A short window of observed step durations, for the timing
        # facts. Bounded so it tracks the engine's current behaviour
        # rather than averaging over a run whose shape has changed.
        self._step_ms_window: list[int] = []
        self._last_step_ms: int | None = None
        # Bytes of KV per token. Declared rather than guessed: it depends
        # on the model's layers, heads and dtype, and a port that assumed
        # one would publish a confident byte count for a different model.
        self._bytes_per_token = int(
            os.environ.get("SGLANG_PLEX_BYTES_PER_TOKEN", "0")
        )
        self._offered_last: set[str] = set()
        self._arrivals = 0
        # Who was here last step, for deriving terminal edges.
        self._present: set[str] = set()
        # Requests the engine has marked finished but which are still
        # tracked. Departure is `absent AND finished`, never absent alone.
        self._finishing: set[str] = set()
        # Requests that vanished without finishing: retracted, not gone.
        self._retracted: list[str] = []
        # Every request ever retracted, for `preempted`. Separate from
        # `_retracted` (this step's) because the fact is "has been", not
        # "is being".
        self._retracted_ever: set[str] = set()
        self._admitted: list[str] = []
        # Stage 3, if a source was named.
        from sglang.srt.managers.plex_verbs import PlexVerbs

        self._verbs = PlexVerbs.maybe(scheduler)

    # ── the hooks ────────────────────────────────────────────────────────

    def on_request_queued(self, req: Req) -> None:
        """A request entered the waiting queue."""
        rid = req.rid
        if rid not in self._arrival_seq:
            self._arrival_seq[rid] = self._arrivals
            # SGLang records no wall-clock arrival the observer can read
            # at this point, so it is taken here. That makes `arrival-ms`
            # the moment the queue saw the request rather than the moment
            # the server did — a difference of the request's own parsing,
            # and one worth stating rather than letting a policy assume
            # otherwise.
            self._arrival_ms[rid] = int(time.time() * 1000)
            self._arrivals += 1
        self._admitted.append(rid)

    def emit_step(self) -> None:
        """Write one step document, and never fail the engine.

        A port that can raise inside the scheduler is a port that can take
        the engine down, and stage ① is supposed to be unable to change
        anything — including whether the step completes. So the write is
        guarded and a failure disables the observer: losing observation is
        a measurement problem, losing the step is an outage.
        """
        try:
            self._sink.write(self.on_step() + "\n")
            self._sink.flush()
        except Exception:  # noqa: BLE001 - see docstring
            self._sink = None
            self._scheduler.plex_observer = None

    def drain_verbs(self) -> int:
        """Enact staged verbs. Called before a scheduling pass begins.

        Separate from the step document for the reason vLLM's port
        learned by crashing twice: a read may happen mid-pass and a write
        may not. SGLang's loop is structured differently, but `finish`
        mutates the same queues a pass consumes, so the rule holds.
        """
        if self._verbs is None:
            return 0
        return self._verbs.drain()

    def on_step(self) -> str:
        """One scheduler step, as a document the port reads."""
        self._step += 1
        tracked = self._tracked()
        present = {req.rid for req in tracked}

        # Terminal edges.
        #
        # Absence is *not* departure. A real run refuted that: SGLang
        # retracts a request under memory pressure and puts it back, so
        # one request vanished at step 80, returned at step 81, and a
        # set-difference derivation reported it as finishing twice.
        # Every accumulator in the corpus counts one departure as one.
        #
        # So absence is a *candidate*, and `req.finished()` — the
        # engine's own answer — is what confirms it. A request that
        # disappears without having finished has been retracted, and is
        # remembered rather than mourned.
        for req in tracked:
            if getattr(req, "finished", None) is not None and req.finished():
                self._finishing.add(req.rid)

        gone = self._present - present
        departed = sorted(rid for rid in gone if rid in self._finishing)
        self._retracted = sorted(rid for rid in gone if rid not in self._finishing)
        self._retracted_ever.update(self._retracted)
        self._present = present
        for rid in departed:
            self._arrival_seq.pop(rid, None)
            self._arrival_ms.pop(rid, None)
            self._finishing.discard(rid)

        document_now_ms = int(time.time() * 1000)
        if self._last_step_ms is not None:
            self._step_ms_window.append(max(document_now_ms - self._last_step_ms, 0))
            if len(self._step_ms_window) > 32:
                self._step_ms_window.pop(0)
        self._last_step_ms = document_now_ms
        document = {
            "step": self._step,
            "now-ms": document_now_ms,
            "target": self._target,
            "subjects": self._subjects(tracked),
            "facts": self._facts(tracked, document_now_ms),
            "events": self._events(departed),
        }
        self._admitted.clear()
        return json.dumps(document)

    # ── scraping ─────────────────────────────────────────────────────────

    def _held_requests(self) -> list[Req]:
        """Requests the gate is holding, which the scheduler cannot see.

        They are out of `waiting_queue` on purpose — that is what the
        hold is — so the observer has to reach the holder to describe
        them. Without this the gate would be offered no subject, which is
        exactly how `llumnix` came to be called and decide nothing.
        """
        policy = getattr(self._scheduler, "policy", None)
        schedule = getattr(policy, "plex_schedule", None)
        held = getattr(schedule, "held_requests", None)
        return list(held()) if held is not None else []

    def _tracked(self) -> list[Req]:
        scheduler = self._scheduler
        running = list(getattr(scheduler.running_batch, "reqs", []) or [])
        return [*self._held_requests(), *scheduler.waiting_queue, *running]

    def _facts(self, tracked: list[Req], now_ms: int) -> dict[str, dict[str, Any]]:
        running_ids = {
            req.rid for req in getattr(self._scheduler.running_batch, "reqs", []) or []
        }
        held_ids = {req.rid for req in self._held_requests()}
        facts: dict[str, dict[str, Any]] = {}
        for req in tracked:
            running = req.rid in running_ids
            prompt = len(req.origin_input_ids)
            generated = len(req.output_ids)
            # SGLang's own total cached prefix length: on-device
            # `prefix_indices` plus any host hit, capped at the max allowed
            # prefix. It is the radix tree's answer, which is what makes
            # pages content-addressed here in the first place — and it is
            # the field SGLang itself uses to estimate uncached tokens,
            # rather than `len(prefix_indices)`, which misses the host half.
            cached = int(getattr(req, "num_matched_prefix_tokens", 0) or 0)
            facts[req.rid] = {
                # `pending` only while genuinely held out of the queue.
                # Publishing it for a queued request would be a lie: the
                # engine has already accepted that one, and a policy
                # answering `reject` would rule on something admitted.
                "state": {
                    "text": "pending"
                    if req.rid in held_ids
                    else ("active" if running else "admitted")
                },
                "arrival_seq": {"num": self._arrival_seq.get(req.rid, 0)},
                "arrival_ms": {"num": self._arrival_ms.get(req.rid, now_ms)},
                # SGLang's retraction is its preemption: a request put
                # back under memory pressure has been preempted, whatever
                # the engine calls it. Derived from the observer's own
                # record because SGLang keeps no counter — and stating it
                # is better than leaving a policy to infer it from a
                # request that mysteriously restarted.
                "preempted": {"flag": req.rid in self._retracted_ever},
                "prompt_tokens": {"num": prompt},
                "generated_tokens": {"num": generated},
                "computation_length": {"num": prompt + generated},
                "dispatch_input_tokens": {"num": max(prompt - cached, 0)},
                "cached_tokens": {"num": cached},
                "queue_member": {"flag": not running and req.rid not in held_ids},
                # Prefix-cache facts. vLLM cannot publish these from an
                # observer — it computes hits inside `schedule()` and
                # never keeps them on the request — but SGLang's radix
                # tree answers per request, so where the engine knows it,
                # the observer says so.
                "uncached_tokens": {"num": max(prompt - cached, 0)},
                "lpm_hit_tokens": {"num": cached},
                "prefix_hit_ratio_ppm": {
                    "num": min(cached * 1_000_000 // prompt, 1_000_000)
                    if prompt
                    else 0
                },
                "waiting_ms": {"num": max(now_ms - self._arrival_ms.get(req.rid, now_ms), 0)},
                "current_queue_ms": {
                    "num": 0
                    if running
                    else max(now_ms - self._arrival_ms.get(req.rid, now_ms), 0)
                },
            }
        for page_id, node in self._offered_pages():
            value = getattr(node, "value", None)
            facts[page_id] = {
                "resident": {"flag": True},
                "targets": {"ids": [self._target]},
                "tier": {"text": "gpu"},
                # `lock_ref` is the tree's own "someone is using this".
                "pinned": {"flag": int(getattr(node, "lock_ref", 0) or 0) > 0},
                "page-tokens": {"num": len(value) if value is not None else 0},
                "leaf": {"flag": True},
            }
        facts[self._target] = self._target_facts()
        return facts


    def _step_timing(self) -> tuple[int, int]:
        """Median step wall clock, and the decode cost it implies.

        **Measured here, not read from the engine.** Neither engine keeps
        a per-step duration an observer can read — `forward_ct` is a
        count — and the honest options were to publish nothing or to
        measure. Publishing nothing loses four names the corpus reads;
        inventing a plausible constant would hand a policy a confident
        number about a machine nobody timed.

        So the observer times its own steps and says so. It is the
        *observer's* view of the step, which includes anything else the
        engine did between two documents — and that is the quantity a
        policy reasoning about "how long until my turn" actually wants.

        Median over a short window rather than a mean: one slow step
        (a cold kernel, a neighbour on the GPU) would drag a mean for
        many steps afterwards, and a policy acting on it would be acting
        on an outlier that has already passed.
        """
        if len(self._step_ms_window) < 3:
            return (0, 0)
        window = sorted(self._step_ms_window)
        median = window[len(window) // 2]
        return (median, median * 1000)

    def _target_facts(self) -> dict[str, Any]:
        scheduler = self._scheduler
        running = getattr(scheduler.running_batch, "reqs", []) or []
        total = int(scheduler.max_total_num_tokens)
        free = int(scheduler.token_to_kv_pool_allocator.available_size())
        max_running = int(scheduler.max_running_requests)
        queued = len(scheduler.waiting_queue)
        step_ms, step_us = self._step_timing()
        pending_decode = sum(
            max(int(getattr(getattr(req, "sampling_params", None),
                            "max_new_tokens", 0) or 0)
                - len(req.output_ids), 0)
            for req in running
        )
        queued_tokens = sum(
            len(req.origin_input_ids) for req in scheduler.waiting_queue
        )
        decoding = sum(1 for req in running if req.output_ids)
        return {
            "queue_depth": {"num": len(scheduler.waiting_queue)},
            "running_requests": {"num": len(running)},
            "batch_size": {"num": len(running)},
            "decode_batch_size": {"num": len(running)},
            "max_batch_size": {"num": max_running},
            # An alias the corpus reads under a second name.
            "max_requests": {"num": max_running},
            "free_decode_slots": {"num": max(max_running - len(running), 0)},
            "pending_decode_tokens": {"num": pending_decode},
            "queued_tokens": {"num": queued_tokens},
            "decoder_ratio_ppm": {
                "num": min(decoding * 1_000_000 // len(running), 1_000_000)
                if running
                else 0
            },
            "kv_overloaded": {"flag": free < total / 10},
            "step_ms": {"num": step_ms},
            # Time per output token, from the observer's own timing. The
            # engine keeps no such number, so this is measured rather
            # than read — and it is the observer's view of a step, which
            # is what a policy asking "how long until my turn" wants.
            "decode_ms_per_token": {"num": step_ms},
            "tpot_us": {"num": step_us},
            # How long a newcomer waits: the requests ahead of it,
            # divided by how many run at once, times a step.
            "estimated_wait_ms": {
                "num": ((queued + max_running - 1) // max_running) * step_ms if max_running else 0
            },
            # The pool in bytes as well as tokens. SGLang's allocator is
            # denominated in tokens, so the conversion needs a per-token
            # size the engine holds and the port does not.
            "memory_capacity": {"num": total * self._bytes_per_token},
            "active_kv_bytes": {"num": (total - free) * self._bytes_per_token},
            "tiers": {"ids": ["gpu"]},
            "throughput_token_cap": {
                "num": int(getattr(scheduler, "max_prefill_tokens", 0) or 0)
            },
            "used_kv_ppm": {
                "num": min((total - free) * 1_000_000 // total, 1_000_000)
                if total
                else 0
            },
            # SGLang's allocator is already denominated in tokens, so
            # unlike vLLM there is no block-size conversion to do here.
            "total_kv_tokens": {"num": total},
            "free_kv_tokens": {"num": free},
            "max_total_tokens": {"num": total},
        }

    def _subjects(self, tracked: list[Req]) -> dict[str, Any]:
        subjects: dict[str, Any] = {
            "request": [req.rid for req in tracked],
            "target": [self._target],
        }
        pages = self._offered_pages()
        if pages:
            subjects["page"] = [page_id for page_id, _ in pages]
        return subjects

    def _offered_pages(self) -> list[tuple[str, Any]]:
        """Radix nodes that are candidates for eviction, coldest first.

        **Offered, not enumerated**, exactly as on vLLM. A snapshot of
        the whole tree is per-node state on every step, which is a
        different order of cost from the per-request scrape, and it hands
        a policy thousands of subjects when a handful are in play.

        `evictable_leaves` ordered by the cache's own
        `eviction_strategy` is SGLang's answer to "what goes next", so
        reading it is reading the answer rather than modelling it.

        Only leaves, and that is the tree's own rule rather than a
        simplification: evicting an interior node orphans everything
        below it, so a policy that could name one could express a plan
        the engine must refuse.
        """
        cache = getattr(self._scheduler, "tree_cache", None)
        leaves = getattr(cache, "evictable_leaves", None)
        if not leaves:
            return []
        strategy = getattr(cache, "eviction_strategy", None)
        nodes = list(leaves)
        if strategy is not None and hasattr(strategy, "get_priority"):
            try:
                nodes.sort(key=strategy.get_priority)
            except (TypeError, ValueError):
                pass
        offered: list[tuple[str, Any]] = []
        for node in nodes[: self._page_budget]:
            node_id = getattr(node, "id", None)
            if node_id is None:
                continue
            offered.append((f"p{int(node_id):08x}", node))
        return offered

    def _events(self, departed: list[str]) -> dict[str, Any]:
        events: dict[str, Any] = {}
        offered = [page_id for page_id, _ in self._offered_pages()]
        # Once per page, not once per step: every accumulator in the
        # corpus counts one offer as one.
        fresh = [page_id for page_id in offered if page_id not in self._offered_last]
        if fresh:
            events["offered"] = fresh
        self._offered_last = set(offered)
        if self._admitted:
            events["admitted"] = list(self._admitted)
        if departed:
            # `completed` is the outcome the contract has for "it ended and
            # nothing said otherwise". The initiator is `host` because the
            # engine ended it: a policy-initiated finish is stage ③ and
            # does not exist yet, and recording one now would attribute the
            # engine's decisions to a component that made none.
            events["finished"] = [[rid, "completed", "host"] for rid in departed]
        return events


def maybe_plex_observer(scheduler: Scheduler) -> PlexObserver | None:
    """Attach an observer if one was asked for, and otherwise cost nothing.

    Environment rather than `server_args`, deliberately. Stage ① changes no
    behaviour, so it needs no place in an argument schema a user has to
    understand — and a feature that is off by default and invisible when off
    is the easiest kind to review.

        SGLANG_PLEX_OBSERVE=/path/to/steps.jsonl

    A path that cannot be opened disables the observer rather than failing
    startup, for the same reason `emit_step` swallows.
    """
    path = os.environ.get("SGLANG_PLEX_OBSERVE")
    if not path:
        return None
    try:
        sink = open(path, "a", encoding="utf-8")  # noqa: SIM115 - lives with the scheduler
    except OSError:
        return None
    target = os.environ.get("SGLANG_PLEX_TARGET", "sglang-0")
    return PlexObserver(scheduler, sink, target)

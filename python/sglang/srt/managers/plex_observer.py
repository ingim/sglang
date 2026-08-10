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
        self._arrivals = 0
        # Who was here last step, for deriving terminal edges.
        self._present: set[str] = set()
        # Requests the engine has marked finished but which are still
        # tracked. Departure is `absent AND finished`, never absent alone.
        self._finishing: set[str] = set()
        # Requests that vanished without finishing: retracted, not gone.
        self._retracted: list[str] = []
        self._admitted: list[str] = []

    # ── the hooks ────────────────────────────────────────────────────────

    def on_request_queued(self, req: Req) -> None:
        """A request entered the waiting queue."""
        rid = req.rid
        if rid not in self._arrival_seq:
            self._arrival_seq[rid] = self._arrivals
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
        self._present = present
        for rid in departed:
            self._arrival_seq.pop(rid, None)
            self._finishing.discard(rid)

        document = {
            "step": self._step,
            "now-ms": int(time.time() * 1000),
            "target": self._target,
            "subjects": {
                "request": [req.rid for req in tracked],
                "target": [self._target],
            },
            "facts": self._facts(tracked),
            "events": self._events(departed),
        }
        self._admitted.clear()
        return json.dumps(document)

    # ── scraping ─────────────────────────────────────────────────────────

    def _tracked(self) -> list[Req]:
        scheduler = self._scheduler
        running = list(getattr(scheduler.running_batch, "reqs", []) or [])
        return [*scheduler.waiting_queue, *running]

    def _facts(self, tracked: list[Req]) -> dict[str, dict[str, Any]]:
        running_ids = {
            req.rid for req in getattr(self._scheduler.running_batch, "reqs", []) or []
        }
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
                "state": {"text": "active" if running else "admitted"},
                "arrival_seq": {"num": self._arrival_seq.get(req.rid, 0)},
                "prompt_tokens": {"num": prompt},
                "generated_tokens": {"num": generated},
                "computation_length": {"num": prompt + generated},
                "dispatch_input_tokens": {"num": max(prompt - cached, 0)},
                "cached_tokens": {"num": cached},
                "queue_member": {"flag": not running},
            }
        facts[self._target] = self._target_facts()
        return facts

    def _target_facts(self) -> dict[str, Any]:
        scheduler = self._scheduler
        running = getattr(scheduler.running_batch, "reqs", []) or []
        total = int(scheduler.max_total_num_tokens)
        free = int(scheduler.token_to_kv_pool_allocator.available_size())
        return {
            "queue_depth": {"num": len(scheduler.waiting_queue)},
            "running_requests": {"num": len(running)},
            "batch_size": {"num": len(running)},
            "max_batch_size": {"num": scheduler.max_running_requests},
            # SGLang's allocator is already denominated in tokens, so
            # unlike vLLM there is no block-size conversion to do here.
            "total_kv_tokens": {"num": total},
            "free_kv_tokens": {"num": free},
            "max_total_tokens": {"num": total},
        }

    def _events(self, departed: list[str]) -> dict[str, Any]:
        events: dict[str, Any] = {}
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

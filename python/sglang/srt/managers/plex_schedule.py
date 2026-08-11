"""PLEX v2 stage-2 attach: the policy's order, applied to SGLang's queue.

Stage 2 of the phased attach: the standing tables. A policy installs an
order over waiting requests and the engine follows it.

## Why this is a sort key and not a patch to the scheduler

SGLang decides order in one place — `SchedulePolicy.calc_priority`, which sorts
`waiting_queue` in place — and every built-in policy is a call to
`waiting_queue.sort(key=...)`. A standing schedule table is another such
policy, so it costs a sort rather than a rewrite.

That is a different shape from vLLM, where the same stage is a `RequestQueue`
subclass. Both engines already had the extension point; neither needed a new
one. Where they differ is that vLLM's queue is an object with an interface and
SGLang's is a list the policy sorts, so the attach follows each engine's own
grain instead of imposing one.

## What "follow the table" means

The same three claims as the vLLM side, because they are claims about the
contract rather than about an engine:

- A request the table names is ordered where the table puts it.
- A request the table does not name sorts after every named one, in arrival
  order among themselves. Unnamed is "the policy expressed no view", not "the
  policy ranked these last".
- A table naming a request that has since left is ignored for that request.
  Tables are standing documents and the world moves under them, so `table_age`
  counts arrivals since the install.

## What this cannot do

Order what is waiting, and nothing else. Retraction and abort are verbs and
belong to stage 3. A policy whose table implies a running request should yield
gets no such effect here — the alternative is a port that half-enacts and
reports success.
"""

from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class PlexSchedule:
    """A standing order over waiting requests."""

    def __init__(self, source: str | None = None) -> None:
        # request id -> rank. Replaced wholesale on install: a table is a
        # document, not a stream of edits, so a partially-applied one is
        # not representable.
        self._rank: dict[str, int] = {}
        # Where the policy's table arrives from. A path rather than a
        # callback because the policy runs in the PLEX host, not in the
        # engine: the engine must not import a runtime, block on one, or
        # be able to fail because one is slow.
        self._source = source if source else os.environ.get("SGLANG_PLEX_TABLE")
        self._source_stamp: tuple[int, int] | None = None

        # ── admission hold (the route channel) ───────────────────────────
        #
        # SGLang appends a request straight onto `waiting_queue`, so there
        # is no moment at which one has arrived and not yet been admitted
        # — and a gate with nothing pending has nothing to rule on.
        # Holding arrivals creates that moment. It is a behaviour change
        # and therefore stage 2, exactly as on vLLM.
        self._gate = os.environ.get("SGLANG_PLEX_GATE")
        self._held: dict[str, tuple[Req, float]] = {}
        self._verdict_stamp: tuple[int, int] | None = None
        # The declared default: admit. A policy that does not answer must
        # not change what the engine would have done, so silence and
        # slowness produce the same behaviour.
        #
        # SGLang needs no equivalent of vLLM's `has_requests` clause: its
        # event loop is a `while True` that receives and steps regardless
        # of whether there is work, so the deadline advances even with
        # every arrival held. vLLM quiesces, which is why holding there
        # hung a real engine until the loop was told that a held request
        # is pending work.
        self._hold_ms = float(os.environ.get("SGLANG_PLEX_GATE_MS", "50"))
        self.released_by_deadline = 0
        self.rejected_by_policy: list[str] = []
        self._installs = 0
        self._seen: set[str] = set()
        # Requests the gate has already ruled on, so a released request
        # is not immediately re-held.
        self._ruled: set[str] = set()
        self._to_release: list[tuple[str, Req]] = []
        self._arrivals = 0
        self._installed_at_arrival = 0

    def install(self, order: list[str]) -> None:
        """Replace the standing order."""
        self._rank = {rid: rank for rank, rid in enumerate(order)}
        self._installs += 1
        self._installed_at_arrival = self._arrivals

    def reload(self) -> None:
        """Pick up a newly written table, if there is one.

        Called from `calc_priority`, which is the moment the scheduler
        asks for an order and is therefore provably not part-way through
        consuming one. vLLM's port learned this the expensive way: a
        reload inside `pop_request` re-sorted between the scheduler's
        peek and its pop and crashed the engine with a `KeyError`. SGLang
        sorts in one call, so the safe point is obvious here — but it is
        the same rule, and it is the contract's own: a document must not
        move under the thing reading it.

        A table that is missing, unreadable or malformed leaves the
        current one standing.
        """
        if not self._source:
            return
        try:
            stat = os.stat(self._source)
            stamp = (stat.st_mtime_ns, stat.st_size)
            if stamp == self._source_stamp:
                return
            with open(self._source, encoding="utf-8") as handle:
                order = json.load(handle)
            if not isinstance(order, list):
                return
            self._source_stamp = stamp
        except (OSError, ValueError):
            return
        self.install([str(entry) for entry in order])

    @property
    def installs(self) -> int:
        return self._installs

    @property
    def table_age(self) -> int:
        """Arrivals since the current table was installed.

        A plan that ruled on a queue that no longer exists is a real v0.7
        finding: one arm's plan was 222 arrivals old and nothing could see
        it. Age travels with the decision so a stale table is visible
        rather than silently obeyed.
        """
        return self._arrivals - self._installed_at_arrival

    def hold_arrivals(self, waiting_queue: list[Req]) -> None:
        """Move newly-arrived requests into the hold, and release rulings.

        Called from `calc_priority`, which is where the scheduler hands
        over the queue and is therefore the one point per pass that can
        change what is in it without racing the scheduler's own
        iteration. vLLM's port established that rule the expensive way.
        """
        if not self._gate:
            return
        self._pick_up_verdicts()

        now = time.monotonic()
        # Anything not yet ruled on goes into the hold and out of the
        # queue. `_seen` is the record of what the gate has been offered,
        # so a released request is not re-held on the next pass.
        for req in list(waiting_queue):
            if req.rid in self._held or req.rid in self._ruled:
                continue
            self._held[req.rid] = (req, now)
            waiting_queue.remove(req)

        # Deadline: admit anything held too long.
        expired = [
            rid for rid, (_, since) in self._held.items()
            if (now - since) * 1000.0 >= self._hold_ms
        ]
        for rid in expired:
            req, _ = self._held.pop(rid)
            self._ruled.add(rid)
            waiting_queue.append(req)
            self.released_by_deadline += 1

        for rid, req in self._to_release:
            waiting_queue.append(req)
        self._to_release.clear()

    def held_requests(self) -> list[Req]:
        """Requests awaiting a verdict — the contract's `pending`."""
        return [req for req, _ in self._held.values()]

    def _pick_up_verdicts(self) -> None:
        """Apply verdicts the policy has written.

        `{"request-id": "assign"|"defer"|"reject"}`. A verdict naming a
        request this engine does not hold is ignored: a policy ruling on
        someone else's world has stale beliefs, and acting on it would
        make those beliefs true.
        """
        if not self._gate:
            return
        try:
            stat = os.stat(self._gate)
            stamp = (stat.st_mtime_ns, stat.st_size)
            if stamp == self._verdict_stamp:
                return
            with open(self._gate, encoding="utf-8") as handle:
                verdicts = json.load(handle)
            if not isinstance(verdicts, dict):
                return
            self._verdict_stamp = stamp
        except (OSError, ValueError):
            return

        for rid, verdict in verdicts.items():
            entry = self._held.get(str(rid))
            if entry is None:
                continue
            req, _ = entry
            if verdict == "assign":
                del self._held[str(rid)]
                self._ruled.add(str(rid))
                self._to_release.append((str(rid), req))
            elif verdict == "reject":
                # Released *and* recorded, not dropped. A refusal is a
                # kind of ending, not a kind of forgetting: a request
                # that vanishes without an outcome leaves its caller
                # waiting forever, which vLLM's port learned by hanging.
                del self._held[str(rid)]
                self._ruled.add(str(rid))
                self._to_release.append((str(rid), req))
                self.rejected_by_policy.append(str(rid))
            # `defer` keeps it held, which is what `defer` means.

    def apply(self, waiting_queue: list[Req]) -> None:
        """Sort the queue by the table, in place.

        In place because that is what `calc_priority` does and what every
        caller downstream expects; returning a new list would leave the
        scheduler holding the old one.

        Linear, not quadratic. The obvious way to keep unnamed requests in
        their existing order is `queue.index(req)` inside the key, which
        is O(n) per comparison and turns a sort into O(n^2 log n) —
        v0.7's post-mortem records a waiting scan that was O(queue) per
        step and "cost four arms a measurement". Enumerating once into a
        dict costs one pass.
        """
        positions = {}
        for index, req in enumerate(waiting_queue):
            positions[req.rid] = index
            if req.rid not in self._seen:
                self._seen.add(req.rid)
                self._arrivals += 1
        waiting_queue.sort(key=lambda req: self._sort_key(req, positions))

    def _sort_key(self, req: Req, positions: dict[str, int]) -> tuple[int, int, int]:
        rank = self._rank.get(req.rid)
        if rank is None:
            # Unnamed requests keep the order they already had, which for a
            # queue SGLang has not otherwise sorted is arrival order. The
            # current index rather than a timestamp, because `Req` has no
            # arrival field the scheduler maintains and inventing one here
            # would be a second clock.
            return (1, 0, positions[req.rid])
        return (0, rank, 0)


def maybe_plex_schedule() -> PlexSchedule | None:
    """Attach a standing table if one was asked for.

    Environment rather than `server_args`, matching the observer: a stage
    that is off by default and invisible when off is the easiest kind to
    review.

        SGLANG_PLEX_SCHEDULE=1
    """
    if not os.environ.get("SGLANG_PLEX_SCHEDULE"):
        return None
    return PlexSchedule()

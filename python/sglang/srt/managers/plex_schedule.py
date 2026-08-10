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

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class PlexSchedule:
    """A standing order over waiting requests."""

    def __init__(self) -> None:
        # request id -> rank. Replaced wholesale on install: a table is a
        # document, not a stream of edits, so a partially-applied one is
        # not representable.
        self._rank: dict[str, int] = {}
        self._installs = 0
        self._seen: set[str] = set()
        self._arrivals = 0
        self._installed_at_arrival = 0

    def install(self, order: list[str]) -> None:
        """Replace the standing order."""
        self._rank = {rid: rank for rank, rid in enumerate(order)}
        self._installs += 1
        self._installed_at_arrival = self._arrivals

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

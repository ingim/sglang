"""The cache channel, enacted: a policy's eviction order, honoured.

The SGLang counterpart of vLLM's `plex_cache.py`, and the same decision
seen through a different data structure.

## Where the decision is

`RadixCache.evict` builds a min-heap of `(priority, node)` over
`evictable_leaves`, where the priority comes from the cache's own
`eviction_strategy`, and pops until it has freed enough tokens. That
heap *is* the eviction order.

So this does not free anything, walk the tree, or decide what is
evictable. It supplies a priority for the nodes a policy named, and
leaves every other node's priority exactly as the strategy computed it.
A policy that names three leaves has expressed a preference about three
leaves.

## Leaves only, which is the tree's rule

Evicting an interior node orphans everything below it, so stage ① offers
only leaves and this can only reorder what was offered. A policy that
could name an interior node could express a plan the engine must refuse,
and refusing plans is worse than not being able to state them.

The heap re-pushes a parent when its last child goes, which is how a
whole branch is reclaimed. Those re-pushed parents keep their natural
priority: they were not leaves when the policy was asked, so the policy
said nothing about them.

## What a wrong order can do

Evict a prefix that would have been reused. That is the decision under
measurement and the worst case is a cache miss, never a correctness
fault -- the nodes are already unlocked and already evictable, and the
freeing is the engine's own.

## If the bridge is silent

Nothing changes and the strategy decides alone. Stage ①'s principle
carries forward: a policy that does not answer must not change what the
engine would have done.

Enabled by `SGLANG_PLEX_EVICT=/path/to/evict.jsonl`, one installed order
per line, most recent wins.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from sglang.srt.managers.plex_observer import _page_id

# Below every priority the strategy can produce, and ordered among
# themselves so the policy's own ranking survives.
#
# A sentinel rather than a subtraction from the natural priority: the
# strategies return timestamps, hit counts and ratios, so there is no
# scale a "large enough" offset could be expressed in. Anything the
# policy named must sort before anything it did not, and the arithmetic
# has to hold for every strategy including ones not written yet.
#
# The magnitude is chosen so that **adding a rank is exact**. At 1e18 the
# spacing of a float64 is 128, so `-1e18 + 1 == -1e18` and every named
# node ties -- the ranking is silently discarded and the tie falls to
# `TreeNode.__lt__`, which orders by node id. That is what the first
# version did, and an offline test caught it: an order of `[5, 3]` came
# back as `[3, 5]`. At 1e15 the spacing is 0.125, so ranks up to a
# million are carried exactly, which is far past the page budget.
_FLOOR = -1e15


class PlexEviction:
    """A standing eviction order, applied where the heap is built."""

    def __init__(self, source: str | None = None) -> None:
        self._source = source if source else os.environ.get("SGLANG_PLEX_EVICT")
        self._source_stamp: tuple[int, int] | None = None
        self._order: list[str] = []
        self.installs = 0
        self.calls = 0
        # Pages named against pages found among the offered leaves. A
        # policy whose every choice names a node that is no longer
        # evictable has decided nothing, and from outside that is
        # indistinguishable from a policy that agrees with the strategy.
        self.named = 0
        self.applied = 0

    @staticmethod
    def maybe() -> PlexEviction | None:
        if not os.environ.get("SGLANG_PLEX_EVICT"):
            return None
        return PlexEviction()

    def reload(self) -> None:
        """Re-read the order if the file changed.

        Stat-then-read rather than watch. The engine must not block on a
        runtime it does not own, and a stale order is a decision about an
        older tree rather than a failure.
        """
        if not self._source:
            return
        try:
            status = os.stat(self._source)
        except OSError:
            return
        stamp = (status.st_mtime_ns, status.st_size)
        if stamp == self._source_stamp:
            return
        self._source_stamp = stamp
        try:
            with open(self._source, encoding="utf-8") as handle:
                lines = handle.readlines()
        except OSError:
            return
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                document = json.loads(line)
            except ValueError:
                continue
            order = document.get("evict")
            if isinstance(order, list):
                self._order = [str(page) for page in order]
                self.installs += 1
            return

    def priorities(self, leaves: list[Any]) -> dict[int, float]:
        """Priority overrides, keyed by `id()` of the node.

        Keyed on object identity rather than on the node's own id
        because the caller is about to build a heap over these exact
        objects, and an override that missed by a rename would fail
        silently.
        """
        self.calls += 1
        if not self._order:
            return {}

        rank = {page: index for index, page in enumerate(self._order)}
        overrides: dict[int, float] = {}
        for node in leaves:
            node_id = getattr(node, "id", None)
            if node_id is None:
                continue
            position = rank.get(_page_id(node_id))
            if position is not None:
                overrides[id(node)] = _FLOOR + position

        # The ranking has to survive the arithmetic. If the order is long
        # enough that two ranks land on the same float, every node past
        # that point ties and the tie falls to node id -- which looks
        # exactly like a policy that ranked by node id. Refuse instead:
        # a silently discarded order is the failure this whole channel is
        # built to be able to see.
        if len(set(overrides.values())) != len(overrides):
            raise ValueError(
                f"eviction order of {len(self._order)} exceeds the float "
                f"precision at {_FLOOR}; ranks collided"
            )

        self.named += len(self._order)
        self.applied += len(overrides)
        self._report()
        return overrides

    def _report(self) -> None:
        """Say what the order reached, periodically.

        In the engine's own log, because that is where a reader goes when
        an arm comes out flat, and because `applied=0` is the difference
        between a policy that agreed with the strategy and one that named
        nodes the strategy had already released.
        """
        if self.calls % 200:
            return
        print(
            f"[plex-cache] calls={self.calls} named={self.named} "
            f"applied={self.applied} order={len(self._order)} "
            f"installs={self.installs}",
            file=sys.stderr,
            flush=True,
        )

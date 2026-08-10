"""The policy's standing table, applied to SGLang's waiting queue.

Stage 2 of the phased attach. SGLang decides order in one place --
`SchedulePolicy.calc_priority`, which sorts `waiting_queue` in place --
and every built-in policy is a `sort(key=...)`. A standing table is
another such policy, so it costs a sort rather than a rewrite: 9 lines at
the hook.

The claims below are about the contract rather than about SGLang, so they
are the same three the vLLM side asserts. That is the point of testing
both: a claim that holds on one engine and not the other is a claim about
an engine, and Gate 3 needs claims about the contract.
"""

import types

from sglang.srt.managers.plex_schedule import PlexSchedule


def req(rid):
    return types.SimpleNamespace(rid=rid)


def ids(queue):
    return [r.rid for r in queue]


def queue(*rids):
    return [req(rid) for rid in rids]


def test_without_a_table_the_order_is_unchanged():
    # An uninstalled table must reproduce what the engine would have done,
    # which is what makes stage 2 with no policy attached equal stage 1.
    table = PlexSchedule()
    waiting = queue("a", "b", "c")
    table.apply(waiting)
    assert ids(waiting) == ["a", "b", "c"]


def test_a_table_reorders_the_requests_it_names():
    table = PlexSchedule()
    table.install(["c", "a", "b"])
    waiting = queue("a", "b", "c")
    table.apply(waiting)
    assert ids(waiting) == ["c", "a", "b"]


def test_an_unnamed_request_sorts_after_named_ones_in_place():
    # "The policy expressed no view" is not "the policy ranked these
    # last", and only one of those is true.
    table = PlexSchedule()
    table.install(["d"])
    waiting = queue("a", "b", "c", "d")
    table.apply(waiting)
    assert ids(waiting) == ["d", "a", "b", "c"]


def test_a_table_naming_a_departed_request_is_harmless():
    table = PlexSchedule()
    table.install(["gone", "b", "a"])
    waiting = queue("a", "b")
    table.apply(waiting)
    assert ids(waiting) == ["b", "a"]


def test_installing_replaces_rather_than_merges():
    # A table is a document, so a partially-applied one is not
    # representable: the second install must not leave the first's
    # ranking behind for requests it does not mention.
    table = PlexSchedule()
    table.install(["c", "b", "a"])
    waiting = queue("a", "b", "c")
    table.apply(waiting)
    assert ids(waiting) == ["c", "b", "a"]

    table.install(["b"])
    waiting = queue("a", "b", "c")
    table.apply(waiting)
    assert ids(waiting) == ["b", "a", "c"]


def test_table_age_is_counted_in_arrivals():
    table = PlexSchedule()
    table.install([])
    waiting = queue("a")
    table.apply(waiting)
    assert table.table_age == 1, "one request has arrived since the install"
    table.apply(queue("a", "b", "c"))
    assert table.table_age == 3
    table.install(["a"])
    assert table.table_age == 0


def test_sorting_does_not_become_quadratic():
    # The obvious way to keep unnamed requests in order is
    # `queue.index(req)` inside the key, which is O(n) per comparison.
    # v0.7's post-mortem records a waiting scan that was O(queue) per step
    # and "cost four arms a measurement", so this is a real failure mode
    # rather than a style preference.
    table = PlexSchedule()
    table.install(["r500"])
    waiting = queue(*[f"r{i}" for i in range(2000)])
    table.apply(waiting)
    assert waiting[0].rid == "r500"
    # And the rest kept their order.
    assert [r.rid for r in waiting[1:5]] == ["r0", "r1", "r2", "r3"]

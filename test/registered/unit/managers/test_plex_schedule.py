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


# ── the admission hold (the route channel) ───────────────────────────────

import json as _json
import os as _os
import tempfile as _tempfile
import time as _time


def _gated(hold_ms="50"):
    """A schedule with a gate attached, and the verdict file to answer it."""
    handle = _tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    handle.close()
    _os.environ["SGLANG_PLEX_GATE"] = handle.name
    _os.environ["SGLANG_PLEX_GATE_MS"] = hold_ms
    try:
        return PlexSchedule(), handle.name
    finally:
        _os.environ.pop("SGLANG_PLEX_GATE", None)
        _os.environ.pop("SGLANG_PLEX_GATE_MS", None)


def _rule(path, verdicts):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        _json.dump(verdicts, handle)
    _os.replace(tmp, path)


def test_an_arrival_is_held_out_of_the_queue_so_the_gate_has_a_subject():
    schedule, _ = _gated(hold_ms="100000")
    queue = [req("a"), req("b")]

    schedule.hold_arrivals(queue)

    assert queue == [], "arrivals must leave the schedulable set to be pending"
    assert {req.rid for req in schedule.held_requests()} == {"a", "b"}


def test_assign_releases_and_the_request_is_not_re_held():
    schedule, gate = _gated(hold_ms="100000")
    queue = [req("a")]
    schedule.hold_arrivals(queue)

    _rule(gate, {"a": "assign"})
    schedule.hold_arrivals(queue)
    assert [req.rid for req in queue] == ["a"]

    # A released request must not be swept back into the hold on the next
    # pass, or it would be gated forever one step at a time.
    schedule.hold_arrivals(queue)
    assert [req.rid for req in queue] == ["a"]
    assert schedule.held_requests() == []


def test_reject_releases_and_records_rather_than_dropping():
    # A refusal is a kind of ending, not a kind of forgetting. vLLM's
    # port learned that by hanging: a request deleted from the hold left
    # its caller waiting for an answer with no code path left to produce
    # one.
    schedule, gate = _gated(hold_ms="100000")
    queue = [req("a")]
    schedule.hold_arrivals(queue)

    _rule(gate, {"a": "reject"})
    schedule.hold_arrivals(queue)

    assert [req.rid for req in queue] == ["a"], "it must come back to be ended"
    assert schedule.rejected_by_policy == ["a"]


def test_defer_keeps_it_held():
    schedule, gate = _gated(hold_ms="100000")
    queue = [req("a")]
    schedule.hold_arrivals(queue)

    _rule(gate, {"a": "defer"})
    schedule.hold_arrivals(queue)

    assert queue == []
    assert [req.rid for req in schedule.held_requests()] == ["a"]


def test_a_silent_policy_strands_nobody():
    # The case that would make this unshippable, so it is the one that
    # matters most. With no verdict ever written, the deadline must admit
    # everything — the declared default is what the engine would have
    # done unaided, because a policy that does not answer must not change
    # behaviour.
    schedule, _ = _gated(hold_ms="1")
    queue = [req("a"), req("b")]
    schedule.hold_arrivals(queue)
    assert queue == []

    _time.sleep(0.02)
    schedule.hold_arrivals(queue)

    assert {req.rid for req in queue} == {"a", "b"}
    assert schedule.held_requests() == []
    assert schedule.released_by_deadline == 2


def test_no_gate_configured_means_no_hold_at_all():
    # Off by default and invisible when off: the queue is untouched.
    schedule = PlexSchedule()
    queue = [req("a")]
    schedule.hold_arrivals(queue)
    assert [req.rid for req in queue] == ["a"]

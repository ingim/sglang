"""Staged verbs, mapped onto SGLang's own primitives -- or declined.

Stage 3 of the phased attach. The claim worth testing here is the one
that differs from vLLM: `move` is native and `pause` is not, which is the
reverse of the other engine on both counts.

That asymmetry is real rather than an oversight. SGLang retracts running
requests only as a whole batch, by its own victim order, to relieve
memory pressure -- there is no "pause this request" entry point. Building
one means either reaching into the batch internals (the engine-core
change Phase 3 forbids) or retracting and hoping the named request is
among the victims (a verb that reports success and does something else).
Declining is the honest third option, and declaring it is what lets a
portable policy notice rather than mis-attribute the difference to its
own logic.
"""

import types


class FakeReq:
    def __init__(self, rid, finished=False):
        self.rid = rid
        self.finished_reason = "stop" if finished else None


class FakeScheduler:
    def __init__(self, waiting, running, hierarchical=False):
        self.waiting_queue = waiting
        self.running_batch = types.SimpleNamespace(reqs=running)
        self.aborted = []
        self.backed_up = []
        if hierarchical:
            self.tree_cache = types.SimpleNamespace(
                write_backup=lambda node: self.backed_up.append(node)
            )
        else:
            self.tree_cache = types.SimpleNamespace()

    def abort_request(self, recv_req):
        self.aborted.append(recv_req.rid)


def verbs(waiting=(), running=(), hierarchical=False):
    from sglang.srt.managers.plex_verbs import PlexVerbs

    scheduler = FakeScheduler(list(waiting), list(running), hierarchical)
    return scheduler, PlexVerbs(scheduler)


def test_finish_aborts():
    scheduler, v = verbs(waiting=[FakeReq("a")])
    assert v.enact("finish", "a") is True
    assert scheduler.aborted == ["a"]
    assert v.take_refusals() == []


def test_finishing_a_finished_request_is_a_lost_race():
    _, v = verbs(waiting=[FakeReq("a", finished=True)])
    assert v.enact("finish", "a") is False
    (refusal,) = v.take_refusals()
    assert refusal.reason == "wrong-request-state"


def test_an_unknown_subject_is_named_as_such():
    _, v = verbs()
    assert v.enact("finish", "ghost") is False
    (refusal,) = v.take_refusals()
    assert refusal.reason == "unknown-subject"
    assert refusal.subject == "ghost"


def test_move_is_native_when_the_cache_is_tiered():
    # The verb SGLang has and vLLM does not.
    scheduler, v = verbs(hierarchical=True)
    assert v.enact("move", "node-7", to="host") is True
    assert scheduler.backed_up == ["node-7"]


def test_move_is_declined_when_there_is_nowhere_to_move_to():
    # A flat radix cache has one tier. Reporting success would tell a
    # policy its page is safe on host memory when it is not.
    _, v = verbs(hierarchical=False)
    assert v.enact("move", "node-7") is False
    (refusal,) = v.take_refusals()
    assert refusal.reason == "unsupported-on-this-engine"


def test_pause_is_declined_and_that_is_the_finding():
    # SGLang retracts whole batches by its own victim order, for memory
    # pressure. There is no per-request pause, and the two ways to fake
    # one are both worse than declining.
    _, v = verbs(running=[FakeReq("a")])
    assert v.enact("pause", "a") is False
    (refusal,) = v.take_refusals()
    assert refusal.verb == "pause"
    assert refusal.reason == "unsupported-on-this-engine"


def test_prefetch_and_rebalance_are_declined_with_a_reason():
    _, v = verbs(hierarchical=True)
    for verb in ["prefetch", "rebalance"]:
        assert v.enact(verb, "a") is False
    refusals = v.take_refusals()
    assert [r.verb for r in refusals] == ["prefetch", "rebalance"]
    assert all(r.reason == "unsupported-on-this-engine" for r in refusals)


def test_refusals_are_drained_not_repeated():
    _, v = verbs()
    v.enact("finish", "ghost")
    assert len(v.take_refusals()) == 1
    assert v.take_refusals() == []

"""The SGLang observer emits a document the v2 port reads, without a GPU.

Stage 1 must be reviewable and testable without hardware, or it is not the
cheap patch it claims to be. Everything the observer touches is read off
attributes, so a stand-in with those attributes exercises the whole path.

The case that matters most is the derived terminal edge. SGLang finishes
requests through several paths -- decode, prefill, the disaggregation modes,
dLLM -- and hooking each is call sites that drift apart as upstream adds a
fifth. Deriving departure from "was here last step, is not here now" is one
hook that cannot miss an edge upstream adds, and these tests hold it to that.
"""

import json
import types

from sglang.srt.managers.plex_observer import PlexObserver


class FakeReq:
    def __init__(self, rid, prompt=8, generated=0, matched=0):
        self.rid = rid
        self.origin_input_ids = list(range(prompt))
        self.output_ids = [0] * generated
        self.num_matched_prefix_tokens = matched


class FakeScheduler:
    def __init__(self, waiting, running):
        self.waiting_queue = waiting
        self.running_batch = types.SimpleNamespace(reqs=running)
        self.max_running_requests = 8
        self.max_total_num_tokens = 4096
        self.token_to_kv_pool_allocator = types.SimpleNamespace(
            available_size=lambda: 1024
        )
        self.plex_observer = None


def observer(waiting, running):
    scheduler = FakeScheduler(waiting, running)
    sink = types.SimpleNamespace(write=lambda _: None, flush=lambda: None)
    obs = PlexObserver(scheduler, sink, "sglang-0")
    scheduler.plex_observer = obs
    return obs


def test_a_step_document_has_the_shape_the_port_parses():
    a = FakeReq("r0", prompt=8, generated=2, matched=8)
    b = FakeReq("r1", prompt=16)
    obs = observer([b], [a])
    obs.on_request_queued(a)
    obs.on_request_queued(b)

    doc = json.loads(obs.on_step())

    assert doc["step"] == 1
    assert doc["target"] == "sglang-0"
    assert set(doc["subjects"]["request"]) == {"r0", "r1"}
    assert doc["events"]["admitted"] == ["r0", "r1"]

    r0 = doc["facts"]["r0"]
    assert r0["state"] == {"text": "active"}
    assert r0["computation_length"] == {"num": 10}
    assert r0["dispatch_input_tokens"] == {"num": 0}, "8 prompt, 8 matched"

    r1 = doc["facts"]["r1"]
    assert r1["state"] == {"text": "admitted"}
    assert r1["dispatch_input_tokens"] == {"num": 16}, "nothing matched"

    target = doc["facts"]["sglang-0"]
    assert target["total_kv_tokens"] == {"num": 4096}
    assert target["free_kv_tokens"] == {"num": 1024}
    assert target["queue_depth"] == {"num": 1}


def test_a_departure_is_derived_rather_than_hooked():
    # The whole reason this observer has two hooks instead of five.
    a, b = FakeReq("a"), FakeReq("b")
    scheduler = FakeScheduler([a, b], [])
    sink = types.SimpleNamespace(write=lambda _: None, flush=lambda: None)
    obs = PlexObserver(scheduler, sink, "sglang-0")
    obs.on_request_queued(a)
    obs.on_request_queued(b)

    first = json.loads(obs.on_step())
    assert "finished" not in first["events"], "nothing has left yet"

    # `b` finishes through whichever of SGLang's paths; the observer is not
    # told and does not need to be.
    scheduler.waiting_queue = [a]
    second = json.loads(obs.on_step())
    assert second["events"]["finished"] == [["b", "completed", "host"]]

    # And it is reported once.
    third = json.loads(obs.on_step())
    assert "finished" not in third["events"]


def test_arrival_order_is_recorded_because_nothing_else_records_it():
    a, b = FakeReq("a"), FakeReq("b")
    obs = observer([a, b], [])
    obs.on_request_queued(a)
    obs.on_request_queued(b)
    facts = json.loads(obs.on_step())["facts"]
    assert facts["a"]["arrival_seq"] == {"num": 0}
    assert facts["b"]["arrival_seq"] == {"num": 1}


def test_admissions_are_drained_not_repeated():
    a = FakeReq("a")
    obs = observer([a], [])
    obs.on_request_queued(a)
    assert json.loads(obs.on_step())["events"]["admitted"] == ["a"]
    # An event delivered twice is one thing a policy counts as two.
    assert "admitted" not in json.loads(obs.on_step())["events"]


def test_a_failing_sink_disables_the_observer_and_not_the_engine():
    a = FakeReq("a")
    scheduler = FakeScheduler([a], [])

    def explode(_):
        raise OSError("disk full")

    scheduler.plex_observer = PlexObserver(
        scheduler, types.SimpleNamespace(write=explode, flush=lambda: None), "sglang-0"
    )
    scheduler.plex_observer.emit_step()  # must not raise
    assert scheduler.plex_observer is None

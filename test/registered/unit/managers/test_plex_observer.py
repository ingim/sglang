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


class FakeNode:
    """One radix leaf: a page the tree could evict."""

    def __init__(self, node_id, tokens=16, lock_ref=0, priority=0.0):
        self.id = node_id
        self.value = list(range(tokens))
        self.lock_ref = lock_ref
        self.priority = priority


class FakeTree:
    def __init__(self, leaves):
        self.evictable_leaves = set(leaves)
        self.eviction_strategy = types.SimpleNamespace(
            get_priority=lambda node: node.priority
        )


class FakeScheduler:
    def __init__(self, waiting, running, leaves=()):
        self.tree_cache = FakeTree(leaves) if leaves else None
        self.waiting_queue = waiting
        self.running_batch = types.SimpleNamespace(reqs=running)
        self.max_running_requests = 8
        self.max_total_num_tokens = 4096
        self.token_to_kv_pool_allocator = types.SimpleNamespace(
            available_size=lambda: 1024
        )
        self.plex_observer = None


def observer(waiting, running, leaves=()):
    scheduler = FakeScheduler(waiting, running, leaves)
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


def test_a_departure_is_derived_but_confirmed_by_the_engine():
    # The reason this observer has two hooks instead of five: SGLang has
    # four terminal paths and hooking them all would grow a fifth.
    #
    # But absence alone is not departure, and a real run proved it. SGLang
    # retracts a request under memory pressure and puts it back, so one
    # request vanished at step 80, returned at 81, and a pure
    # set-difference reported it as finishing twice. Every accumulator in
    # the corpus counts one departure as one.
    a, b = FakeReq("a"), FakeReq("b")
    b.finished = lambda: True
    scheduler = FakeScheduler([a, b], [])
    sink = types.SimpleNamespace(write=lambda _: None, flush=lambda: None)
    obs = PlexObserver(scheduler, sink, "sglang-0")
    obs.on_request_queued(a)
    obs.on_request_queued(b)

    first = json.loads(obs.on_step())
    assert "finished" not in first["events"], "nothing has left yet"

    # `b` had marked itself finished, so its absence is a departure.
    scheduler.waiting_queue = [a]
    second = json.loads(obs.on_step())
    assert second["events"]["finished"] == [["b", "completed", "host"]]

    # And it is reported once.
    third = json.loads(obs.on_step())
    assert "finished" not in third["events"]


def test_a_retracted_request_is_remembered_rather_than_mourned():
    # The case that refuted the first derivation on a real engine.
    a = FakeReq("a")          # never marks itself finished
    scheduler = FakeScheduler([a], [])
    sink = types.SimpleNamespace(write=lambda _: None, flush=lambda: None)
    obs = PlexObserver(scheduler, sink, "sglang-0")
    obs.on_request_queued(a)
    json.loads(obs.on_step())

    scheduler.waiting_queue = []          # retracted under memory pressure
    gone = json.loads(obs.on_step())
    assert "finished" not in gone["events"], "absence is not departure"

    scheduler.waiting_queue = [a]         # and back again
    back = json.loads(obs.on_step())
    assert "finished" not in back["events"]
    assert back["subjects"]["request"] == ["a"]


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


def test_only_evictable_leaves_are_offered_and_in_the_tree_s_own_order():
    # Offered, not enumerated: a snapshot of the whole tree is per-node
    # state every step, and hands a policy thousands of subjects when a
    # handful are in play.
    #
    # Leaves only, and that is the tree's rule rather than a
    # simplification — evicting an interior node orphans everything below
    # it, so a policy able to name one could express a plan the engine
    # must refuse.
    leaves = [
        FakeNode(3, priority=0.9),
        FakeNode(1, priority=0.1),
        FakeNode(2, priority=0.5),
    ]
    obs = observer([], [], leaves=leaves)

    doc = json.loads(obs.on_step())
    pages = doc["subjects"]["page"]

    assert len(pages) == 3
    # Coldest first: the tree's own `eviction_strategy`, which is the
    # answer rather than a model of it.
    assert pages == ["p00000001", "p00000002", "p00000003"], pages
    assert doc["facts"]["p00000001"]["leaf"] == {"flag": True}
    assert doc["facts"]["p00000001"]["page-tokens"] == {"num": 16}


def test_a_locked_leaf_is_offered_but_marked_pinned():
    # `lock_ref` is the tree's own "someone is using this". Withholding
    # the page would tell a policy less than it needs: a plan that names
    # it is refusable, and a policy that cannot see it cannot learn why
    # its plans keep coming back short.
    obs = observer([], [], leaves=[FakeNode(7, lock_ref=2)])
    doc = json.loads(obs.on_step())
    assert doc["facts"]["p00000007"]["pinned"] == {"flag": True}


def test_offered_is_raised_once_per_page_not_once_per_step():
    obs = observer([], [], leaves=[FakeNode(1)])
    first = json.loads(obs.on_step())
    second = json.loads(obs.on_step())
    assert first["events"]["offered"] == ["p00000001"]
    assert "offered" not in second["events"], second["events"]


def test_no_tree_cache_means_no_pages_rather_than_no_engine():
    # Observation is never allowed to stop inference. An engine build
    # without a radix cache must lose the cache channel, not the run.
    obs = observer([], [])
    doc = json.loads(obs.on_step())
    assert "page" not in doc["subjects"]

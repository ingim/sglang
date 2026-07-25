"""Tests for SGLang's PLEX binding.

The binding is a port now: the state machine lives in `plex.engine`, which
ships with the contract and is tested there. What is worth testing here is
that this port describes SGLang correctly, and that the scheduler-facing
adapter still speaks the shape `scheduler.py` expects.

Set PLEX_TEST_POLICY to a built .plexpkg to also run the contract's own
conformance harness against this port, which drives a real policy through a
real host.
"""

import os
from array import array
from types import SimpleNamespace

import pytest

from sglang.srt.managers.async_plex import (
    AsyncPlexPolicyController,
    SGLangEnginePort,
)

# The host always reports what it changed, and the binding refuses an outcome
# that omits it rather than guessing the policy left state alone.
NO_STATE_CHANGE = {"requests": [], "groups": [], "shared": None}

plex_engine = pytest.importorskip("plex.engine")

PolicyController = plex_engine.PolicyController
POLICY = os.environ.get("PLEX_TEST_POLICY")


class FakeRequest:
    def __init__(self, request_id):
        self.rid = request_id
        self.origin_input_ids = array("q", [1, 2, 3, 4])
        self.origin_input_ids_unpadded = array("q", self.origin_input_ids)
        self.output_ids = array("q")
        self.sampling_params = SimpleNamespace(
            max_new_tokens=8,
            custom_params=None,
        )
        self.priority = None
        self.kv_committed_len = 0
        self.req_pool_idx = None
        self.num_matched_prefix_tokens = 0
        self.is_retracted = False
        self.kv = SimpleNamespace(kv_allocated_len=1)
        self.time_stats = SimpleNamespace(
            wait_queue_entry_time=0.0,
            scheduler_recv_time=0.0,
            created_time=0.0,
        )
        self.finished_reason = None

    def finished(self):
        return self.finished_reason is not None


def fake_scheduler():
    kv_cache = SimpleNamespace(get_kv_size_bytes=lambda: 1024)
    allocator = SimpleNamespace(
        available_size=lambda: 64,
        size_full=128,
        get_kvcache=lambda: kv_cache,
    )
    return SimpleNamespace(
        waiting_queue=[],
        running_batch=SimpleNamespace(reqs=[]),
        last_batch=None,
        token_to_kv_pool_allocator=allocator,
        model_config=SimpleNamespace(context_len=4096),
        page_size=1,
        max_prefill_tokens=128,
        max_running_requests=16,
    )


def build(scheduler, runtime=None):
    from plex.engine.testing import RecordingRuntime

    port = SGLangEnginePort(scheduler)
    controller = PolicyController(
        runtime or RecordingRuntime(),
        port,
        model="test-model",
        target_id="test",
    )
    return AsyncPlexPolicyController(controller, port)


def test_port_reads_the_schedulers_own_names():
    scheduler = fake_scheduler()
    waiting = [FakeRequest("a")]
    running = [FakeRequest("b")]
    scheduler.waiting_queue = waiting
    scheduler.running_batch = SimpleNamespace(reqs=running)
    port = SGLangEnginePort(scheduler)

    assert [view.engine_id for view in port.candidates()] == ["a"]
    assert [view.engine_id for view in port.residents()] == ["b"]

    capacity = port.capacity()
    assert capacity.max_total_tokens == scheduler.max_prefill_tokens
    assert capacity.max_selections == 1

    facts = port.engine_facts()
    assert facts["queue_depth"] == 1
    assert facts["running_requests"] == 1
    assert facts["free_kv_tokens"] == 64
    assert facts["total_kv_tokens"] == 128


def test_cache_budget_keeps_a_no_retraction_ordering_valid():
    """SGLang ranks rather than evicts, so keeping everything is an answer."""
    scheduler = fake_scheduler()
    scheduler.running_batch = SimpleNamespace(
        reqs=[FakeRequest("a"), FakeRequest("b")]
    )
    port = SGLangEnginePort(scheduler)
    residents = port.residents()
    resident_bytes = sum(view.size_bytes() for view in residents)

    assert port.cache_capacity(residents).max_bytes == resident_bytes


def test_schedule_plan_is_published_and_consumed():
    from plex.engine.testing import RecordingRuntime

    runtime = RecordingRuntime()
    scheduler = fake_scheduler()
    requests = [FakeRequest("a"), FakeRequest("b")]
    scheduler.waiting_queue = requests
    plex = build(scheduler, runtime)
    for request in requests:
        plex.register_request(request)

    plex.publish()
    epoch = runtime.submitted[0][1]
    runtime.reply(
        "schedule",
        epoch,
        {
            "status": "success",
            "plan": {
                "operation": "schedule",
                "plan": {"selections": [{"requests": [1], "token_budgets": [4]}]},
            },
            "state_update": NO_STATE_CHANGE,
        },
    )
    submitted = len(runtime.submitted)

    plan = plex.poll_schedule()

    assert plan is not None
    assert plan.rank("a") is None
    assert plan.rank("b") == 0
    assert len(runtime.submitted) == submitted


def test_missing_plan_is_native_fallback():
    assert build(fake_scheduler()).poll_schedule() is None


def test_cache_retraction_order_is_parsed():
    from plex.engine.testing import RecordingRuntime

    runtime = RecordingRuntime()
    scheduler = fake_scheduler()
    residents = [FakeRequest("a"), FakeRequest("b")]
    scheduler.running_batch = SimpleNamespace(reqs=residents)
    plex = build(scheduler, runtime)
    for request in residents:
        plex.register_request(request)

    plex.publish()
    epoch = next(
        e for channel, e, _ in runtime.submitted if channel == "cache"
    )
    runtime.reply(
        "cache",
        epoch,
        {
            "status": "success",
            "plan": {"operation": "cache", "plan": {"reclaim": [0]}},
            "state_update": NO_STATE_CHANGE,
        },
    )

    assert plex.cached_retraction_order(residents) == [1, 0]


def test_retraction_order_for_a_different_batch_is_refused():
    """A ranking over one set is not a ranking over another."""
    from plex.engine.testing import RecordingRuntime

    runtime = RecordingRuntime()
    scheduler = fake_scheduler()
    residents = [FakeRequest("a"), FakeRequest("b")]
    scheduler.running_batch = SimpleNamespace(reqs=residents)
    plex = build(scheduler, runtime)
    for request in residents:
        plex.register_request(request)
    plex.publish()
    epoch = next(e for channel, e, _ in runtime.submitted if channel == "cache")
    runtime.reply(
        "cache",
        epoch,
        {
            "status": "success",
            "plan": {"operation": "cache", "plan": {"reclaim": [0]}},
            "state_update": NO_STATE_CHANGE,
        },
    )

    assert plex.cached_retraction_order(list(reversed(residents))) is None


def test_outcome_with_unnegotiated_actions_is_rejected():
    from plex.engine.testing import RecordingRuntime

    runtime = RecordingRuntime()
    scheduler = fake_scheduler()
    requests = [FakeRequest("a"), FakeRequest("b")]
    scheduler.waiting_queue = requests
    plex = build(scheduler, runtime)
    for request in requests:
        plex.register_request(request)

    plex.publish()
    epoch = runtime.submitted[0][1]
    runtime.reply(
        "schedule",
        epoch,
        {
            "status": "success",
            "plan": {
                "operation": "schedule",
                "plan": {"selections": [{"requests": [1], "token_budgets": [4]}]},
            },
            "state_update": NO_STATE_CHANGE,
            "actions": [{"mechanic": "request.pause@1"}],
        },
    )

    assert plex.poll_schedule() is None


def test_feedback_waits_for_publish():
    from plex.engine.testing import RecordingRuntime

    runtime = RecordingRuntime()
    scheduler = fake_scheduler()
    request = FakeRequest("request")
    plex = build(scheduler, runtime)
    plex.register_request(request)
    request.finished_reason = SimpleNamespace(to_json=lambda: {"type": "stop"})
    plex.mark_finished(request)

    assert runtime.submitted == []
    plex.publish()

    feedback = [
        event for event in runtime.events() if event["operation"] == "feedback"
    ]
    assert len(feedback) == 1


@pytest.mark.skipif(not POLICY, reason="PLEX_TEST_POLICY is not set")
def test_port_passes_the_contract_conformance_harness():
    """Drive this port through a real policy and a real host."""
    from plex.engine.testing import conformance

    scheduler = fake_scheduler()
    scheduler.waiting_queue = [FakeRequest("a"), FakeRequest("b")]
    scheduler.running_batch = SimpleNamespace(reqs=[FakeRequest("c")])

    report = conformance(SGLangEnginePort(scheduler), POLICY)

    assert report.problems == [], report.problems

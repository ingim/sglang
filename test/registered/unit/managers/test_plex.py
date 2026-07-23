import json
from array import array
from types import SimpleNamespace

from sglang.srt.managers.async_plex import AsyncPlexPolicyController


class FakeAsyncRuntime:
    def __init__(self):
        self.submissions = []
        self.latest_results = {}

    def try_submit(self, channel, epoch, event):
        self.submissions.append((channel, epoch, event))
        return True

    def try_submit_bytes(self, channel, epoch, event):
        self.submissions.append((channel, epoch, json.loads(event)))
        return True

    def latest(self, channel, after_epoch=0):
        result = self.latest_results.get(channel)
        if result is None or result[0] <= after_epoch:
            return None
        return result

    def shutdown(self):
        return None


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


def test_async_schedule_plan_is_published_and_consumed():
    runtime = FakeAsyncRuntime()
    scheduler = fake_scheduler()
    requests = [FakeRequest("a"), FakeRequest("b")]
    scheduler.waiting_queue = requests
    controller = AsyncPlexPolicyController(
        runtime,
        model="test-model",
        target_id="test",
    )
    for request in requests:
        controller.register_request(request)

    controller.publish(scheduler)
    epoch = controller.epoch
    runtime.latest_results["schedule"] = (
        epoch,
        {
            "status": "success",
            "decision": {"selected": [{"candidate_index": 1, "token_budget": 4}]},
        },
    )
    submitted = len(runtime.submissions)

    plan = controller.poll_schedule()

    assert plan is not None
    assert plan.rank("a") is None
    assert plan.rank("b") == 0
    assert len(runtime.submissions) == submitted


def test_async_missing_plan_is_native_fallback():
    controller = AsyncPlexPolicyController(
        FakeAsyncRuntime(),
        model="test-model",
        target_id="test",
    )

    assert controller.poll_schedule() is None


def test_async_feedback_waits_for_publish():
    runtime = FakeAsyncRuntime()
    scheduler = fake_scheduler()
    request = FakeRequest("request")
    controller = AsyncPlexPolicyController(
        runtime,
        model="test-model",
        target_id="test",
    )
    controller.register_request(request)
    request.finished_reason = SimpleNamespace(to_json=lambda: {"type": "stop"})
    controller.mark_finished(request)

    assert runtime.submissions == []
    controller.publish(scheduler)

    feedback = [
        event for channel, _epoch, event in runtime.submissions if channel == "feedback"
    ]
    assert len(feedback) == 1

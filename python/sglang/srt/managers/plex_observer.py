"""PLEX v2 observer: the SGLang scheduler, in the contract's vocabulary.

Stage ① of the phased attach in `.wiki/v2/plan.md`: events and facts flowing,
zero influence. The engine schedules exactly as it does now and a policy
watches.

## Two hooks, and one of them is a hook only because arrival order is not
## recoverable afterwards

vLLM has a single funnel every terminal edge passes through (`_free_request`),
so the observer there hooks arrival, terminal and step. SGLang does not:
requests finish through `process_batch_result_decode`,
`process_batch_result_prefill`, the disaggregation paths and the dLLM path,
and hooking each is four call sites that can drift apart as upstream adds a
fifth.

So this derives the terminal edge instead. The observer already walks the
waiting queue and the running batch every step; a request that was there last
step and is not there now has left. One hook, no possibility of missing an
edge upstream adds, and the cost is that the *reason* is unknown — which is
honest, because at the step boundary it is.

Arrival cannot be derived the same way. A request appearing in the queue tells
you it arrived, but not in what order relative to one that arrived and was
scheduled within the same step, and a fairness policy that cannot order
arrivals has no fairness to enforce. Hence the second hook.

## Why this stays small

The facts are derived by `plex-port-vllm`'s sibling on the Rust side, not
here. v0.7's SGLang integration computed them in-engine; v2's shim reads what
SGLang already holds and hands it over. Every entry below is an attribute read
or a subtraction of two — deriving the same quantity in two places is how two
descriptions of one engine start to disagree.

## What attaching this can change

Nothing. There is no branch on policy output, because stage ① has none. Run
the engine with and without it and the outputs must be identical.
"""

from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler import Scheduler


# The prefix match as the engine first stated it, per request.
#
# It cannot be read off a request when the observer samples one. SGLang
# publishes a request only once it is already running, and by then
# `prefix_indices` covers everything the request has computed, not
# everything it reused -- a hit rate summed from that reads 0.9990 on a
# baseline arm, which is a clock started after the race, not a cache
# working perfectly. And `num_matched_prefix_tokens` is not a fallback:
# it is populated at schedule time and SGLang stops populating it above
# 128 queued requests, which is inside the regime under test.
#
# So it is recorded where it is true: at the one call that asks the
# radix tree, and only the first time for a request.
FIRST_MATCH: dict[str, int] = {}
FIRST_MATCH_LIMIT = 65536


def note_match(rid: str, matched_tokens: int) -> None:
    """Record the engine's own answer to "how much of this did you already have"."""
    if rid in FIRST_MATCH:
        return
    if len(FIRST_MATCH) >= FIRST_MATCH_LIMIT:
        FIRST_MATCH.clear()
    FIRST_MATCH[rid] = int(matched_tokens)


def _page_id(node_id: Any) -> str:
    """A page's name, from the radix node's own id.

    SGLang's node ids are integers from a counter, so they are stable
    for as long as the node lives and mean nothing across a restart --
    which is the right property here, because a radix node *is* a
    per-process object. vLLM's equivalent had to be derived from content
    bytes because `hash()` of `bytes` is randomised per interpreter and
    the id read as content-addressed while being process-addressed.

    Defined here and imported by `plex_cache` rather than copied. A
    second copy is how the referent gets lost: a policy naming a page
    that resolves to nothing is indistinguishable from a policy that
    agrees with the engine.
    """
    return f"p{int(node_id):08x}"


# Mirrors of the two constants vLLM's port had to introduce, kept at the
# same values so a cross-engine comparison is comparing engines and not
# two different guesses about the cost of a token.
#
# `_MS_PER_PREFILL_TOKEN` is coarse on purpose: the policies that read it
# compare recompute against swap, and swap is unavailable here, so the
# comparison's outcome does not depend on the rate being accurate. The
# falsifier is to vary it by two orders of magnitude and confirm the
# disposition split does not move.
_MS_PER_PREFILL_TOKEN = 0.05
# Not "very expensive" — *unavailable*. SGLang retracts by recompute:
# `retract_decode` releases the KV and restarts the request, and there is
# no request-level CPU swap for a port to drive. A policy comparing
# against an option that does not exist should get the same answer every
# time, which a sentinel gives and a large guess does not.
_NO_SWAP_MS = 1_000_000_000


def _client_of(rid: str) -> str:
    """The tenant a request belongs to, from the caller's own id."""
    head, sep, _ = rid.partition("::")
    return head if sep else rid


def _group_of(rid: str) -> str:
    """The group a request belongs to, from the caller's own id.

    `<client>::<group>::<rest>`. Falls back to the client when there is
    no second separator, so a workload that names no groups puts each
    tenant in one group rather than each request in its own — the
    latter is the degenerate case the group-aware policies are written
    against.
    """
    parts = rid.split("::")
    if len(parts) >= 3:
        return f"{parts[0]}::{parts[1]}"
    return _client_of(rid)


def _field_of(rid: str, index: int) -> str:
    """The nth `::`-separated field of a request id, or the last there is.

    Degrading to a coarser level rather than to nothing matters: a
    workload that names a user and no application should look like one
    application per user, not like an absent fact a policy silently
    defaults.
    """
    parts = rid.split("::")
    if not parts:
        return rid
    return parts[min(index, len(parts) - 1)]


class PlexObserver:
    """One SGLang scheduler, in the contract's vocabulary. Holds no policy."""

    def __init__(self, scheduler: Scheduler, sink: Any, target: str = "sglang-0") -> None:
        self._scheduler = scheduler
        self._sink = sink
        self._target = target
        self._step = 0
        # Arrival order, which nothing in SGLang records.
        self._arrival_seq: dict[str, int] = {}
        self._arrival_ms: dict[str, int] = {}
        # How many eviction candidates to offer per step. Bounded for the
        # same reason as on vLLM: the cost of describing pages must stay
        # proportional to how many the engine is about to touch.
        self._page_budget = int(os.environ.get("SGLANG_PLEX_PAGE_BUDGET", "64"))
        # A short window of observed step durations, for the timing
        # facts. Bounded so it tracks the engine's current behaviour
        # rather than averaging over a run whose shape has changed.
        self._step_ms_window: list[int] = []
        self._last_step_ms: int | None = None
        # Bytes of KV per token. Declared rather than guessed: it depends
        # on the model's layers, heads and dtype, and a port that assumed
        # one would publish a confident byte count for a different model.
        self._bytes_per_token = int(
            os.environ.get("SGLANG_PLEX_BYTES_PER_TOKEN", "0")
        )
        self._offered_last: set[str] = set()
        # Branch demand, computed once per step: every offered page asks
        # the same question of the same queue.
        self._demand: dict[int, set[str]] = {}
        self._demand_step = -1
        self._arrivals = 0
        # Who was here last step, for deriving terminal edges.
        self._present: set[str] = set()
        # Requests the engine has marked finished but which are still
        # tracked. Departure is `absent AND finished`, never absent alone.
        self._finishing: set[str] = set()
        # Requests that vanished without finishing: retracted, not gone.
        self._retracted: list[str] = []
        # Every request ever retracted, for `preempted`. Separate from
        # `_retracted` (this step's) because the fact is "has been", not
        # "is being".
        self._retracted_ever: set[str] = set()
        self._admitted: list[str] = []
        # Cumulative engine time per request, in microseconds. Nothing in
        # SGLang records it, and it is the fact the whole fair-share half
        # of the corpus ranks on: `vtc`, `fairserve` and `dlpm` are all
        # written about service received.
        self._service_us: dict[str, int] = {}
        # Output length last step, for the `progress` event. A policy
        # that accumulates work per request needs an edge, not a level.
        self._progress_last: dict[str, int] = {}
        # One deadline, or one per tenant. See vLLM's port for why the
        # second form is not a convenience: a deadline-ordering policy
        # sorts by `slo-ms - waiting-ms`, and under a constant `slo-ms`
        # that expression is strictly decreasing in waiting time, so the
        # earliest-deadline order *is* the arrival order and EDF is FCFS
        # with extra arithmetic. Heterogeneous deadlines are what make a
        # deadline schedulable.
        #
        #   SGLANG_PLEX_SLO_MS=30000
        #   SGLANG_PLEX_SLO_MS=c0=60000,c1=15000,default=30000
        self._slo_ms = 0
        self._slo_by_client: dict[str, int] = {}
        raw = (os.environ.get("SGLANG_PLEX_SLO_MS", "") or "").strip()
        if "=" in raw:
            for item in raw.split(","):
                name, _, value = item.partition("=")
                try:
                    number = int(float(value))
                except ValueError:
                    continue
                if name.strip() == "default":
                    self._slo_ms = number
                else:
                    self._slo_by_client[name.strip()] = number
        elif raw:
            try:
                self._slo_ms = int(float(raw))
            except ValueError:
                self._slo_ms = 0
        # The service classes a deployment sells. Neither is derivable:
        # whether a request may be deferred is a commercial fact about
        # the tenant, and a per-token latency target is a promise
        # someone made. The operator states both, exactly as the tenant
        # label is stated in the request id.
        self._best_effort = {
            name.strip()
            for name in os.environ.get("SGLANG_PLEX_BEST_EFFORT", "").split(",")
            if name.strip()
        }
        self._tpot_ms = int(os.environ.get("SGLANG_PLEX_TPOT_MS", "0") or 0)
        # Stage 3, if a source was named.
        from sglang.srt.managers.plex_verbs import PlexVerbs

        self._verbs = PlexVerbs.maybe(scheduler)

    # ── the hooks ────────────────────────────────────────────────────────

    def on_request_queued(self, req: Req) -> None:
        """A request entered the waiting queue."""
        rid = req.rid
        if rid not in self._arrival_seq:
            self._arrival_seq[rid] = self._arrivals
            # SGLang records no wall-clock arrival the observer can read
            # at this point, so it is taken here. That makes `arrival-ms`
            # the moment the queue saw the request rather than the moment
            # the server did — a difference of the request's own parsing,
            # and one worth stating rather than letting a policy assume
            # otherwise.
            self._arrival_ms[rid] = int(time.time() * 1000)
            self._arrivals += 1
        self._admitted.append(rid)

    def emit_step(self) -> None:
        """Write one step document, and never fail the engine.

        A port that can raise inside the scheduler is a port that can take
        the engine down, and stage ① is supposed to be unable to change
        anything — including whether the step completes. So the write is
        guarded and a failure disables the observer: losing observation is
        a measurement problem, losing the step is an outage.
        """
        try:
            self._sink.write(self.on_step() + "\n")
            self._sink.flush()
        except Exception:  # noqa: BLE001 - see docstring
            self._sink = None
            self._scheduler.plex_observer = None

    def drain_verbs(self) -> int:
        """Enact staged verbs. Called before a scheduling pass begins.

        Separate from the step document for the reason vLLM's port
        learned by crashing twice: a read may happen mid-pass and a write
        may not. SGLang's loop is structured differently, but `finish`
        mutates the same queues a pass consumes, so the rule holds.
        """
        if self._verbs is None:
            return 0
        return self._verbs.drain()

    def on_step(self) -> str:
        """One scheduler step, as a document the port reads."""
        self._step += 1
        tracked = self._tracked()
        present = {req.rid for req in tracked}

        # Terminal edges.
        #
        # Absence is *not* departure. A real run refuted that: SGLang
        # retracts a request under memory pressure and puts it back, so
        # one request vanished at step 80, returned at step 81, and a
        # set-difference derivation reported it as finishing twice.
        # Every accumulator in the corpus counts one departure as one.
        #
        # So absence is a *candidate*, and `req.finished()` — the
        # engine's own answer — is what confirms it. A request that
        # disappears without having finished has been retracted, and is
        # remembered rather than mourned.
        for req in tracked:
            if getattr(req, "finished", None) is not None and req.finished():
                self._finishing.add(req.rid)

        gone = self._present - present
        departed = sorted(rid for rid in gone if rid in self._finishing)
        self._retracted = sorted(rid for rid in gone if rid not in self._finishing)
        self._retracted_ever.update(self._retracted)
        self._present = present
        for rid in departed:
            self._arrival_seq.pop(rid, None)
            self._arrival_ms.pop(rid, None)
            self._finishing.discard(rid)

        document_now_ms = int(time.time() * 1000)
        elapsed_ms = 0
        if self._last_step_ms is not None:
            elapsed_ms = max(document_now_ms - self._last_step_ms, 0)
            self._step_ms_window.append(elapsed_ms)
            if len(self._step_ms_window) > 32:
                self._step_ms_window.pop(0)
        self._last_step_ms = document_now_ms
        self._charge_service(elapsed_ms)
        document = {
            "step": self._step,
            "now-ms": document_now_ms,
            "target": self._target,
            "subjects": self._subjects(tracked),
            "facts": self._facts(tracked, document_now_ms),
            "events": self._events(departed, self._progress(tracked)),
        }
        self._admitted.clear()
        return json.dumps(document)

    # ── scraping ─────────────────────────────────────────────────────────

    def _held_requests(self) -> list[Req]:
        """Requests the gate is holding, which the scheduler cannot see.

        They are out of `waiting_queue` on purpose — that is what the
        hold is — so the observer has to reach the holder to describe
        them. Without this the gate would be offered no subject, which is
        exactly how `llumnix` came to be called and decide nothing.
        """
        policy = getattr(self._scheduler, "policy", None)
        schedule = getattr(policy, "plex_schedule", None)
        held = getattr(schedule, "held_requests", None)
        return list(held()) if held is not None else []

    def _tracked(self) -> list[Req]:
        scheduler = self._scheduler
        running = list(getattr(scheduler.running_batch, "reqs", []) or [])
        return [*self._held_requests(), *scheduler.waiting_queue, *running]

    def _charge_service(self, elapsed_ms: int) -> None:
        """Charge the step that just ended to whoever was running in it.

        The interval is measured between step documents, so it is time
        the engine spent, not time the port spent looking. It is charged
        to `running_batch.reqs` because that is the set the engine
        actually computed over.

        Consequence worth stating, and the same one vLLM's port states:
        the sum over requests exceeds wall clock whenever the batch
        holds more than one. This is not a partition of engine time and
        must not be summed to get utilisation.
        """
        if elapsed_ms <= 0:
            return
        charge = elapsed_ms * 1000
        live = set()
        for req in getattr(self._scheduler.running_batch, "reqs", []) or []:
            rid = getattr(req, "rid", None)
            if rid is None:
                continue
            live.add(rid)
            self._service_us[rid] = self._service_us.get(rid, 0) + charge
        # A ledger that outlives its subjects is a leak on a long run,
        # and the fact is only ever read for a request that is present.
        if len(self._service_us) > len(live) + 4096:
            self._service_us = {
                key: value for key, value in self._service_us.items() if key in live
            }

    @staticmethod
    def _tenant_of(rid: str) -> str:
        """The tenant a request belongs to, from its id.

        The same field vLLM's port reads: everything before the first
        `::` in a `<user>::<app>::<stage>::<unique>` id, with any
        engine-added generation prefix stripped so an operator
        declaring a service class writes the tenant they sell to and
        not an id format.

        This matters more here than on vLLM. SGLang's server invents a
        uuid unless the client supplies `rid`, so an unlabelled harness
        gets the default deadline and `is-best-effort: false` for every
        request — silently, and indistinguishably from a deployment
        that sells one tier.
        """
        head = rid.split("::", 1)[0]
        for prefix in ("cmpl-", "chat-", "chatcmpl-", "embd-"):
            if head.startswith(prefix):
                return head[len(prefix) :]
        return head

    def _slo_of(self, rid: str) -> int:
        """This request's deadline: its tenant's, or the deployment's."""
        return self._slo_by_client.get(self._tenant_of(rid), self._slo_ms)

    def _progress(self, tracked: list[Req]) -> list[list[Any]]:
        """Output growth since last step, per request.

        An edge and not a level. A policy that accumulates work — every
        one that maintains a virtual clock does — needs to be told what
        changed, and reconstructing that from a level is exactly the
        bookkeeping the contract exists to not make policies do.
        """
        deltas: list[list[Any]] = []
        live = set()
        for req in tracked:
            rid = req.rid
            live.add(rid)
            generated = len(req.output_ids)
            grew = generated - self._progress_last.get(rid, 0)
            self._progress_last[rid] = generated
            if grew > 0:
                deltas.append([rid, grew])
        for rid in list(self._progress_last):
            if rid not in live:
                del self._progress_last[rid]
        return deltas

    def _facts(self, tracked: list[Req], now_ms: int) -> dict[str, dict[str, Any]]:
        running_ids = {
            req.rid for req in getattr(self._scheduler.running_batch, "reqs", []) or []
        }
        held_ids = {req.rid for req in self._held_requests()}
        facts: dict[str, dict[str, Any]] = {}
        for req in tracked:
            running = req.rid in running_ids
            prompt = len(req.origin_input_ids)
            generated = len(req.output_ids)
            # SGLang's own total cached prefix length: on-device
            # `prefix_indices` plus any host hit, capped at the max allowed
            # prefix. It is the radix tree's answer, which is what makes
            # pages content-addressed here in the first place — and it is
            # the field SGLang itself uses to estimate uncached tokens,
            # rather than `len(prefix_indices)`, which misses the host half.
            cached = int(getattr(req, "num_matched_prefix_tokens", 0) or 0)
            # The engine's stated match, kept apart from the total
            # above. Recorded at the match itself; see `note_match`.
            stated = FIRST_MATCH.get(req.rid, 0)
            # `len` and not truthiness: `prefix_indices` is a tensor, and
            # `x or ()` on a tensor with more than one element raises
            # rather than falling back.
            device_hit = getattr(req, "prefix_indices", None)
            if device_hit is not None:
                cached = max(cached, len(device_hit))
            # Whether the engine looked, not whether it found anything.
            #
            # `num_matched_prefix_tokens` is populated at schedule time
            # and SGLang stops populating it under load: LPM is its
            # default policy and `_determine_active_policy` degrades to
            # FCFS once the waiting queue passes 128, while the
            # cache-agnostic fallback path is gated on
            # `supports_fast_match_prefix()`, which no cache in this
            # tree overrides to True. So above 128 queued requests
            # nothing computes the match.
            #
            # 128 is inside the regime `v2/host/src/load.rs` requires.
            # Measured on a two-replica 14B fleet at mean queue depth
            # 80: every one of 89,196 step documents published
            # `cached_tokens` 0, while the engine's own Prometheus
            # counter reported a 53% hit rate over the same run. The
            # observer was not reporting an absence of hits, it was
            # reporting an absence of measurement in the words of a
            # measurement -- and precisely when the load qualifies.
            #
            # A fact the engine has not computed is not a fact, so it is
            # not published. `last_node` is the discriminator: it is
            # None until a match is attempted and a node afterwards.
            matched = (
                getattr(req, "last_node", None) is not None or cached > 0
            )
            facts[req.rid] = {
                # `pending` only while genuinely held out of the queue.
                # Publishing it for a queued request would be a lie: the
                # engine has already accepted that one, and a policy
                # answering `reject` would rule on something admitted.
                "state": {
                    "text": "pending"
                    if req.rid in held_ids
                    else ("active" if running else "admitted")
                },
                "arrival_seq": {"num": self._arrival_seq.get(req.rid, 0)},
                "arrival_ms": {"num": self._arrival_ms.get(req.rid, now_ms)},
                # SGLang's retraction is its preemption: a request put
                # back under memory pressure has been preempted, whatever
                # the engine calls it. Derived from the observer's own
                # record because SGLang keeps no counter — and stating it
                # is better than leaving a policy to infer it from a
                # request that mysteriously restarted.
                "preempted": {"flag": req.rid in self._retracted_ever},
                "prompt_tokens": {"num": prompt},
                "generated_tokens": {"num": generated},
                "computation_length": {"num": prompt + generated},
                "queue_member": {"flag": not running and req.rid not in held_ids},
                "waiting_ms": {"num": max(now_ms - self._arrival_ms.get(req.rid, now_ms), 0)},
                "current_queue_ms": {
                    "num": 0
                    if running
                    else max(now_ms - self._arrival_ms.get(req.rid, now_ms), 0)
                },
                # Engine time received, cumulative. The fact the
                # fair-share half of the corpus ranks on, and the one
                # SGLang keeps no counter for; see `_charge_service`.
                "service_us": {"num": self._service_us.get(req.rid, 0)},
                # The two costs an eviction policy compares, published
                # because this engine is the only party that knows
                # either. Unanswered, `qlm` falls back to comparing two
                # token counts as if they were durations and returns the
                # same disposition every time.
                #
                # Recompute is the tokens a retraction would discard, at
                # a stated rate. SGLang's `retract_decode` writes what it
                # can back into the radix tree, so a restart may recover
                # part of the prefix and this is an upper bound — which
                # is the safe direction for a policy deciding whether
                # preemption is worth it. Swap is the sentinel: not
                # expensive, unavailable.
                "recompute_cost_ms": {
                    "num": int(max(prompt + generated - cached, 0)
                               * _MS_PER_PREFILL_TOKEN)
                },
                "swap_cost_ms": {"num": _NO_SWAP_MS},
                # Published only when stated. A deadline of zero is not
                # "no deadline" to a policy that subtracts from it, it
                # is a deadline already missed, and the two must not
                # read the same.
                **(
                    {"slo_ms": {"num": self._slo_of(req.rid)}}
                    if self._slo_of(req.rid) > 0
                    else {}
                ),
                # Published unconditionally once any tenant is declared
                # best-effort, because `false` is then a real answer —
                # "this request is promised" — and must not read the
                # same as "nobody said".
                **(
                    {
                        "is_best_effort": {
                            "flag": self._tenant_of(req.rid) in self._best_effort
                        }
                    }
                    if self._best_effort
                    else {}
                ),
                **({"tpot_ms": {"num": self._tpot_ms}} if self._tpot_ms > 0 else {}),
                # Who this request belongs to. None of the four is
                # anything SGLang knows: a tenant is a deployment's
                # concept, a group is the caller's agent or session, a
                # user and an application and a stage are the caller's
                # own structure. All of them arrive in the request id or
                # not at all — and an engine cannot invent an ownership
                # it was never told about.
                #
                # Absent, every request reports the same tenant and a
                # fairness policy sees one tenant to be fair between,
                # which is not a hard case for it, it is no case at all.
                # Measured on vLLM before the equivalent existed: `vtc`'s
                # two arms byte-identical at a gap of 7.000.
                "client_id": {"text": _client_of(req.rid)},
                "group": {"text": _group_of(req.rid)},
                # `<user>::<application>::<stage>::<unique>`, each
                # falling back to what precedes it so a workload that
                # names fewer levels degrades to coarser accounting
                # rather than to none.
                "user_id": {"text": _field_of(req.rid, 0)},
                "application_id": {"text": _field_of(req.rid, 1)},
                "stage_id": {"text": _field_of(req.rid, 2)},
                # Spellings the ports use for quantities already
                # published above. A port reading `input-tokens` gets
                # `unknown-key` and takes a default, and the default is
                # indistinguishable from an answer.
                "input_tokens": {"num": prompt},
                "output_tokens": {"num": generated},
            }
            # Prefix-cache facts, and only where the engine computed
            # one. vLLM cannot publish these from an observer at all --
            # it computes hits inside `schedule()` and never keeps them
            # on the request -- and SGLang's radix tree answers per
            # request only when something asked it. Where nothing asked,
            # the observer says nothing rather than saying zero.
            if matched:
                facts[req.rid].update({
                    "dispatch_input_tokens": {"num": max(prompt - cached, 0)},
                    "cached_tokens": {"num": cached},
                    "uncached_tokens": {"num": max(prompt - cached, 0)},
                    "lpm_hit_tokens": {"num": stated},
                    "prefix_hit_ratio_ppm": {
                        "num": min(cached * 1_000_000 // prompt, 1_000_000)
                        if prompt
                        else 0
                    },
                })
        # The facts a cache policy ranks on, and they were absent.
        #
        # SGLang published six: resident, targets, tier, pinned,
        # page-tokens and leaf. None of them says how recently a page was
        # touched or how often, so `hotprefix` -- whose whole kernel is
        # hotness -- read `unknown-key` for every page, scored them
        # identically, and would have installed the offer's own order
        # under a different name. The tree already holds all of it:
        # `last_access_time` and `hit_count` are TreeNode fields, and the
        # radix structure gives the prefix and its beneficiaries directly,
        # which on vLLM's flat pool had to be reconstructed.
        now_monotonic = time.monotonic()
        for page_id, node in self._offered_pages():
            value = getattr(node, "value", None)
            tokens = len(value) if value is not None else 0
            last_access = getattr(node, "last_access_time", None)
            facts[page_id] = {
                "resident": {"flag": True},
                "targets": {"ids": [self._target]},
                "tier": {"text": "gpu"},
                # `lock_ref` is the tree's own "someone is using this".
                "pinned": {"flag": int(getattr(node, "lock_ref", 0) or 0) > 0},
                "page-tokens": {"num": tokens},
                # The same number under the name the ports read. vLLM
                # publishes both and the corpus is split over which it
                # asks for.
                "size-tokens": {"num": tokens},
                "leaf": {"flag": True},
                # Wall clock, because that is what the fact name promises
                # and what every port compares against `now-ms`.
                # `last_access_time` is monotonic, so it is carried back
                # to the same origin rather than published raw.
                **(
                    {
                        "last-access-ms": {
                            "num": max(
                                now_ms - int((now_monotonic - last_access) * 1000),
                                0,
                            )
                        }
                    }
                    if last_access is not None
                    else {}
                ),
                "hit-count": {"num": int(getattr(node, "hit_count", 0) or 0)},
                # The prefix this page belongs to: the root of its branch,
                # which is what survives the page. SGLang has a tree, so
                # this is a walk rather than the side table vLLM needs.
                "prefix": {"text": self._prefix_of(node, page_id)},
                # Who wants it. The requests currently holding a lock on
                # this branch -- `lock_ref` is the tree's own count of
                # them, and the ids come from the requests whose last
                # matched node is at or below here.
                "beneficiaries": {"ids": self._wanted_by(node)},
            }
        facts[self._target] = self._target_facts()
        return facts


    def _step_timing(self) -> tuple[int, int]:
        """Median step wall clock, and the decode cost it implies.

        **Measured here, not read from the engine.** Neither engine keeps
        a per-step duration an observer can read — `forward_ct` is a
        count — and the honest options were to publish nothing or to
        measure. Publishing nothing loses four names the corpus reads;
        inventing a plausible constant would hand a policy a confident
        number about a machine nobody timed.

        So the observer times its own steps and says so. It is the
        *observer's* view of the step, which includes anything else the
        engine did between two documents — and that is the quantity a
        policy reasoning about "how long until my turn" actually wants.

        Median over a short window rather than a mean: one slow step
        (a cold kernel, a neighbour on the GPU) would drag a mean for
        many steps afterwards, and a policy acting on it would be acting
        on an outlier that has already passed.
        """
        if len(self._step_ms_window) < 3:
            return (0, 0)
        window = sorted(self._step_ms_window)
        median = window[len(window) // 2]
        return (median, median * 1000)

    def _table_staleness(self) -> dict[str, Any]:
        """How old the installed order is, when there is one.

        The counter has existed on both engines since their tables were
        written and neither ever surfaced it — the same shape of defect
        as the verbs nobody carried, and found for the same reason: a
        number that could have answered a question nobody had yet asked.

        Absent when nothing is installed, because zero would read as
        "perfectly fresh" to anything reporting a mean, and an FCFS arm
        has no table rather than a maximally fresh one.
        """
        policy = getattr(self._scheduler, "policy", None)
        schedule = getattr(policy, "plex_schedule", None)
        if schedule is None:
            return {}
        age = getattr(schedule, "table_age", None)
        installs = getattr(schedule, "installs", None)
        if age is None or installs is None or installs == 0:
            return {}
        return {
            "table_age_arrivals": {"num": int(age)},
            "table_installs": {"num": int(installs)},
        }

    def _target_facts(self) -> dict[str, Any]:
        scheduler = self._scheduler
        running = getattr(scheduler.running_batch, "reqs", []) or []
        total = int(scheduler.max_total_num_tokens)
        free = int(scheduler.token_to_kv_pool_allocator.available_size())
        max_running = int(scheduler.max_running_requests)
        queued = len(scheduler.waiting_queue)
        step_ms, step_us = self._step_timing()
        pending_decode = sum(
            max(int(getattr(getattr(req, "sampling_params", None),
                            "max_new_tokens", 0) or 0)
                - len(req.output_ids), 0)
            for req in running
        )
        queued_tokens = sum(
            len(req.origin_input_ids) for req in scheduler.waiting_queue
        )
        decoding = sum(1 for req in running if req.output_ids)
        return {
            "queue_depth": {"num": len(scheduler.waiting_queue)},
            **self._table_staleness(),
            "running_requests": {"num": len(running)},
            "batch_size": {"num": len(running)},
            "decode_batch_size": {"num": len(running)},
            "max_batch_size": {"num": max_running},
            # An alias the corpus reads under a second name.
            "max_requests": {"num": max_running},
            "free_decode_slots": {"num": max(max_running - len(running), 0)},
            "pending_decode_tokens": {"num": pending_decode},
            "queued_tokens": {"num": queued_tokens},
            "decoder_ratio_ppm": {
                "num": min(decoding * 1_000_000 // len(running), 1_000_000)
                if running
                else 0
            },
            "kv_overloaded": {"flag": free < total / 10},
            "step_ms": {"num": step_ms},
            # Time per output token, from the observer's own timing. The
            # engine keeps no such number, so this is measured rather
            # than read — and it is the observer's view of a step, which
            # is what a policy asking "how long until my turn" wants.
            "decode_ms_per_token": {"num": step_ms},
            "tpot_us": {"num": step_us},
            # How long a newcomer waits: the requests ahead of it,
            # divided by how many run at once, times a step.
            "estimated_wait_ms": {
                "num": ((queued + max_running - 1) // max_running) * step_ms if max_running else 0
            },
            # The pool in bytes as well as tokens. SGLang's allocator is
            # denominated in tokens, so the conversion needs a per-token
            # size the engine holds and the port does not.
            "memory_capacity": {"num": total * self._bytes_per_token},
            "active_kv_bytes": {"num": (total - free) * self._bytes_per_token},
            "tiers": {"ids": ["gpu"]},
            "throughput_token_cap": {
                "num": int(getattr(scheduler, "max_prefill_tokens", 0) or 0)
            },
            "used_kv_ppm": {
                "num": min((total - free) * 1_000_000 // total, 1_000_000)
                if total
                else 0
            },
            # SGLang's allocator is already denominated in tokens, so
            # unlike vLLM there is no block-size conversion to do here.
            "total_kv_tokens": {"num": total},
            "free_kv_tokens": {"num": free},
            "max_total_tokens": {"num": total},
            # The engine's own count of what it could evict, *before*
            # the page budget truncates the offer.
            #
            # Published so that a narrow offer can be attributed. A
            # policy handed one candidate per step reads identically
            # whether the tree had one to give or the observer chose to
            # publish one, and those are a fact about SGLang and a bug
            # in this file respectively. `RadixCache.evict` builds its
            # heap over exactly this set, so it is the engine's answer
            # to the same question rather than a model of it.
            "evictable_leaves": {"num": len(
                getattr(getattr(self._scheduler, "tree_cache", None),
                        "evictable_leaves", None) or ()
            )},
        }

    def _subjects(self, tracked: list[Req]) -> dict[str, Any]:
        subjects: dict[str, Any] = {
            "request": [req.rid for req in tracked],
            "target": [self._target],
        }
        pages = self._offered_pages()
        if pages:
            subjects["page"] = [page_id for page_id, _ in pages]
        return subjects

    def _offered_pages(self) -> list[tuple[str, Any]]:
        """Radix nodes that are candidates for eviction, coldest first.

        **Offered, not enumerated**, exactly as on vLLM. A snapshot of
        the whole tree is per-node state on every step, which is a
        different order of cost from the per-request scrape, and it hands
        a policy thousands of subjects when a handful are in play.

        `evictable_leaves` ordered by the cache's own
        `eviction_strategy` is SGLang's answer to "what goes next", so
        reading it is reading the answer rather than modelling it.

        Only leaves, and that is the tree's own rule rather than a
        simplification: evicting an interior node orphans everything
        below it, so a policy that could name one could express a plan
        the engine must refuse.
        """
        cache = getattr(self._scheduler, "tree_cache", None)
        leaves = getattr(cache, "evictable_leaves", None)
        if not leaves:
            return []
        strategy = getattr(cache, "eviction_strategy", None)
        nodes = list(leaves)
        if strategy is not None and hasattr(strategy, "get_priority"):
            try:
                nodes.sort(key=strategy.get_priority)
            except (TypeError, ValueError):
                pass
        offered: list[tuple[str, Any]] = []
        for node in nodes[: self._page_budget]:
            node_id = getattr(node, "id", None)
            if node_id is None:
                continue
            offered.append((_page_id(node_id), node))
        return offered

    def _prefix_of(self, node: Any, page_id: str) -> str:
        """The root of this node's branch, as a page id.

        A page id names a page and a page dies when the engine takes it;
        a prefix outlives its pages, so a policy that wants a prefix kept
        names this instead. vLLM has to keep a side table to answer it
        because its pool is flat. The radix tree answers it by walking.
        """
        current = node
        seen = 0
        while seen < 4096:
            parent = getattr(current, "parent", None)
            if parent is None or getattr(parent, "parent", None) is None:
                break
            current = parent
            seen += 1
        node_id = getattr(current, "id", None)
        return _page_id(node_id) if node_id is not None else page_id

    def _wanted_by(self, node: Any) -> list[str]:
        """The requests whose matched prefix runs through this node.

        Demand, not authorship -- the distinction that cost the vLLM port
        three runs. A request's `last_node` is where its match ended, so
        every node on the path from there to the root is a node that
        request is currently relying on.
        """
        wanted = self._branch_demand()
        return sorted(wanted.get(id(node), ()))

    def _branch_demand(self) -> dict[int, set[str]]:
        """Per step, because every offered page asks the same question."""
        if self._demand_step == self._step:
            return self._demand
        demand: dict[int, set[str]] = {}
        batch = getattr(self._scheduler, "running_batch", None)
        pending = list(getattr(batch, "reqs", []) or [])
        pending.extend(getattr(self._scheduler, "waiting_queue", []) or [])
        for req in pending:
            node = getattr(req, "last_node", None)
            rid = getattr(req, "rid", None)
            if node is None or rid is None:
                continue
            seen = 0
            while node is not None and seen < 4096:
                demand.setdefault(id(node), set()).add(rid)
                node = getattr(node, "parent", None)
                seen += 1
        self._demand = demand
        self._demand_step = self._step
        return demand

    def _events(self, departed: list[str], progress: list[list[Any]]) -> dict[str, Any]:
        events: dict[str, Any] = {}
        offered = [page_id for page_id, _ in self._offered_pages()]
        # Once per page, not once per step: every accumulator in the
        # corpus counts one offer as one.
        fresh = [page_id for page_id in offered if page_id not in self._offered_last]
        if fresh:
            events["offered"] = fresh
        self._offered_last = set(offered)
        if progress:
            events["progress"] = progress
        if self._admitted:
            events["admitted"] = list(self._admitted)
        if departed:
            # `completed` is the outcome the contract has for "it ended and
            # nothing said otherwise". The initiator is `host` because the
            # engine ended it: a policy-initiated finish is stage ③ and
            # does not exist yet, and recording one now would attribute the
            # engine's decisions to a component that made none.
            events["finished"] = [[rid, "completed", "host"] for rid in departed]
        return events


def maybe_plex_observer(scheduler: Scheduler) -> PlexObserver | None:
    """Attach an observer if one was asked for, and otherwise cost nothing.

    Environment rather than `server_args`, deliberately. Stage ① changes no
    behaviour, so it needs no place in an argument schema a user has to
    understand — and a feature that is off by default and invisible when off
    is the easiest kind to review.

        SGLANG_PLEX_OBSERVE=/path/to/steps.jsonl

    A path that cannot be opened disables the observer rather than failing
    startup, for the same reason `emit_step` swallows.
    """
    path = os.environ.get("SGLANG_PLEX_OBSERVE")
    if not path:
        return None
    try:
        sink = open(path, "a", encoding="utf-8")  # noqa: SIM115 - lives with the scheduler
    except OSError:
        return None
    target = os.environ.get("SGLANG_PLEX_TARGET", "sglang-0")
    return PlexObserver(scheduler, sink, target)

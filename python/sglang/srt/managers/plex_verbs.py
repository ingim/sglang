"""PLEX v2 stage-3 attach: the verbs, mapped onto SGLang's own primitives.

Stage ③ of the phased attach. A policy stages a verb and the engine either
enacts it through a primitive it already has, or **declines with a reason**.

## What SGLang can and cannot do, and where it differs from vLLM

  finish     `abort_request`, the path the tokenizer already drives.
  move       the radix cache is tiered, so a node can be written to host
             memory and read back. **The one verb SGLang has and vLLM does
             not**, which is why both ports declare their tables rather than
             sharing one.
  pause      declined. This is the interesting one, and it is not obvious.
  prefetch   declined. The tree holds what was computed; a prefix that was
             never produced cannot be made resident without running the
             prefill.
  rebalance  declined. One engine, one target.

### Why `pause` is declined here and native on vLLM

SGLang *does* retract running requests — `ScheduleBatch.retract_decode` — but
it is not the same shape of primitive. It takes the whole batch, chooses its
own victims by its own retraction order, and exists to relieve memory
pressure. There is no "pause this request" entry point, and building one would
mean either reaching into the batch's internals or asking the scheduler to
retract and hoping the request the policy named is among the victims.

Both are worse than declining. The first is the engine-core change Phase 3
forbids; the second is a verb that reports success and does something else,
which is the exact failure `on-refused` exists to prevent — a policy would
learn nothing and its beliefs would silently diverge from the world.

So the port says `unsupported-on-this-engine` and means it. That a policy
portable across both engines gets `pause` on one and not the other is a real
asymmetry, and declaring it is what lets a policy notice rather than
mis-attribute the difference to its own logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


class Refusal(NamedTuple):
    """Why a staged verb did not take effect.

    A closed set, mirroring `wit/io.wit`'s `refusal` variant. Free text
    would let a port invent reasons a policy cannot match on, which is the
    same failure as a fact name nobody publishes.
    """

    verb: str
    subject: str
    reason: str

    def as_dict(self) -> dict[str, str]:
        return {"verb": self.verb, "subject": self.subject, "reason": self.reason}


UNKNOWN_SUBJECT = "unknown-subject"
WRONG_STATE = "wrong-request-state"
UNSUPPORTED = "unsupported-on-this-engine"


class PlexVerbs:
    """Staged verbs, enacted through SGLang's own primitives."""

    def __init__(self, scheduler: Scheduler) -> None:
        self._scheduler = scheduler
        self._refusals: list[Refusal] = []

    def take_refusals(self) -> list[Refusal]:
        """Drain what was refused, for delivery as `on-refused`.

        Drained rather than read: a refusal delivered twice is one
        correction a policy applies twice.
        """
        refusals, self._refusals = self._refusals, []
        return refusals

    def enact(self, verb: str, subject: str, **kwargs: object) -> bool:
        """Enact one verb. Returns whether it took effect.

        The bool is for the caller's accounting; the *reason* goes to the
        policy through `take_refusals`, because a bool cannot say why and
        a policy that cannot tell a lost race from an unsupported verb
        will either retry forever or stop asking when it should not.
        """
        handler = {
            "finish": self._finish,
            "move": self._move,
        }.get(verb)
        if handler is None:
            self._refuse(verb, subject, UNSUPPORTED)
            return False
        return handler(subject, **kwargs)

    def _finish(self, subject: str, **_: object) -> bool:
        req = self._find(subject)
        if req is None:
            self._refuse("finish", subject, UNKNOWN_SUBJECT)
            return False
        if getattr(req, "finished_reason", None) is not None:
            self._refuse("finish", subject, WRONG_STATE)
            return False

        from sglang.srt.managers.io_struct import AbortReq

        self._scheduler.abort_request(AbortReq(rid=subject))
        return True

    def _move(self, subject: str, to: str = "host", **_: object) -> bool:
        """Move a page between the radix cache's tiers.

        Declined unless the tree is hierarchical: a flat radix cache has
        one tier, so there is nowhere to move to, and reporting success
        would tell a policy its page is safe on host memory when it is
        not.
        """
        cache = getattr(self._scheduler, "tree_cache", None)
        writer = getattr(cache, "write_backup", None)
        if writer is None:
            self._refuse("move", subject, UNSUPPORTED)
            return False
        if to != "host":
            # `gpu` would be a promotion, which is `prefetch`'s job and is
            # declined for its own reason. Saying so here keeps the two
            # from being confused.
            self._refuse("move", subject, UNSUPPORTED)
            return False
        writer(subject)
        return True

    def _find(self, subject: str):
        for req in self._scheduler.waiting_queue:
            if req.rid == subject:
                return req
        for req in getattr(self._scheduler.running_batch, "reqs", []) or []:
            if req.rid == subject:
                return req
        return None

    def _refuse(self, verb: str, subject: str, reason: str) -> None:
        self._refusals.append(Refusal(verb, subject, reason))

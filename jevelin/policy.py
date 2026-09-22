"""Deterministic decisions over classifier answers.

The invariant: a classifier answer can only move a turn onto a pre-approved,
more cautious path. Nothing here lets an answer unlock an action the rules
don't already allow, so a missed flag leaves the turn where the rules put it,
and a false alarm costs one extra scripted sentence.
"""

from dataclasses import dataclass

from .executor import TIMEOUT


@dataclass
class Decision:
    action: str            # a script key, or "reply" for generated text
    tier: str              # script | small_llm | large_llm
    why: str
    kind: str              # protective | rule | route | fail_closed


def fired(q, a):
    return a is not None and a.kind in ("noul", "score") and float(a.value) >= q.threshold


def _protective(pack, answers, early):
    hits = [p for p in pack.protective
            if p.flag in answers and fired(pack.inbound[p.flag], answers[p.flag])
            and (p.early or not early)]
    return sorted(hits, key=lambda p: p.severity)


def _rules(pack, d, ctx):
    """First matching rule wins. "*" matches any action, including a fail-closed hold."""
    for r in pack.rules:
        if r["if_action"] in ("*", d.action) and ctx.get(r["if_context"]):
            return Decision(r["then"], "script", r["why"], "rule")
    return d


def decide(pack, answers, ctx, route=True):
    """Classifier-driven decision, then deterministic rules on top. Rules need no model."""
    return _rules(pack, _decide(pack, answers, ctx, route), ctx)


def _decide(pack, answers, ctx, route):
    if answers is None or answers is TIMEOUT:
        return Decision("hold", "script", "classifier unavailable, fail closed", "fail_closed")

    hits = _protective(pack, answers, early=False)
    if hits:
        p = hits[0]
        return Decision(p.action, "script", "%s fired" % p.flag, "protective")

    if not route:
        return Decision("reply", pack.route["default_tier"], "no router", "route")

    a = answers[pack.route["question"]]
    if a.confidence is not None and a.confidence < pack.route["confidence_floor"]:
        return Decision("reply", "large_llm", "low routing confidence %.2f" % a.confidence, "route")
    if a.value in pack.scripts:
        return Decision(a.value, "script", "routine: %s" % a.value, "route")
    return Decision("reply", pack.route["tiers"].get(a.value, "large_llm"), "%s turn" % a.value, "route")


def early_protect(pack, answers):
    """Protective flags allowed to act on a partial transcript. Caution only, never routing."""
    if answers is None or answers is TIMEOUT:
        return None
    hits = _protective(pack, answers, early=True)
    return hits[0] if hits else None


def guard_blocks(pack, answers):
    if answers is TIMEOUT:
        return True, "guard unavailable, fail closed"
    for k, a in answers.items():
        if fired(pack.guard[k], a):
            return True, "%s fired" % k
    return False, ""

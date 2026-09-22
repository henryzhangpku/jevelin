"""Two ways to run one caller turn.

baseline   The common shape today. Wait for the turn to end, ask an LLM to
           judge it, then generate a reply with the large model.

fast       A System One classifier reads the transcript while the caller is
           still talking. Routine turns get a pre-approved script, and the rest
           go to the smallest model that fits. Reply generation starts early
           on a guess from the partial transcript. Every generated draft passes
           an output guard before anyone hears it.
"""

import hashlib
import json
from dataclasses import dataclass, field

from .backends import keyed_rng
from .calls import context, timing
from .executor import TIMEOUT
from .policy import decide, early_protect, guard_blocks


@dataclass
class TurnResult:
    pipeline: str
    call: int
    turn: int
    kind: str
    text: str
    decision: object
    ttfa: float                   # ms from end of caller speech to first agent audio
    reply: str
    protect_ms: float = None      # when a protective action was decided
    early_protect: bool = False
    guard_blocked: bool = False
    risky_spoken: bool = False
    fail_closed: bool = False
    stages: list = field(default_factory=list)
    timing: object = None


class Env:
    def __init__(self, pack, profile, system_one, judge, llm, tts, ledger=None):
        self.pack, self.profile = pack, profile
        self.system_one, self.judge, self.llm, self.tts = system_one, judge, llm, tts
        self.ledger = ledger if ledger is not None else []


# --------------------------------------------------------------------------
# stage functions: each takes a clock and returns (result, [Usage])
# --------------------------------------------------------------------------

def _classify(backend, state, questions, key):
    async def fn(clock):
        return await backend.classify(clock, state, questions, key)
    return fn


def _reply(env, tier, state, key):
    async def fn(clock):
        return await env.llm.reply(clock, tier, state, key)
    return fn


def _speak(env, key):
    async def fn(clock):
        return await env.tts.speak(clock, key)
    return fn


def _policy(env, answers, ctx, route, key):
    async def fn(clock):
        await clock.sleep(env.profile.latency["policy"].sample(keyed_rng(key, "policy")))
        return decide(env.pack, answers, ctx, route=route), []
    return fn


def _log(env, pipeline, call, i, h, state, questions):
    ans = h.result
    env.ledger.append({
        "pipeline": pipeline, "call": call.id, "turn": i, "stage": h.name,
        "model": h.usage[0].model if h.usage else None,
        "state_sha256": hashlib.sha256(state.encode("utf-8")).hexdigest(),
        "questions": sorted(questions),
        "answers": None if ans is TIMEOUT else dict(
            (k, {"value": a.value, "probs": a.probs}) for k, a in ans.items()),
        "start_ms": round(h.start, 1), "end_ms": round(h.end, 1),
        "timed_out": h.timed_out, "error": h.error,
    })


# --------------------------------------------------------------------------
# pipelines
# --------------------------------------------------------------------------

async def baseline_turn(ex, env, call, i, turn, history):
    pack, T = env.pack, timing(turn, env.profile)
    key = "c%dt%d" % (call.id, i)
    ex.begin_turn(T.speech_ms)
    state = context(history, turn.text)

    judge = ex.spawn("llm judge", _classify(env.judge, state, pack.inbound, key), at=T.vad)
    answers = await judge.get()
    _log(env, "baseline", call, i, judge, state, pack.inbound)

    pol = ex.spawn("policy", _policy(env, answers, call.ctx, False, key), after=[judge])
    dec = await pol.get()

    risky = False
    if dec.tier == "script":
        reply = pack.scripts[dec.action]
        speak = ex.spawn("tts", _speak(env, key), after=[pol])
    else:
        gen = ex.spawn(dec.tier, _reply(env, dec.tier, state, key), after=[pol])
        reply, risky = await gen.get()
        speak = ex.spawn("tts", _speak(env, key), after=[gen])
    await speak.get()
    await ex.settle()

    protective = dec.kind in ("protective", "rule")
    return TurnResult("baseline", call.id, i, turn.kind, turn.text, dec, speak.end, reply,
                      protect_ms=pol.end if protective else None,
                      risky_spoken=risky, fail_closed=dec.kind == "fail_closed",
                      stages=list(ex.stages), timing=T)


async def fast_turn(ex, env, call, i, turn, history):
    pack, T, to = env.pack, timing(turn, env.profile), env.profile.timeouts
    key = "c%dt%d" % (call.id, i)
    ex.begin_turn(T.speech_ms)
    partial_state = context(history, T.partial_text)
    state = context(history, turn.text)

    # Both classifier passes are in flight before the turn is even over.
    pj = ex.spawn("jev · partial", _classify(env.system_one, partial_state, pack.inbound, key + "p"),
                  at=T.partial_at, timeout=to["jev_inbound"])
    fj = ex.spawn("jev · final", _classify(env.system_one, state, pack.inbound, key + "f"),
                  at=T.asr_final, timeout=to["jev_inbound"])

    p_ans = await pj.get()
    _log(env, "fast", call, i, pj, partial_state, pack.inbound)
    early = early_protect(pack, p_ans)          # caution may act on a partial; routing may not
    protect_ms = pj.end if early else None

    # Speculate only on a confident guess: lower thresholds buy latency with wasted tokens.
    guess = None
    if p_ans is not TIMEOUT:
        conf = p_ans[pack.route["question"]].confidence or 0.0
        if conf >= pack.route.get("speculate_min_confidence", 0.0):
            guess = decide(pack, p_ans, call.ctx).tier
    spec = None
    if guess in ("small_llm", "large_llm"):
        spec = ex.spawn(guess + " · speculative", _reply(env, guess, state, key),
                        after=[pj], at=T.asr_final)

    f_ans = await fj.get()
    _log(env, "fast", call, i, fj, state, pack.inbound)
    pol = ex.spawn("policy", _policy(env, f_ans, call.ctx, True, key), after=[fj], at=T.vad)
    dec = await pol.get()
    if protect_ms is None and dec.kind in ("protective", "rule"):
        protect_ms = pol.end

    blocked = risky = False
    fail_closed = dec.kind == "fail_closed"
    if dec.tier == "script":
        if spec:
            spec.discard()
        reply = pack.scripts[dec.action]
        speak = ex.spawn("tts", _speak(env, key), after=[pol])
    else:
        if spec is not None and guess == dec.tier:
            gen = spec
        else:
            if spec:
                spec.discard()
            gen = ex.spawn(dec.tier, _reply(env, dec.tier, state, key), after=[pol])
        draft, risky = await gen.get()
        guard = ex.spawn("jev · guard", _classify(env.system_one, draft, pack.guard, key + "g"),
                         after=[gen, pol], timeout=to["jev_guard"])
        g = await guard.get()
        _log(env, "fast", call, i, guard, draft, pack.guard)
        blocked, _ = guard_blocks(pack, g)
        fail_closed = fail_closed or g is TIMEOUT
        reply = pack.scripts["safe_fallback"] if blocked else draft
        risky = risky and not blocked
        speak = ex.spawn("tts · safe fallback" if blocked else "tts", _speak(env, key), after=[guard])
    await speak.get()
    await ex.settle()

    return TurnResult("fast", call.id, i, turn.kind, turn.text, dec, speak.end, reply,
                      protect_ms=protect_ms, early_protect=early is not None,
                      guard_blocked=blocked, risky_spoken=risky, fail_closed=fail_closed,
                      stages=list(ex.stages), timing=T)


PIPELINES = {"baseline": baseline_turn, "fast": fast_turn}


async def run_call(ex, env, call, pipeline):
    history, out = [], []
    for i, turn in enumerate(call.turns):
        r = await PIPELINES[pipeline](ex, env, call, i, turn, history)
        history += ["caller: " + turn.text, "agent: " + r.reply]
        out.append(r)
    return out


def dump_ledger(ledger, path):
    with open(path, "w", encoding="utf-8") as fh:
        for row in ledger:
            fh.write(json.dumps(row) + "\n")

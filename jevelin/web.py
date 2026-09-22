"""Entry point for the browser demo: settings in, plain JSON-ready data out.

The page runs this module under Pyodide, so the numbers on the page come from
the same engine as the command line.
"""

import copy
import math

from .backends import MockLLM, MockLLMJudge, MockSystemOne, MockTTS
from .calls import make_calls
from .executor import SimExecutor
from .pack import Pack
from .pipelines import Env, run_call
from .profile import Profile
from .report import summarize

DETAIL_CALLS = 40          # calls whose per-stage timelines are sent to the page
TAIL = {"jev": 1.6, "llm_judge": 1.75, "small_llm": 1.83, "large_llm": 2.0}


def _clean(v):
    if isinstance(v, float):
        return None if math.isnan(v) else round(v, 4)
    if isinstance(v, dict):
        return dict((k, _clean(x)) for k, x in v.items())
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


def _kind(name):
    if name.startswith("jev"):
        return "classifier"
    if name.startswith("tts"):
        return "speech"
    if name == "policy":
        return "policy"
    return "llm"


def _turn(r):
    return {
        "ttfa": r.ttfa,
        "action": r.decision.action,
        "why": r.decision.why,
        "decision_kind": r.decision.kind,
        "tier": r.decision.tier,
        "reply": r.reply,
        "protect_ms": r.protect_ms,
        "early_protect": r.early_protect,
        "guard_blocked": r.guard_blocked,
        "risky_spoken": r.risky_spoken,
        "fail_closed": r.fail_closed,
        "stages": [{
            "name": s.name, "kind": _kind(s.name),
            "start": round(s.start, 1), "end": round(s.end, 1),
            "wasted": s.wasted, "timed_out": s.timed_out,
            "tokens": sum(u.input + u.output for u in s.usage),
        } for s in sorted(r.stages, key=lambda s: (s.start, s.end))],
    }


def apply_settings(pack_raw, profile_raw, s):
    pack_raw, profile_raw = copy.deepcopy(pack_raw), copy.deepcopy(profile_raw)
    lat = profile_raw["latency_ms"]
    for key, tail in TAIL.items():
        if key in s:
            lat[key] = {"p50": float(s[key]), "p95": float(s[key]) * tail}
    if "timeout" in s:
        profile_raw["timeouts_ms"]["jev_inbound"] = float(s["timeout"])
        profile_raw["timeouts_ms"]["jev_guard"] = float(s["timeout"])
    if "speculate_min" in s:
        pack_raw["route"]["speculate_min_confidence"] = float(s["speculate_min"])
    if "confidence_floor" in s:
        pack_raw["route"]["confidence_floor"] = float(s["confidence_floor"])
    return Pack(pack_raw), Profile(profile_raw)


async def simulate(pack_raw, profile_raw, settings):
    pack, profile = apply_settings(pack_raw, profile_raw, settings)
    seed = int(settings.get("seed", 7))
    env = Env(pack, profile, MockSystemOne(profile, seed), MockLLMJudge(profile, pack, seed),
              MockLLM(profile, pack, seed), MockTTS(profile, seed))
    calls = make_calls(pack, int(settings.get("calls", 300)), seed)

    runs = {}
    for pipeline in ("baseline", "fast"):
        ex, results = SimExecutor(), []
        for call in calls:
            results.append(await run_call(ex, env, call, pipeline))
        runs[pipeline] = results

    detail = []
    for call, base, fast in zip(calls[:DETAIL_CALLS], runs["baseline"], runs["fast"]):
        detail.append({
            "id": call.id,
            "ctx": [k for k, v in call.ctx.items() if v],
            "turns": [{
                "text": t.text, "kind": t.kind, "speech_ms": b.timing.speech_ms,
                "baseline": _turn(b), "fast": _turn(f),
            } for t, b, f in zip(call.turns, base, fast)],
        })

    flat = dict((p, [r for call in rs for r in call]) for p, rs in runs.items())
    return _clean({
        "budget_ms": profile.budget_ms,
        "turns": len(flat["fast"]),
        "calls": len(calls),
        "summary": dict((p, summarize(rs, profile)) for p, rs in flat.items()),
        "ttfa": dict((p, sorted(round(r.ttfa, 1) for r in rs)) for p, rs in flat.items()),
        "detail": detail,
    })

"""Command line entry point.

  python -m jevelin compare               # 300 simulated calls, both pipelines
  python -m jevelin trace --call 3        # per-turn timelines for one call
  python -m jevelin live --call 3         # run one call concurrently in real time
  python -m jevelin compare --classifier jev   # real Jev for every classifier stage
  python -m jevelin compare --classifier laya  # Laya, local or via LAYA_BASE_URL
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from .backends import (JevSystemOne, LayaSystemOne, MockLLM, MockLLMJudge,
                       MockSystemOne, MockTTS)
from .calls import make_calls
from .executor import LiveExecutor, SimExecutor
from .pack import Pack
from .pipelines import Env, dump_ledger, run_call
from .profile import Profile
from .report import comparison, gantt, summarize

ROOT = Path(__file__).resolve().parent.parent


def _resolve(value, folder):
    p = Path(value)
    if p.exists():
        return p
    named = ROOT / folder / (value + ".json")
    if named.exists():
        return named
    sys.exit("not found: %s (looked in %s too)" % (value, ROOT / folder))


def build_env(args):
    pack = Pack.load(_resolve(args.pack, "packs"))
    profile = Profile.load(_resolve(args.profile, "profiles"))
    if args.classifier == "jev":
        if not os.environ.get("TYPESAFE_API_KEY"):
            sys.exit("--classifier jev needs TYPESAFE_API_KEY (https://console.typesafe.ai/keys)")
        system_one = JevSystemOne()
    elif args.classifier == "laya":
        system_one = LayaSystemOne()
    else:
        system_one = MockSystemOne(profile, args.seed)
    if getattr(args, "speculate_min", None) is not None:
        pack.route["speculate_min_confidence"] = args.speculate_min
    return Env(pack, profile, system_one,
               MockLLMJudge(profile, pack, args.seed),
               MockLLM(profile, pack, args.seed),
               MockTTS(profile, args.seed))


async def _run(env, calls, pipeline, live=False, on_turn=None):
    ex = LiveExecutor() if live else SimExecutor()
    results = []
    for call in calls:
        for r in await run_call(ex, env, call, pipeline):
            results.append(r)
            if on_turn:
                on_turn(r)
    return results


def cmd_compare(args):
    env = build_env(args)
    calls = make_calls(env.pack, args.calls, args.seed)
    base = asyncio.run(_run(env, calls, "baseline"))
    fast = asyncio.run(_run(env, calls, "fast"))
    turns = sum(len(c.turns) for c in calls)
    print("pack: %s | %d calls, %d caller turns | classifier: %s | seed %d\n"
          % (env.pack.name, len(calls), turns, args.classifier, args.seed))
    print(comparison(summarize(base, env.profile), summarize(fast, env.profile), env.profile))
    print("\nLatencies and prices come from the profile (%s). They are illustrative"
          "\nuntil replaced with measurements." % Path(_resolve(args.profile, "profiles")).name)
    if args.ledger:
        dump_ledger(env.ledger, args.ledger)
        print("decision ledger: %s (%d rows)" % (args.ledger, len(env.ledger)))


def _pick_call(args, env):
    calls = make_calls(env.pack, args.call + 1, args.seed)
    return calls[args.call]


def cmd_trace(args):
    env = build_env(args)
    call = _pick_call(args, env)
    print("call %d | context: %s\n" % (call.id, ", ".join(k for k, v in call.ctx.items() if v) or "none"))
    runs = dict((p, asyncio.run(_run(env, [call], p))) for p in ("baseline", "fast"))
    for i in range(len(call.turns)):
        if args.turn is not None and i != args.turn:
            continue
        for p in (["baseline", "fast"] if args.pipeline == "both" else [args.pipeline]):
            print(gantt(runs[p][i]))
            print()


def cmd_live(args):
    env = build_env(args)
    call = _pick_call(args, env)
    print("call %d, fast path, running in real time\n" % call.id)

    def show(r):
        print(gantt(r))
        print()
    asyncio.run(_run(env, [call], "fast", live=True, on_turn=show))


def main(argv=None):
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(prog="jevelin")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("compare", "trace", "live"):
        p = sub.add_parser(name)
        p.add_argument("--pack", default="support", help="pack name in packs/ or a path")
        p.add_argument("--profile", default="default", help="profile name in profiles/ or a path")
        p.add_argument("--classifier", choices=["mock", "jev", "laya"], default="mock")
        p.add_argument("--seed", type=int, default=7)
        p.add_argument("--speculate-min", type=float, dest="speculate_min",
                       help="only start speculative generation above this routing confidence (0 = always)")
        if name == "compare":
            p.add_argument("--calls", type=int, default=300)
            p.add_argument("--ledger", help="write the decision ledger as JSONL")
        else:
            p.add_argument("--call", type=int, default=0)
        if name == "trace":
            p.add_argument("--turn", type=int)
            p.add_argument("--pipeline", choices=["both", "baseline", "fast"], default="both")
    args = ap.parse_args(argv)
    {"compare": cmd_compare, "trace": cmd_trace, "live": cmd_live}[args.cmd](args)


if __name__ == "__main__":
    main()

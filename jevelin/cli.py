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
import textwrap
from pathlib import Path

from .backends import (JevSystemOne, LayaSystemOne, MockLLM, MockLLMJudge,
                       MockSystemOne, MockTTS)
from .calls import make_calls
from .executor import LiveExecutor, SimExecutor
from .pack import Pack
from .pipelines import Env, dump_ledger, run_call
from .profile import Profile
from .report import comparison, gantt, summarize, tint

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
    print(tint("pack: %s | %d calls, %d caller turns | classifier: "
            % (env.pack.name, len(calls), turns), "dim")
          + tint(args.classifier, "bold", "cyan")
          + tint(" | seed %d" % args.seed, "dim") + "\n")
    print(comparison(summarize(base, env.profile), summarize(fast, env.profile), env.profile))
    print(tint("\nLatencies and prices come from the profile (%s). They are illustrative"
            "\nuntil replaced with measurements."
            % Path(_resolve(args.profile, "profiles")).name, "dim"))
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
            print(gantt(runs[p][i], budget_ms=env.profile.budget_ms))
            print()


def cmd_live(args):
    env = build_env(args)
    call = _pick_call(args, env)
    print("call %d, fast path, running in real time\n" % call.id)

    def show(r):
        print(gantt(r, budget_ms=env.profile.budget_ms))
        print()
    asyncio.run(_run(env, [call], "fast", live=True, on_turn=show))


# ---------------------------------------------------------------------------
# the guided demo: one command instead of six remembered ones
# ---------------------------------------------------------------------------

DEMO_BAR = "=" * 74


def _say(lines):
    for i, line in enumerate(lines):
        label = "  SAY   " if i == 0 else "        "
        print(tint(label, "bold", "yellow") + tint(line, "yellow"))
    print()


def _head(n, total, title, lines):
    print()
    print(tint(DEMO_BAR, "dim"))
    print(tint("  STEP %d/%d" % (n, total), "bold", "green") + tint("  " + title, "bold"))
    print(tint(DEMO_BAR, "dim"))
    print()
    if lines:
        _say(lines)


def _pause(nxt, total, nopause):
    print()
    if nopause or not sys.stdin.isatty():
        return
    try:
        input(tint("        -- Enter for step %d/%d, Ctrl-C to stop --" % (nxt, total), "dim"))
    except (EOFError, KeyboardInterrupt):
        print()
        raise SystemExit(0)


def _by_turn(results):
    return dict(((r.call, r.turn), r) for r in results)


def cmd_demo(args):
    """Run the whole interview demo in order, pausing between steps.

    Nothing here is pack-specific. The interesting turn in step 2 is FOUND
    rather than hardcoded, so this works on any pack, including one written
    after this code.
    """
    env = build_env(args)
    budget = env.profile.budget_ms
    total, nopause = 6, args.no_pause
    calls = make_calls(env.pack, args.calls, args.seed)
    only = args.step

    def want(n):
        return only is None or only == n

    base = asyncio.run(_run(env, calls, "baseline"))
    fast = asyncio.run(_run(env, calls, "fast"))
    sb, sf = summarize(base, env.profile), summarize(fast, env.profile)
    bmap, fmap = _by_turn(base), _by_turn(fast)

    # ---- 1. the headline -------------------------------------------------
    if want(1):
        _head(1, total, "The headline number", [
            "These are simulated from an illustrative latency and price profile,",
            "not production traffic. The profile is a JSON file, so put your",
            "numbers in it and the same code gives your numbers.",
            "",
            "The baseline judge is assumed exactly as accurate as the classifier,",
            "so this isolates latency and cost. Accuracy is a separate measurement,",
            "and your QA-scored calls are the right dataset for it.",
        ])
        print("  pack: %s | %d calls, %d caller turns"
              % (env.pack.name, len(calls), sum(len(c.turns) for c in calls)))
        print()
        print(comparison(sb, sf, env.profile))
        _pause(2, total, nopause)

    # ---- 2. the earliest protective decision -----------------------------
    early = [r for r in fast
             if getattr(r, "early_protect", False) and r.protect_ms is not None]
    pick = min(early, key=lambda r: r.protect_ms) if early else None
    if want(2):
        if pick is None:
            _head(2, total, "Protective decisions", [
                "No early protective decision in this run. Raise --calls, or check",
                "that some protective entries in the pack are marked early.",
            ])
        else:
            lead = abs(pick.protect_ms) / 1000.0
            quote = textwrap.wrap('The caller says: "%s"' % pick.text, 70,
                                  subsequent_indent="  ")
            _head(2, total, "Caught before the caller finished", quote + [
                "",
                "The flag fires about %.2f seconds BEFORE they stop talking, so the" % lead,
                "agent is already committed to the protective script before the",
                "sentence is finished. That is the one thing a baseline cannot buy",
                "at any price, because it needs the partial transcript.",
            ])
            b = bmap.get((pick.call, pick.turn))
            if b is not None:
                print(gantt(b, budget_ms=budget))
                print()
            print(gantt(pick, budget_ms=budget))
        _pause(3, total, nopause)

    # ---- 3. a whole call, turn by turn -----------------------------------
    if want(3):
        _head(3, total, "One call, turn by turn", [
            "Read left to right. The partial pass runs while they are still",
            "talking, and the final pass starts just after they stop. Routine",
            "turns go straight to a script with no model call at all.",
            "",
            "Any ~~~ bar is a guess from the partial transcript that turned out",
            "wrong. I paid tokens for it, which is the next step.",
        ])
        first = calls[0]
        for i in range(len(first.turns)):
            for m in (bmap, fmap):
                r = m.get((first.id, i))
                if r is not None:
                    print(gantt(r, budget_ms=budget))
                    print()
        _pause(4, total, nopause)

    # ---- 4. the knob -----------------------------------------------------
    if want(4):
        _head(4, total, "The speculation knob", [
            "Speculating on the partial guess buys latency and wastes tokens when",
            "the guess is wrong. It is a business decision, and making it one",
            "flag is the point.",
        ])
        tight = build_env(args)
        tight.pack.route["speculate_min_confidence"] = 0.8
        tcalls = make_calls(tight.pack, args.calls, args.seed)
        tf = summarize(asyncio.run(_run(tight, tcalls, "fast")), tight.profile)
        rows = [
            ("wasted on wrong guesses", "%.1f%%" % (100 * sf["wasted_share"]),
             "%.1f%%" % (100 * tf["wasted_share"])),
            ("time to first audio, p95", "%d ms" % round(sf["ttfa_p95"]),
             "%d ms" % round(tf["ttfa_p95"])),
            ("model cost per call", "$%.4f" % sf["cost_per_call"],
             "$%.4f" % tf["cost_per_call"]),
        ]
        w = max(len(r[0]) for r in rows) + 2
        print("  " + "".ljust(w) + tint("%14s" % "speculate always", "bold", "grey")
              + tint("%16s" % "only above 0.8", "bold", "green"))
        for a, x, y in rows:
            print("  " + a.ljust(w) + tint("%14s" % x, "grey")
                  + tint("%16s" % y, "green"))
        _pause(5, total, nopause)

    # ---- 5. rules that need no model -------------------------------------
    if want(5):
        _head(5, total, "Deterministic rules outrank the model", [
            "Account state is not a classifier output. These rules apply whatever",
            "the model says, and they still hold when the classifier is timing out.",
        ])
        if env.pack.rules:
            for r in env.pack.rules:
                print("  " + tint("%-20s" % r.get("if_context", "*"), "cyan")
                      + tint("%-24s" % ("on " + str(r.get("if_action", "*"))), "dim")
                      + tint("-> " + r["then"], "bold"))
                if r.get("why"):
                    print("  " + tint(" " * 44 + r["why"], "dim"))
        else:
            print("  (this pack declares no deterministic rules)")
        print()
        print(tint("  covered by test_wildcard_rules_hold_even_when_the_classifier_is_down",
                   "dim"))
        _pause(6, total, nopause)

    # ---- 6. real concurrency ---------------------------------------------
    if want(6):
        _head(6, total, "The same code, in real time", [
            "Everything above was simulated time. This is the identical pipeline",
            "run concurrently against the wall clock, so the shape is not an",
            "artefact of the simulator.",
        ])

        def show(r):
            print(gantt(r, budget_ms=budget))
            print()

        asyncio.run(_run(env, [calls[0]], "fast", live=True, on_turn=show))
        print(tint("  Done.", "bold", "green"))
        print()


def main(argv=None):
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(prog="jevelin")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("compare", "trace", "live", "demo"):
        p = sub.add_parser(name)
        p.add_argument("--pack", default="support", help="pack name in packs/ or a path")
        p.add_argument("--profile", default="default", help="profile name in profiles/ or a path")
        p.add_argument("--classifier", choices=["mock", "jev", "laya"], default="mock")
        p.add_argument("--seed", type=int, default=7)
        p.add_argument("--speculate-min", type=float, dest="speculate_min",
                       help="only start speculative generation above this routing confidence (0 = always)")
        if name in ("compare", "demo"):
            p.add_argument("--calls", type=int, default=300)
        if name == "compare":
            p.add_argument("--ledger", help="write the decision ledger as JSONL")
        if name in ("trace", "live"):
            p.add_argument("--call", type=int, default=0)
        if name == "trace":
            p.add_argument("--turn", type=int)
            p.add_argument("--pipeline", choices=["both", "baseline", "fast"], default="both")
        if name == "demo":
            p.add_argument("--no-pause", action="store_true", dest="no_pause",
                           help="do not wait for Enter between steps")
            p.add_argument("--step", type=int, choices=range(1, 7), metavar="N",
                           help="run only step N (1-6)")
    args = ap.parse_args(argv)
    {"compare": cmd_compare, "trace": cmd_trace,
     "live": cmd_live, "demo": cmd_demo}[args.cmd](args)


if __name__ == "__main__":
    main()

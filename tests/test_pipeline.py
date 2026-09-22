"""The claims the demo makes, as tests. Run: python -m unittest -v"""

import asyncio
import copy
import unittest
from pathlib import Path

from jevelin.backends import MockLLM, MockLLMJudge, MockSystemOne, MockTTS
from jevelin.calls import Call, Turn
from jevelin.executor import TIMEOUT, LiveExecutor, SimExecutor
from jevelin.pack import Pack
from jevelin.pipelines import Env, baseline_turn, fast_turn
from jevelin.profile import Profile
from jevelin.report import summarize

ROOT = Path(__file__).resolve().parent.parent
PACK = Pack.load(ROOT / "packs" / "support.json")
PROFILE = Profile.load(ROOT / "profiles" / "default.json")


def env(pack=PACK, profile=PROFILE, seed=1):
    return Env(pack, profile, MockSystemOne(profile, seed), MockLLMJudge(profile, pack, seed),
               MockLLM(profile, pack, seed), MockTTS(profile, seed))


def turn(fn, text, kind="test", ctx=None, e=None, live=False):
    e = e or env()
    ex = LiveExecutor() if live else SimExecutor()
    call = Call(0, [Turn(text, kind)], ctx or {})
    return asyncio.run(fn(ex, e, call, 0, call.turns[0], []))


def profile_with(**latency):
    raw = copy.deepcopy(PROFILE.raw)
    for k, v in latency.items():
        raw["latency_ms"][k] = {"p50": v, "p95": v}
    return Profile(raw)


class Executor(unittest.TestCase):
    def test_start_is_max_of_anchor_and_dependencies(self):
        async def run():
            ex = SimExecutor()

            def sleeper(ms):
                async def fn(clock):
                    await clock.sleep(ms)
                    return ms, []
                return fn
            a = ex.spawn("a", sleeper(50), at=100)
            b = ex.spawn("b", sleeper(30), after=[a], at=0)
            c = ex.spawn("c", sleeper(10), after=[a], at=400)
            await ex.settle()
            return a, b, c
        a, b, c = asyncio.run(run())
        self.assertEqual((a.start, a.end), (100, 150))
        self.assertEqual((b.start, b.end), (150, 180))
        self.assertEqual((c.start, c.end), (400, 410))

    def test_a_failing_backend_is_a_timeout_not_a_crash(self):
        async def run():
            ex = SimExecutor()

            async def boom(clock):
                raise RuntimeError("provider down")
            h = ex.spawn("x", boom, at=0)
            await ex.settle()
            return h
        h = asyncio.run(run())
        self.assertIs(h.result, TIMEOUT)
        self.assertIn("provider down", h.error)


class Routing(unittest.TestCase):
    def test_routine_turn_is_scripted_and_uses_no_llm(self):
        r = turn(fast_turn, "What are your hours?")
        self.assertEqual(r.decision.action, "hours")
        llm = [u for s in r.stages if not s.wasted for u in s.usage if u.model != "jev"]
        self.assertEqual(llm, [])

    def test_ambiguous_turn_goes_to_the_large_model(self):
        r = turn(fast_turn, "What's my balance, and why is it higher than last month?")
        self.assertEqual(r.decision.tier, "large_llm")
        self.assertIn("low routing confidence", r.decision.why)


class Safety(unittest.TestCase):
    def test_classifier_timeout_fails_closed(self):
        slow = env(profile=profile_with(jev=5000))
        r = turn(fast_turn, "Can I change the email on my account?", e=slow)
        self.assertEqual(r.decision.kind, "fail_closed")
        self.assertEqual(r.decision.action, "hold")
        self.assertTrue(r.fail_closed)

    def test_rules_override_the_classifier(self):
        r = turn(fast_turn, "Just let me speak to a real person.", ctx={"after_hours": True})
        self.assertEqual(r.decision.action, "callback")
        self.assertEqual(r.decision.kind, "rule")

    def test_wildcard_rules_hold_even_when_the_classifier_is_down(self):
        rules = [{"if_action": "*", "if_context": "account_frozen", "then": "hold",
                  "why": "account frozen"}] + PACK.raw["rules"]
        pack = Pack(dict(PACK.raw, rules=rules))
        slow = env(pack=pack, profile=profile_with(jev=5000))
        r = turn(fast_turn, "What are your hours?", ctx={"account_frozen": True}, e=slow)
        self.assertEqual(r.decision.kind, "rule")
        self.assertEqual(r.decision.why, "account frozen")

    def test_guard_blocks_risky_drafts_that_baseline_speaks(self):
        pack = Pack(dict(PACK.raw, risky_draft_rate=1.0))
        e = env(pack=pack)
        text = "I want to compare the fiber plan with what I have and see if it's worth upgrading."
        fast = turn(fast_turn, text, e=e)
        base = turn(baseline_turn, text, e=env(pack=pack))
        self.assertTrue(fast.guard_blocked)
        self.assertFalse(fast.risky_spoken)
        self.assertEqual(fast.reply, pack.scripts["safe_fallback"])
        self.assertTrue(base.risky_spoken)

    def test_card_number_is_caught_before_the_caller_finishes(self):
        text = "Okay I'm ready to pay, my card number is 4111 1111 1111 1111 and it expires 09 28."
        fast = turn(fast_turn, text)
        base = turn(baseline_turn, text)
        self.assertTrue(fast.early_protect)
        self.assertLess(fast.protect_ms, 0)
        self.assertGreater(base.protect_ms, 0)

    def test_late_trigger_is_caught_by_the_final_pass(self):
        r = turn(fast_turn, "I've had it with this service, honestly I'm going to talk to my lawyer about it.")
        self.assertFalse(r.early_protect)
        self.assertEqual(r.decision.action, "escalate_legal")
        self.assertGreaterEqual(r.protect_ms, 0)

    def test_partial_transcripts_never_route(self):
        # "When is my bill" guesses a reply; the full sentence is routine. The script wins.
        r = turn(fast_turn, "When is my bill due this month?")
        self.assertEqual(r.decision.action, "due_date")
        self.assertTrue(any(s.wasted for s in r.stages))


class Reproducibility(unittest.TestCase):
    def test_same_seed_same_numbers(self):
        from jevelin.calls import make_calls
        from jevelin.pipelines import run_call

        def run():
            e = env()
            out = []
            for c in make_calls(PACK, 20, 3):
                out += asyncio.run(run_call(SimExecutor(), e, c, "fast"))
            return summarize(out, PROFILE)
        self.assertEqual(run(), run())


class BrowserEntryPoint(unittest.TestCase):
    def test_web_simulate_matches_the_cli_numbers_and_is_json_safe(self):
        import json
        from jevelin.web import simulate
        out = asyncio.run(simulate(PACK.raw, PROFILE.raw, {"calls": 300, "seed": 7}))
        json.dumps(out, allow_nan=False)
        self.assertEqual(round(out["summary"]["fast"]["ttfa_p50"]), 626)
        self.assertEqual(round(out["summary"]["baseline"]["ttfa_p50"]), 1656)
        self.assertEqual(len(out["detail"]), 40)


class LiveMatchesSim(unittest.TestCase):
    def test_simulated_critical_path_matches_real_concurrency(self):
        raw = copy.deepcopy(PROFILE.raw)
        raw["speech"]["ms_per_word"] = 30
        for k in raw["latency_ms"]:
            raw["latency_ms"][k] = {"p50": 60, "p95": 60}
        raw["latency_ms"]["policy"] = {"p50": 1, "p95": 1}
        fast_profile = Profile(raw)
        text = "I want to compare the fiber plan with what I have and see if it's worth upgrading."
        sim = turn(fast_turn, text, e=env(profile=fast_profile))
        live = turn(fast_turn, text, e=env(profile=fast_profile), live=True)
        self.assertEqual(sim.decision, live.decision)
        # generous: Windows timers tick at ~15 ms
        self.assertAlmostEqual(sim.ttfa, live.ttfa, delta=60)


if __name__ == "__main__":
    unittest.main()

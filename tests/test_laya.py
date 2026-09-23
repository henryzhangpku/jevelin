"""The Laya backend, tested without the package, a GPU, or a network.

Everything here exercises the two halves that can actually be wrong: the
request we build, and the reply we read. The transport is stubbed, because a
test that needs a model download is a test nobody runs.
"""

import asyncio
import unittest
from pathlib import Path

from jevelin.backends import Answer, LayaSystemOne, SimClock
from jevelin.pack import Pack

ROOT = Path(__file__).resolve().parent.parent
PACK = Pack.load(ROOT / "packs" / "support.json")


def bare(**env):
    """A backend with no transport wired, for testing shaping in isolation."""
    obj = LayaSystemOne.__new__(LayaSystemOne)
    obj.base_url = env.get("base_url")
    obj.api_key = env.get("api_key")
    obj.model_name = env.get("model_name", "convaiinnovations/laya")
    obj.timeout = 10.0
    obj.router = None
    return obj


class Shaping(unittest.TestCase):
    def test_pack_questions_need_no_translation(self):
        """jevelin's pack format IS Laya's question format. Renaming only."""
        payload = LayaSystemOne.as_payload(PACK.inbound)
        self.assertEqual(set(payload), set(PACK.inbound))
        for key, q in PACK.inbound.items():
            self.assertEqual(payload[key]["type"], q.type)
            self.assertEqual(payload[key]["instructions"], q.instructions)
            if q.criteria:
                self.assertEqual(payload[key]["criteria"], q.criteria)

    def test_choice_criteria_stay_a_mapping_and_score_stays_a_list(self):
        """Laya types these differently, so a flattened criteria would break it."""
        payload = LayaSystemOne.as_payload(PACK.inbound)
        for key, q in PACK.inbound.items():
            if q.type == "choice":
                self.assertIsInstance(payload[key]["criteria"], dict)
            elif q.type == "score":
                self.assertIsInstance(payload[key]["criteria"], list)

    def test_noul_sends_no_criteria(self):
        q = next(q for q in PACK.inbound.values() if q.type == "noul")
        item = LayaSystemOne.as_payload({"q": q})["q"]
        self.assertNotIn("criteria", item)


class ReadingAnswers(unittest.TestCase):
    def setUp(self):
        self.questions = {}
        for kind in ("noul", "choice", "score"):
            q = next((q for q in PACK.inbound.values() if q.type == kind), None)
            if q:
                self.questions[kind] = q

    def reply_for(self, questions):
        out = {}
        for key, q in questions.items():
            if q.type == "noul":
                out[key] = {"noul": 0.892}
            elif q.type == "choice":
                opts = list(q.criteria)
                probs = dict((o, 0.02) for o in opts)
                probs[opts[0]] = 1.0 - 0.02 * (len(opts) - 1)
                out[key] = {"choice": opts[0], "confidence": probs[opts[0]],
                            "probabilities": probs}
            else:
                out[key] = {"score": 1.84, "confidence": 0.87}
        return {"answers": out}

    def test_every_type_maps_onto_a_jevelin_answer(self):
        answers = LayaSystemOne.read_answers(self.reply_for(self.questions), self.questions)
        for key, q in self.questions.items():
            self.assertIsInstance(answers[key], Answer)
            self.assertEqual(answers[key].kind, q.type)

    def test_choice_confidence_comes_from_the_distribution(self):
        q = self.questions.get("choice")
        if not q:
            self.skipTest("pack has no choice question")
        answers = LayaSystemOne.read_answers(self.reply_for({"c": q}), {"c": q})
        a = answers["c"]
        self.assertEqual(a.value, list(q.criteria)[0])
        self.assertAlmostEqual(sum(a.probs.values()), 1.0, places=6)
        self.assertAlmostEqual(a.confidence, a.probs[a.value])

    def test_a_missing_answer_raises_rather_than_passing(self):
        """The fail-closed path only works if a missing answer is loud.

        Silently defaulting a safety question to 'no flag' would lower the
        drawbridge, which is the one thing the design forbids.
        """
        q = next(iter(self.questions.values()))
        with self.assertRaises(KeyError):
            LayaSystemOne.read_answers({"answers": {}}, {"only": q})

    def test_answers_may_be_returned_unwrapped(self):
        q = self.questions["noul"]
        wrapped = self.reply_for({"n": q})
        self.assertEqual(
            LayaSystemOne.read_answers(wrapped["answers"], {"n": q})["n"].value,
            LayaSystemOne.read_answers(wrapped, {"n": q})["n"].value)


class Transport(unittest.TestCase):
    def test_http_body_names_the_model_only_when_a_key_is_present(self):
        sent = {}

        def fake_urlopen(req, timeout=None):
            sent["url"] = req.full_url
            sent["headers"] = dict(req.headers)
            sent["body"] = req.data.decode()
            raise RuntimeError("stop here, the request is what we are checking")

        import jevelin.backends as backends
        real = backends.urllib.request.urlopen
        backends.urllib.request.urlopen = fake_urlopen
        try:
            keyed = bare(base_url="https://api.impossibl.com/v1", api_key="test-key-not-real")
            with self.assertRaises(RuntimeError):
                keyed._post("hello", {})
            self.assertEqual(sent["url"], "https://api.impossibl.com/v1/systemone")
            self.assertIn("convaiinnovations/laya", sent["body"])
            self.assertEqual(sent["headers"].get("Authorization"), "Bearer test-key-not-real")

            local = bare(base_url="http://localhost:8000/v1")
            with self.assertRaises(RuntimeError):
                local._post("hello", {})
            self.assertNotIn("Authorization", sent["headers"])
            self.assertNotIn("model", sent["body"])
        finally:
            backends.urllib.request.urlopen = real

    def test_a_trailing_slash_on_the_base_url_is_tolerated(self):
        seen = {}

        def fake_urlopen(req, timeout=None):
            seen["url"] = req.full_url
            raise RuntimeError("stop")

        import jevelin.backends as backends
        real = backends.urllib.request.urlopen
        backends.urllib.request.urlopen = fake_urlopen
        try:
            with self.assertRaises(RuntimeError):
                bare(base_url="http://localhost:8000/v1/")._post("hi", {})
            self.assertEqual(seen["url"], "http://localhost:8000/v1/systemone")
        finally:
            backends.urllib.request.urlopen = real

    def test_classify_charges_the_clock_and_never_blocks_the_loop(self):
        """The transports are synchronous, so they must run off the event loop.

        If classify awaited them directly, one slow classifier call would stall
        every other turn in flight. The probe below stays responsive only if
        the blocking call went to a thread.
        """
        q = next(q for q in PACK.inbound.values() if q.type == "noul")
        backend = bare(base_url="http://stub/v1")
        ticks = []

        def slow_post(state, payload):
            import time
            time.sleep(0.05)
            return {"answers": {"n": {"noul": 0.9}}}

        backend._post = slow_post

        async def main():
            clock = SimClock()

            async def probe():
                for _ in range(5):
                    ticks.append(1)
                    await asyncio.sleep(0.005)

            answers, usage = (await asyncio.gather(
                backend.classify(clock, "caller: hello", {"n": q}, "k"), probe()))[0]
            return clock, answers, usage

        clock, answers, usage = asyncio.run(main())
        self.assertEqual(len(ticks), 5, "the event loop was blocked by the transport")
        self.assertAlmostEqual(answers["n"].value, 0.9)
        self.assertGreater(clock.ms, 0.0, "real elapsed time must be charged to the clock")
        self.assertEqual(usage[0].model, "jev", "same slot, so existing profiles price it")
        self.assertGreater(usage[0].input, 0)


if __name__ == "__main__":
    unittest.main()

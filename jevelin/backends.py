"""Model backends. Every one takes a clock, so the same code runs simulated or live.

Randomness is keyed by (seed, call, turn, stage) rather than drawn from one
stream, so the baseline and the fast path see the same world: the same caller,
the same LLM latency on the same turn, the same risky draft.
"""

import asyncio
import json
import os
import random
import re
import time
import urllib.request
from dataclasses import dataclass

from .pack import est_tokens


@dataclass
class Usage:
    model: str
    input: int
    output: int = 0


@dataclass
class Answer:
    kind: str              # noul | choice | score
    value: object          # float for noul and score, option name for choice
    probs: dict = None     # choice only

    @property
    def confidence(self):
        if self.kind == "choice" and self.probs:
            return self.probs.get(self.value)
        return None


class SimClock:
    """Accumulates virtual time instead of waiting."""
    def __init__(self):
        self.ms = 0.0

    async def sleep(self, ms):
        self.ms += ms

    def charge(self, ms):
        """Account for time a real network call already took."""
        self.ms += ms


class RealClock:
    ms = 0.0

    async def sleep(self, ms):
        await asyncio.sleep(ms / 1000.0)

    def charge(self, ms):
        pass                     # already elapsed on the wall clock


def keyed_rng(seed, *parts):
    return random.Random("|".join(str(p) for p in (seed,) + parts))


# --------------------------------------------------------------------------
# classifiers: answer typed questions about a state
# --------------------------------------------------------------------------

def _hits(patterns, text):
    return sum(1 for p in patterns if re.search(p, text, re.I))


def latest_turn(state):
    """The classifier sees the whole conversation, but the questions ask about the latest turn."""
    marker = "caller: "
    return state.rsplit(marker, 1)[-1] if marker in state else state


def mock_answer(q, state):
    """Deterministic pattern answers: a stand-in for a real classifier's judgement."""
    m, text = q.mock, latest_turn(state)
    if q.type == "noul":
        return Answer("noul", m.get("hit", 0.92) if _hits(m["patterns"], text) else m.get("miss", 0.04))
    if q.type == "score":
        top = len(q.criteria) - 1
        return Answer("score", float(min(top, _hits(m["patterns"], text) * m.get("per_hit", 1.0))))
    scores = dict((opt, _hits(pats, text)) for opt, pats in m["patterns"].items())
    best_n = max(scores.values()) if scores else 0
    leaders = [o for o, n in scores.items() if n == best_n and n > 0]
    if len(leaders) == 1:
        best, conf = leaders[0], m.get("confidence", 0.88)
    elif leaders:                                  # ambiguous turn
        best, conf = leaders[0], m.get("tie_confidence", 0.48)
    else:
        best, conf = m["default"], m.get("default_confidence", 0.72)
    rest = [o for o in q.criteria if o != best]
    probs = dict((o, (1.0 - conf) / len(rest)) for o in rest)
    probs[best] = conf
    return Answer("choice", best, probs)


class MockSystemOne:
    """Simulated System One classifier: one call answers every question, same answer every time."""
    model = "jev"

    def __init__(self, profile, seed):
        self.lat = profile.latency["jev"]
        self.seed = seed

    async def classify(self, clock, state, questions, key):
        await clock.sleep(self.lat.sample(keyed_rng(self.seed, key, self.model)))
        answers = dict((k, mock_answer(q, state)) for k, q in questions.items())
        tokens = est_tokens(state) + sum(q.tokens for q in questions.values())
        return answers, [Usage(self.model, tokens)]


class JevSystemOne:
    """The real thing: TypeSafe AI's Jev via the official typesafe-sdk. Needs TYPESAFE_API_KEY."""
    model = "jev"

    def __init__(self, *_):
        from typesafe_sdk import AsyncTypeSafeClient
        self.client = AsyncTypeSafeClient()

    async def classify(self, clock, state, questions, key):
        from typesafe_sdk import Choice, Noul, Score
        build = {
            "noul": lambda q: Noul(instructions=q.instructions),
            "choice": lambda q: Choice(instructions=q.instructions, criteria=q.criteria),
            "score": lambda q: Score(instructions=q.instructions, criteria=q.criteria),
        }
        t0 = time.perf_counter()
        resp = await self.client.system_one(
            state=state, questions=dict((k, build[q.type](q)) for k, q in questions.items()))
        clock.charge((time.perf_counter() - t0) * 1000.0)
        answers = {}
        for k, a in resp.answers.items():
            if a.type == "noul":
                answers[k] = Answer("noul", float(a.noul))
            elif a.type == "choice":
                answers[k] = Answer("choice", a.choice, dict(a.probabilities))
            else:
                answers[k] = Answer("score", float(a.score))
        tokens = resp.usage.input_tokens or est_tokens(state)
        return answers, [Usage(self.model, tokens)]


class LayaSystemOne:
    """Laya: an Apache-2.0 System One classifier, run locally or behind HTTP.

    Same architectural slot as Jev, so it reports under the `jev` model key and
    every existing profile and pack works unchanged. Three ways to run it, and
    the point of supporting all three is that the *architecture* here does not
    depend on a vendor:

        local      pip install laya           -> in-process, your hardware
        self-host  laya-serve                 -> LAYA_BASE_URL=http://host:8000/v1
        hosted     api.impossibl.com          -> LAYA_BASE_URL + LAYA_API_KEY

    No API key is ever read from a file or committed. It comes from the
    environment, and only when you point this at an endpoint that needs one.

    On cost: usage is priced with the profile's `jev` rate, which is right for a
    hosted endpoint and *conservative* for a self-hosted one. If you run it on
    your own GPU the marginal per-token price is zero and the real cost is the
    card, so set `price_per_m.jev` to 0 (or to your amortised rate) in your
    profile rather than reading the cost card literally.
    """

    model = "jev"

    #: env var names, in the order they are consulted
    URL_ENV = ("LAYA_BASE_URL", "IMPOSSIBL_BASE_URL")
    KEY_ENV = ("LAYA_API_KEY", "IMPOSSIBL_API_KEY")

    def __init__(self, *_, base_url=None, api_key=None, model_name=None, timeout=10.0):
        self.base_url = base_url or _first_env(self.URL_ENV)
        self.api_key = api_key or _first_env(self.KEY_ENV)
        self.model_name = model_name or os.environ.get("LAYA_MODEL", "convaiinnovations/laya")
        self.timeout = timeout
        self.router = None
        if not self.base_url:
            # local, in-process. Imported lazily so the package stays optional.
            try:
                from laya import Router
            except ImportError:
                raise SystemExit(
                    "--classifier laya needs either the `laya` package "
                    "(pip install laya) for local inference, or LAYA_BASE_URL "
                    "pointing at a laya-serve / api.impossibl.com endpoint."
                )
            self.router = Router(preload=True)

    # -- request shaping ---------------------------------------------------

    @staticmethod
    def as_payload(questions):
        """jevelin's pack format is already Laya's question format.

        `criteria` is a {option: description} map for choice and a list of
        levels for score, which is exactly what Laya expects, so this is a
        rename rather than a translation.
        """
        out = {}
        for k, q in questions.items():
            item = {"type": q.type, "instructions": q.instructions}
            if q.criteria:
                item["criteria"] = q.criteria
            out[k] = item
        return out

    @staticmethod
    def read_answers(result, questions):
        """Map Laya's reply onto jevelin Answers, failing loudly on a missing key.

        A missing answer is NOT silently treated as a pass. The pipeline's
        fail-closed path exists for that, and it only works if this raises.
        """
        raw = result.get("answers", result)
        answers = {}
        for k, q in questions.items():
            a = raw[k]
            if q.type == "noul":
                answers[k] = Answer("noul", float(a["noul"]))
            elif q.type == "choice":
                answers[k] = Answer("choice", a["choice"], dict(a["probabilities"]))
            else:
                answers[k] = Answer("score", float(a["score"]))
        return answers

    # -- transports --------------------------------------------------------

    def _post(self, state, payload):
        body = {"state": state, "questions": payload}
        if self.api_key:                       # hosted endpoints want the model named
            body["model"] = self.model_name
        req = urllib.request.Request(
            self.base_url.rstrip("/") + "/systemone",
            data=json.dumps(body).encode(),
            headers={"content-type": "application/json"},
            method="POST",
        )
        if self.api_key:
            req.add_header("authorization", "Bearer " + self.api_key)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode())

    def _local(self, state, payload):
        return self.router.predict(state, payload)

    # -- the interface -----------------------------------------------------

    async def classify(self, clock, state, questions, key):
        payload = self.as_payload(questions)
        call = self._post if self.base_url else self._local
        t0 = time.perf_counter()
        # Both transports are synchronous. Running them on the event loop would
        # stall every other turn in flight, so they go to a worker thread.
        result = await asyncio.to_thread(call, state, payload)
        clock.charge((time.perf_counter() - t0) * 1000.0)
        answers = self.read_answers(result, questions)
        usage = (result.get("usage") or {}).get("input_tokens")
        tokens = usage or est_tokens(state) + sum(q.tokens for q in questions.values())
        return answers, [Usage(self.model, tokens)]


def _first_env(names):
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return None


class MockLLMJudge:
    """The usual alternative: one LLM prompt that answers the same questions.

    Assumed exactly as accurate as the classifier, so the comparison isolates
    latency and cost. Accuracy has to be measured separately on real data.
    """
    model = "llm_judge"

    def __init__(self, profile, pack, seed):
        self.lat = profile.latency["llm_judge"]
        self.rubric = pack.prompt_tokens["judge_rubric"]
        self.out = profile.tokens["judge_output"]
        self.seed = seed

    async def classify(self, clock, state, questions, key):
        await clock.sleep(self.lat.sample(keyed_rng(self.seed, key, self.model)))
        answers = dict((k, mock_answer(q, state)) for k, q in questions.items())
        return answers, [Usage(self.model, est_tokens(state) + self.rubric, self.out)]


# --------------------------------------------------------------------------
# generation and speech
# --------------------------------------------------------------------------

class MockLLM:
    """Reply generation; latency is time to the first sentence."""

    def __init__(self, profile, pack, seed):
        self.profile, self.pack, self.seed = profile, pack, seed

    async def reply(self, clock, tier, state, key):
        await clock.sleep(self.profile.latency[tier].sample(keyed_rng(self.seed, key, tier)))
        risky = keyed_rng(self.seed, key, "risky").random() < self.pack.risky_draft_rate
        pool = self.pack.risky_drafts if risky else self.pack.drafts[tier]
        draft = keyed_rng(self.seed, key, "draft", tier).choice(pool)
        usage = Usage(tier, est_tokens(state) + self.pack.prompt_tokens["system"],
                      self.profile.tokens["reply_output"])
        return (draft, risky), [usage]


class MockTTS:
    def __init__(self, profile, seed):
        self.lat = profile.latency["tts"]
        self.seed = seed

    async def speak(self, clock, key):
        await clock.sleep(self.lat.sample(keyed_rng(self.seed, key, "tts")))
        return None, []

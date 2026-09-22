"""Scenario packs: the domain lives in data, the pipeline stays generic.

A pack defines what the classifier is asked every turn, which answers force a
protective script, the deterministic rules that override it, the pre-approved
scripts, and the synthetic traffic used to exercise all of it.
"""

import json
from dataclasses import dataclass, field


def est_tokens(text):
    return max(1, len(text) // 4)


@dataclass
class Question:
    key: str
    type: str                      # noul | choice | score
    instructions: str
    criteria: object = None        # choice: {option: description}; score: [level, ...]
    threshold: float = 0.5         # noul: probability; score: level
    mock: dict = field(default_factory=dict)

    @property
    def tokens(self):
        return est_tokens(self.instructions + json.dumps(self.criteria or ""))


@dataclass
class Protective:
    flag: str
    action: str                    # key of a pre-approved script
    severity: int                  # lower wins when several fire
    early: bool = False            # may fire on a partial transcript


class Pack:
    def __init__(self, data):
        self.raw = data
        self.name = data["name"]
        self.description = data.get("description", "")
        self.inbound = self._questions(data["inbound"])
        self.guard = self._questions(data["guard"])
        self.protective = [Protective(**p) for p in data["protective"]]
        self.route = data["route"]
        self.rules = data.get("rules", [])
        self.scripts = data["scripts"]
        self.drafts = data["drafts"]
        self.risky_drafts = data["risky_drafts"]
        self.risky_draft_rate = data.get("risky_draft_rate", 0.0)
        self.turns = data["turns"]
        self.mix = data["mix"]
        self.turns_per_call = data.get("turns_per_call", [3, 7])
        self.context_rates = data.get("context_rates", {})
        self.prompt_tokens = data["prompt_tokens"]
        self._check()

    @staticmethod
    def _questions(items):
        return {q["key"]: Question(**q) for q in items}

    def _check(self):
        for p in self.protective:
            assert p.flag in self.inbound, "protective flag %r has no inbound question" % p.flag
            assert p.action in self.scripts, "protective action %r has no script" % p.action
        for r in self.rules:
            assert r["then"] in self.scripts, "rule target %r has no script" % r["then"]
        for key in ("hold", "safe_fallback"):
            assert key in self.scripts, "pack needs a %r script" % key
        assert self.route["question"] in self.inbound

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as fh:
            return cls(json.load(fh))

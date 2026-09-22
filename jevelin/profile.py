"""Latency and price assumptions, kept in data so they can be replaced by measurements."""

import json
import math
from dataclasses import dataclass


@dataclass
class LatencyModel:
    """Log-normal latency fitted to a p50 and a p95, in milliseconds."""
    p50: float
    p95: float

    def sample(self, rng):
        if self.p95 <= self.p50:
            return float(self.p50)
        sigma = (math.log(self.p95) - math.log(self.p50)) / 1.645
        return rng.lognormvariate(math.log(self.p50), sigma)


class Profile:
    def __init__(self, data):
        self.raw = data
        self.speech = data["speech"]
        self.latency = {k: LatencyModel(v["p50"], v["p95"]) for k, v in data["latency_ms"].items()}
        self.timeouts = data["timeouts_ms"]
        self.prices = data["price_per_m"]
        self.tokens = data["tokens"]
        self.budget_ms = data.get("ttfa_budget_ms", 800)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as fh:
            return cls(json.load(fh))

    def cost(self, usage):
        p = self.prices[usage.model]
        return (usage.input * p["in"] + usage.output * p["out"]) / 1e6

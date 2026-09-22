"""Synthetic traffic and per-turn speech timing."""

import random
from dataclasses import dataclass


@dataclass
class Turn:
    text: str
    kind: str


@dataclass
class Call:
    id: int
    turns: list
    ctx: dict


@dataclass
class Timing:
    speech_ms: float       # how long the caller talks
    partial_at: float      # when the partial transcript is available (negative: still talking)
    partial_text: str
    asr_final: float       # when the complete transcript is available
    vad: float             # when silence detection confirms the turn is over


def make_calls(pack, n, seed):
    rng = random.Random("%s|calls" % seed)
    kinds = list(pack.mix)
    weights = [pack.mix[k] for k in kinds]
    lo, hi = pack.turns_per_call
    calls = []
    for i in range(n):
        picks = rng.choices(kinds, weights, k=rng.randint(lo, hi))
        turns = [Turn(rng.choice(pack.turns[k]), k) for k in picks]
        ctx = dict((name, rng.random() < rate) for name, rate in pack.context_rates.items())
        calls.append(Call(i, turns, ctx))
    return calls


def timing(turn, profile):
    sp = profile.speech
    words = turn.text.split()
    speech = len(words) * sp["ms_per_word"]
    cut = max(1, int(len(words) * sp["partial_at"]))
    return Timing(
        speech_ms=speech,
        partial_at=-speech * (1 - cut / len(words)) + sp["asr_lag_ms"],
        partial_text=" ".join(words[:cut]),
        asr_final=sp["asr_lag_ms"],
        vad=sp["vad_ms"],
    )


def context(history, utterance, keep=8):
    return "\n".join(history[-keep:] + ["caller: " + utterance])

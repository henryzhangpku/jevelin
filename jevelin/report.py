"""Summaries, the comparison table, and per-turn timelines.

Colour is opt-out and self-disabling. It is on only when stdout is a real
terminal, so piping to a file, redirecting, or running the engine under Pyodide
in the browser all produce exactly the same plain text as before. NO_COLOR
turns it off, FORCE_COLOR turns it on regardless.
"""

import os
import statistics
import sys
import textwrap


def _colour_enabled():
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if os.environ.get("TERM") == "dumb":
        return False
    try:
        return bool(sys.stdout.isatty())
    except Exception:
        return False


COLOUR = _colour_enabled()

if COLOUR and sys.platform == "win32":          # ask the console for ANSI
    try:
        import ctypes
        _k = ctypes.windll.kernel32
        _k.SetConsoleMode(_k.GetStdHandle(-11), 7)
    except Exception:
        COLOUR = False

_CODES = {"bold": "1", "dim": "2", "red": "31", "green": "32", "yellow": "33",
          "blue": "34", "magenta": "35", "cyan": "36", "grey": "90"}


def tint(text, *styles):
    """Wrap text in ANSI styles, or return it untouched when colour is off.

    Always applied AFTER padding, never before, or the invisible escape bytes
    are counted as width and every column drifts.
    """
    if not COLOUR or not styles:
        return text
    codes = ";".join(_CODES[s] for s in styles if s in _CODES)
    return "\033[%sm%s\033[0m" % (codes, text) if codes else text


def _stage_style(name):
    """One colour per kind of work, so a timeline is readable at a glance."""
    n = name.lower()
    if "jev" in n or "laya" in n or "classifier" in n or "guard" in n:
        return ("cyan",)
    if "judge" in n:
        return ("red",)
    if "large" in n:
        return ("magenta",)
    if "small" in n:
        return ("yellow",)
    if "tts" in n or "speech" in n:
        return ("green",)
    if "policy" in n or "rule" in n:
        return ("blue",)
    return ("grey",)


def pct(xs, q):
    s = sorted(xs)
    if not s:
        return float("nan")
    return s[min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))]


def summarize(results, profile):
    n_calls = len(set(r.call for r in results)) or 1
    llm_tok = jev_tok = 0
    cost = wasted = 0.0
    for r in results:
        for s in r.stages:
            for u in s.usage:
                c = profile.cost(u)
                cost += c
                if s.wasted:
                    wasted += c
                if u.model == "jev":
                    jev_tok += u.input + u.output
                else:
                    llm_tok += u.input + u.output
    ttfa = [r.ttfa for r in results]
    protective = [r for r in results if r.decision.kind in ("protective", "rule")]
    ptimes = [r.protect_ms for r in protective if r.protect_ms is not None]
    return {
        "turns": len(results),
        "calls": n_calls,
        "ttfa_p50": pct(ttfa, 0.50),
        "ttfa_p95": pct(ttfa, 0.95),
        "over_budget": sum(1 for t in ttfa if t > profile.budget_ms) / max(len(ttfa), 1),
        "llm_tokens_per_call": llm_tok / n_calls,
        "jev_tokens_per_call": jev_tok / n_calls,
        "cost_per_call": cost / n_calls,
        "wasted_share": wasted / cost if cost else 0.0,
        "script_share": sum(1 for r in results if r.decision.tier == "script") / max(len(results), 1),
        "protective": len(protective),
        "protect_p50": statistics.median(ptimes) if ptimes else float("nan"),
        "protect_early": sum(1 for r in protective if r.early_protect),
        "risky_spoken": sum(1 for r in results if r.risky_spoken),
        "guard_blocked": sum(1 for r in results if r.guard_blocked),
        "fail_closed": sum(1 for r in results if r.fail_closed),
    }


def _ms(v):
    return "%s ms" % format(int(round(v)), ",")


def comparison(base, fast, profile):
    budget = profile.budget_ms
    rows = [
        ("Time to first audio, p50", _ms(base["ttfa_p50"]), _ms(fast["ttfa_p50"])),
        ("Time to first audio, p95", _ms(base["ttfa_p95"]), _ms(fast["ttfa_p95"])),
        ("Turns over %d ms budget" % budget, "%.0f%%" % (100 * base["over_budget"]),
         "%.0f%%" % (100 * fast["over_budget"])),
        ("LLM tokens per call", format(int(base["llm_tokens_per_call"]), ","),
         format(int(fast["llm_tokens_per_call"]), ",")),
        ("Classifier tokens per call", format(int(base["jev_tokens_per_call"]), ","),
         format(int(fast["jev_tokens_per_call"]), ",")),
        ("Model cost per call", "$%.4f" % base["cost_per_call"], "$%.4f" % fast["cost_per_call"]),
        ("  of which wasted speculation", "-", "%.1f%%" % (100 * fast["wasted_share"])),
        ("Turns answered by a script", "%.0f%%" % (100 * base["script_share"]),
         "%.0f%%" % (100 * fast["script_share"])),
        ("Protective actions", str(base["protective"]), str(fast["protective"])),
        ("  decided at, p50 (0 = caller stops)", _ms(base["protect_p50"]), _ms(fast["protect_p50"])),
        ("  decided before caller finished", str(base["protect_early"]), str(fast["protect_early"])),
        ("Risky drafts spoken to caller", str(base["risky_spoken"]), str(fast["risky_spoken"])),
        ("Risky drafts blocked", str(base["guard_blocked"]), str(fast["guard_blocked"])),
        ("Fail-closed turns", str(base["fail_closed"]), str(fast["fail_closed"])),
    ]
    # what the fast path column MEANS on each row: an improvement, or the
    # price you pay for it. Colour follows the meaning, not the direction of
    # the number, so the classifier tokens and the wasted spend read as costs
    # rather than as wins.
    meaning = {
        "Time to first audio, p50": "good",
        "Time to first audio, p95": "good",
        "Turns over %d ms budget" % budget: "good",
        "LLM tokens per call": "good",
        "Classifier tokens per call": "cost",
        "Model cost per call": "good",
        "  of which wasted speculation": "cost",
        "Turns answered by a script": "good",
        "  decided at, p50 (0 = caller stops)": "good",
        "  decided before caller finished": "good",
        "Risky drafts spoken to caller": "good",
        "Risky drafts blocked": "good",
        "Fail-closed turns": "cost",
    }

    w0 = max(len(r[0]) for r in rows) + 2
    lines = ["%s%s%s" % ("".ljust(w0),
                         tint("%14s" % "baseline", "bold", "grey"),
                         tint("%14s" % "fast path", "bold", "green")),
             tint("%s%14s%14s" % ("".ljust(w0), "--------", "---------"), "dim")]
    for label, b, f in rows:
        kind = meaning.get(label)
        style = ("green", "bold") if kind == "good" else (("yellow",) if kind == "cost" else ())
        lines.append("%s%s%s" % (
            tint(label.ljust(w0), "dim") if label.startswith("  ") else label.ljust(w0),
            tint("%14s" % b, "grey"),
            tint("%14s" % f, *style)))
    return "\n".join(lines)


def gantt(r, ms_per_char=50, name_w=26, budget_ms=None):
    T = r.timing
    lo = min([-T.speech_ms] + [s.start for s in r.stages])
    lo = max(lo, -3000.0)
    lo = (int(lo) // 500 - (1 if lo % 500 else 0)) * 500
    hi = max(s.end for s in r.stages) + 100
    width = int((hi - lo) / ms_per_char) + 2

    def col(t):
        return max(0, min(width - 1, int(round((t - lo) / ms_per_char))))

    zero = col(0)
    out = []
    head = '%s  turn %d  [%s]  "%s"' % (r.pipeline.upper(), r.turn, r.kind, r.text)
    # wrap on the PLAIN string, then tint. Wrapping coloured text miscounts
    # width, because the escape bytes are invisible but still characters.
    wrapped = textwrap.wrap(head, 100, subsequent_indent="    ")
    if COLOUR and wrapped:
        tag = r.pipeline.upper()
        wrapped[0] = wrapped[0].replace(
            tag, tint(tag, "bold", "green" if r.pipeline == "fast" else "yellow"), 1)
        wrapped[0] = wrapped[0].replace("[%s]" % r.kind, tint("[%s]" % r.kind, "cyan"), 1)
    out += wrapped

    axis = [" "] * (width + 8)
    t = lo
    while t <= hi:
        label = "0" if t == 0 else ("%+d" % t)
        at = col(t)
        for j, ch in enumerate(label):
            if at + j < len(axis):
                axis[at + j] = ch
        t += 500
    out.append(tint("".ljust(name_w) + "".join(axis).rstrip() + "  ms", "dim"))

    def row(name, a, b, ch, styles=()):
        cells = [" "] * width
        cells[zero] = "|"
        lo_i, hi_i = col(a), max(col(b), col(a) + 1)
        for i in range(lo_i, hi_i):      # NOT `c`: that is the colour helper
            cells[i] = ch
        bar = "".join(cells).rstrip()
        if COLOUR and styles and lo_i < len(bar):
            end = min(hi_i, len(bar))
            bar = bar[:lo_i] + tint(bar[lo_i:end], *styles) + bar[end:]
        return tint(name[:name_w - 1].ljust(name_w), "dim") + bar

    out.append(row("caller speaking", -T.speech_ms, 0, ".", ("dim",)))
    for s in sorted(r.stages, key=lambda s: (s.start, s.end)):
        if s.wasted:
            ch, styles, note = "~", ("dim",), tint("  (discarded)", "dim")
        elif s.timed_out:
            ch, styles, note = "!", ("red", "bold"), tint("  (timed out)", "red", "bold")
        else:
            ch, styles, note = "#", _stage_style(s.name), ""
        out.append(row(s.name, s.start, s.end, ch, styles) + note)

    on_budget = budget_ms is None or r.ttfa <= budget_ms
    out.append("".ljust(name_w)
               + tint("-> ", "dim")
               + tint("%s: %s" % (r.decision.action, r.decision.why), "bold")
               + tint(" | first audio at ", "dim")
               + tint(_ms(r.ttfa), "bold", "green" if on_budget else "red"))
    return "\n".join(out)

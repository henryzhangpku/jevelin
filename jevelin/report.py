"""Summaries, the comparison table, and per-turn timelines."""

import statistics
import textwrap


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
    w0 = max(len(r[0]) for r in rows) + 2
    lines = ["%s%14s%14s" % ("".ljust(w0), "baseline", "fast path"),
             "%s%14s%14s" % ("".ljust(w0), "--------", "---------")]
    lines += ["%s%14s%14s" % (a.ljust(w0), b, c) for a, b, c in rows]
    return "\n".join(lines)


def gantt(r, ms_per_char=50, name_w=26):
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
    out += textwrap.wrap(head, 100, subsequent_indent="    ")

    axis = [" "] * (width + 8)
    t = lo
    while t <= hi:
        label = "0" if t == 0 else ("%+d" % t)
        c = col(t)
        for j, ch in enumerate(label):
            if c + j < len(axis):
                axis[c + j] = ch
        t += 500
    out.append("".ljust(name_w) + "".join(axis).rstrip() + "  ms")

    def row(name, a, b, ch):
        cells = [" "] * width
        cells[zero] = "|"
        for c in range(col(a), max(col(b), col(a) + 1)):
            cells[c] = ch
        return name[:name_w - 1].ljust(name_w) + "".join(cells).rstrip()

    out.append(row("caller speaking", -T.speech_ms, 0, "."))
    for s in sorted(r.stages, key=lambda s: (s.start, s.end)):
        ch = "~" if s.wasted else ("!" if s.timed_out else "#")
        note = "  (discarded)" if s.wasted else ("  (timed out)" if s.timed_out else "")
        out.append(row(s.name, s.start, s.end, ch) + note)
    out.append("".ljust(name_w) + "-> %s: %s | first audio at %s"
               % (r.decision.action, r.decision.why, _ms(r.ttfa)))
    return "\n".join(out)

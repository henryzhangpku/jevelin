"""Stage executors. The pipeline declares stages and their dependencies once.

SimExecutor computes each stage's start as max(its anchor time, its
dependencies' ends), so time-to-first-audio falls out as the critical path.
Backends that do real I/O (the live Jev client) charge their measured wall
time to the clock, so real latencies slot into the same arithmetic.

LiveExecutor runs the identical stages concurrently in real time.

Times are milliseconds relative to the moment the caller stops speaking, so
work done while they are still talking has negative timestamps.
"""

import asyncio
import time

from .backends import RealClock, SimClock


class _Timeout:
    def __repr__(self):
        return "TIMEOUT"


TIMEOUT = _Timeout()          # also returned when a stage raises: fail closed


class Handle:
    def __init__(self, name):
        self.name = name
        self.start = self.end = None
        self.result = None
        self.usage = []
        self.wasted = False
        self.timed_out = False
        self.error = None
        self.task = None
        self._done = asyncio.Event()

    async def get(self):
        await self._done.wait()
        return self.result

    def discard(self):
        """Mark speculative work as thrown away. Its tokens were still spent."""
        self.wasted = True


class _Executor:
    live = False

    def __init__(self):
        self.stages = []

    def spawn(self, name, fn, *, after=(), at=None, timeout=None):
        assert after or at is not None, "stage %r needs an anchor time or dependencies" % name
        h = Handle(name)
        self.stages.append(h)
        h.task = asyncio.ensure_future(self._run(h, fn, tuple(after), at, timeout))
        return h

    async def settle(self):
        """Wait for every stage, including discarded speculation, before accounting."""
        await asyncio.gather(*(h.task for h in self.stages))


class SimExecutor(_Executor):
    def begin_turn(self, lead_ms):
        self.stages = []

    async def _run(self, h, fn, after, at, timeout):
        for d in after:
            await d._done.wait()
        h.start = max([at if at is not None else float("-inf")] + [d.end for d in after])
        clock = SimClock()
        try:
            result, usage = await fn(clock)
        except Exception as exc:          # a failing backend is treated as no answer
            result, usage, h.error = TIMEOUT, [], repr(exc)
        dur = clock.ms
        if timeout is not None and dur > timeout:
            h.end, h.result, h.timed_out = h.start + timeout, TIMEOUT, True
        else:
            h.end, h.result = h.start + dur, result
        h.timed_out = h.timed_out or result is TIMEOUT
        h.usage = usage
        h._done.set()


class LiveExecutor(_Executor):
    live = True

    def __init__(self):
        super().__init__()
        self.zero = 0.0

    def now(self):
        return time.perf_counter() * 1000.0 - self.zero

    def begin_turn(self, lead_ms):
        """The caller starts speaking now and stops lead_ms from now."""
        self.stages = []
        self.zero = time.perf_counter() * 1000.0 + lead_ms

    async def _run(self, h, fn, after, at, timeout):
        for d in after:
            await d._done.wait()
        if at is not None and at > self.now():
            await asyncio.sleep((at - self.now()) / 1000.0)
        h.start = self.now()
        try:
            coro = fn(RealClock())
            if timeout is not None:
                coro = asyncio.wait_for(coro, timeout / 1000.0)
            h.result, h.usage = await coro
        except asyncio.TimeoutError:
            h.result, h.timed_out = TIMEOUT, True
        except Exception as exc:
            h.result, h.timed_out, h.error = TIMEOUT, True, repr(exc)
        h.end = self.now()
        h._done.set()

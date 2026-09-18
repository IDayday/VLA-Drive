"""Overlap CPU scoring with independent reference inference, never collectives.

Only the CPU score callable enters the worker thread. All reference work and
control-plane error exchanges stay on the calling (training) thread.
"""
import time
from .distributed import synchronized_call


def score_with_reference(score, reference, device, executor=None):
    def timed_score():
        start = time.monotonic()
        return score(), time.monotonic() - start

    phases = {}
    started = time.monotonic()
    if executor is None:
        records, elapsed = synchronized_call(timed_score, device)
        phases["official_reward"] = time.monotonic() - started
        reference_start = time.monotonic()
        statistics = reference()
        phases["reference"] = time.monotonic() - reference_start
    else:
        future = executor.submit(timed_score)
        try:
            reference_start = time.monotonic()
            statistics = reference()
            phases["reference"] = time.monotonic() - reference_start
            wait_start = time.monotonic()
            records, elapsed = synchronized_call(future.result, device)
            phases["official_reward_remaining_wait"] = time.monotonic() - wait_start
            # Nested spans overlap and must NOT be added to reference time.
            phases["official_reward_nested"] = elapsed
        except BaseException:
            future.cancel()
            raise
    phases["score_reference_total"] = time.monotonic() - started
    return records, statistics, phases

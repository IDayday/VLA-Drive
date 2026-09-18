from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock, current_thread
import pytest
from starVLA.rl.flow_grpo.overlap import score_with_reference
from starVLA.rl.flow_grpo.reward import RewardService


def test_real_helper_overlaps_only_cpu_work_and_preserves_results():
    started, reference_seen = Event(), Event()
    caller = current_thread()

    def score():
        assert current_thread() is not caller
        started.set()
        assert reference_seen.wait(5), "reference blocked behind scoring"
        return [0., .5, 1.]

    def reference():
        assert current_thread() is caller
        assert started.wait(5)
        reference_seen.set()
        return {"mean": [1., 2.], "std": [.1, .2]}

    with ThreadPoolExecutor(1) as executor:
        records, stats, timing = score_with_reference(score, reference, "cpu", executor)
    original = score_with_reference(lambda: [0., .5, 1.], lambda: stats, "cpu")
    assert (records, stats) == original[:2]
    assert timing["score_reference_total"] >= timing["reference"]
    assert "official_reward_nested" in timing


def test_score_failure_propagates_after_reference():
    completed = []
    def score():
        raise ValueError("official scoring failed")
    with ThreadPoolExecutor(1) as executor:
        with pytest.raises(Exception, match="official scoring failed"):
            score_with_reference(score, lambda: completed.append(True), "cpu", executor)
    assert completed == [True]


def test_reference_failure_is_not_masked():
    def reference():
        raise RuntimeError("reference failed")
    with ThreadPoolExecutor(1) as executor:
        with pytest.raises(RuntimeError, match="reference failed"):
            score_with_reference(lambda: [], reference, "cpu", executor)


def test_concurrent_cleanup_takes_pool_once():
    class Pool:
        _processes = {}
        calls = 0
        def shutdown(self, **kwargs):
            self.calls += 1
    service = RewardService.__new__(RewardService)
    service._close_lock = Lock()
    service._closed = False
    pool = service.pool = Pool()
    with ThreadPoolExecutor(2) as executor:
        futures = [executor.submit(service.close, True) for _ in range(2)]
        for future in futures:
            future.result()
    assert pool.calls == 1 and service.pool is None
    with pytest.raises(RuntimeError, match="closed"):
        service.score([], [])

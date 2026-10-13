from copy import deepcopy
import numpy as np
from iqe.contracts import require
from iqe.io import digest


class TwoStreamSampler:
    """Only consume commits RNG/cursors. Peeking/prefetching never advances them."""
    def __init__(self, scene_ids, gap_weights, *, natural_seed, comparison_seed, pointwise_updates=500, bank_hash):
        self.ids = tuple(scene_ids)
        self.weights = np.asarray(gap_weights, dtype=np.float64)
        require(len(self.ids) > 0 and len(self.ids) == len(self.weights), "sampler population")
        require(np.isfinite(self.weights).all() and (self.weights >= 0).all(), "invalid comparison distribution")
        self.q = self.weights/self.weights.sum() if self.weights.sum() else None
        self.natural = np.random.default_rng(natural_seed)
        self.comparison = np.random.default_rng(comparison_seed)
        self.pointwise_updates, self.bank_hash = pointwise_updates, bank_hash
        self.q_hash = digest({"ids": self.ids, "weights": self.weights.tolist(), "bank": bank_hash})
        self.step, self.natural_consumed, self.comparison_consumed = 0, 0, 0
        self.pending = None

    def peek(self, n):
        require(n == 128, "fixed scorer global scene budget=128")
        if self.pending is None:
            nr, cr = deepcopy(self.natural), deepcopy(self.comparison)
            # With absent comparison signal, pointwise training continues naturally.
            n_natural = 128 if self.step < self.pointwise_updates or self.q is None else 64
            ni = nr.integers(len(self.ids), size=n_natural)
            ci = cr.choice(len(self.ids), size=128-n_natural, p=self.q) if n_natural < 128 else []
            entries = [{"scene_id": self.ids[i], "natural": True, "comparison": False, "stream": "natural"} for i in ni]
            entries += [{"scene_id": self.ids[i], "natural": False, "comparison": True, "stream": "comparison"} for i in ci]
            self.pending = (entries, nr.bit_generator.state, cr.bit_generator.state, n_natural)
        return deepcopy(self.pending[0])

    def consume(self, n):
        require(self.pending is not None and n == len(self.pending[0]), "consume must match delivered batch")
        _, ns, cs, count = self.pending
        self.natural.bit_generator.state, self.comparison.bit_generator.state = ns, cs
        self.natural_consumed += count; self.comparison_consumed += n-count; self.step += 1
        self.pending = None

    def state_dict(self):
        return {"schema": "two_stream_v1", "step": self.step, "phase": "pointwise" if self.step < self.pointwise_updates else "comparison",
                "natural_rng": self.natural.bit_generator.state, "comparison_rng": self.comparison.bit_generator.state,
                "natural_consumed": self.natural_consumed, "comparison_consumed": self.comparison_consumed,
                "bank_hash": self.bank_hash, "q_hash": self.q_hash, "pointwise_updates": self.pointwise_updates}

    def load_state_dict(self, state):
        require(state["bank_hash"] == self.bank_hash and state["q_hash"] == self.q_hash
                and state["pointwise_updates"] == self.pointwise_updates, "resume bank/comparison distribution/stage changed")
        self.natural.bit_generator.state = state["natural_rng"]
        self.comparison.bit_generator.state = state["comparison_rng"]
        self.step, self.natural_consumed, self.comparison_consumed = state["step"], state["natural_consumed"], state["comparison_consumed"]
        self.pending = None


class ExpertSampler:
    def __init__(self, hard, ordinary, seed, manifest_hash, hard_fraction=.9):
        require(hard and ordinary, "hard and ordinary buckets must both contain audited GT")
        self.hard, self.ordinary = tuple(hard), tuple(ordinary)
        self.rng = np.random.default_rng(seed)
        require(0 < hard_fraction < 1, "hard/ordinary fractions")
        self.hard_fraction = hard_fraction
        self.identity = digest([self.hard, self.ordinary, seed, manifest_hash, hard_fraction])
        self.consumed, self.pending = 0, None

    def peek(self, n):
        if self.pending is None:
            rng = deepcopy(self.rng)
            # Cumulative quotas apply to draws, never duplicated in the IL loss.
            nh = round((self.consumed+n)*self.hard_fraction)-round(self.consumed*self.hard_fraction)
            entries = [{"scene_id": self.hard[i], "bucket": "hard"} for i in rng.integers(len(self.hard), size=nh)]
            entries += [{"scene_id": self.ordinary[i], "bucket": "ordinary"} for i in rng.integers(len(self.ordinary), size=n-nh)]
            rng.shuffle(entries)
            self.pending = (entries, rng.bit_generator.state)
        return deepcopy(self.pending[0])

    def consume(self, n):
        require(self.pending is not None and n == len(self.pending[0]), "expert consumed batch mismatch")
        self.rng.bit_generator.state = self.pending[1]; self.consumed += n; self.pending = None

    def state_dict(self):
        return {"identity": self.identity, "rng": self.rng.bit_generator.state, "consumed": self.consumed}

    def load_state_dict(self, state):
        require(state["identity"] == self.identity, "expert manifest/sampler changed")
        self.rng.bit_generator.state = state["rng"]; self.consumed = state["consumed"]; self.pending = None

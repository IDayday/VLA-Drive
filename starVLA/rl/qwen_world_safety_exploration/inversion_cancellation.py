"""Pure CPU canonical-dominance sign cancellation; no runtime registration.

This is distinct from simply setting every high-quality negative advantage to
zero. It transfers only existing positive credit from a lower-canonical,
strictly feasible candidate to an opposing high-quality negative candidate.
It never changes A_safe, normalization, rewards, masks, or actor material.
Callers retain those fields and add ``delta`` to A_EP and A_raw only under an
independently registered credit schema. This module does not register one.
"""
import math

import torch

PROTOCOL = 'world_strict_quality_canonical_inversion_pair_cancellation_math_v1'


def _feedback(value, name, shape=None):
    if (not torch.is_tensor(value) or value.requires_grad or value.dtype != torch.float64
            or (shape is not None and tuple(value.shape) != tuple(shape))
            or not torch.isfinite(value).all()):
        raise ValueError('Detached finite FP64 feedback required: '+name)
    return value.detach().cpu()


@torch.no_grad()
def cancel_inversions(raw_A, canonical_scores, components, comfort, *, high_threshold=.9):
    """Return deterministic CPU delta, updated rawA, transfers and trigger counts.

    Inputs are detached FP64 official feedback: raw_A/scores/comfort [S,G],
    components [S,G,4] in NC,DAC,TTC,EP order, G8 or G16. Scores must be the
    official FP64 values, not rounded FP32 reward labels. Inputs are not mutated.

    Order exactly matches the preceding full negative-high CPU replay: negative
    strict4 high endpoints by descending canonical score / ascending candidate
    index, positives by ascending score / ascending index. Strict4 means
    NC=DAC=TTC=Comfort=1, not EP=1. Strictly lower score is required. Each transfer
    is min(-A_high,A_lower), so at least one opposing sign disappears. Some high
    negatives can remain after available lower-scoring positive mass is spent.

    Group rawA sums are conserved in real arithmetic, subject to FP64 rounding;
    no recentering or standard-deviation operation is performed. Use the returned
    updated_rawA as the canonical transformed raw tensor; delta is its difference
    from the original tensor. Unchanged locations are copied bit for bit.
    """
    raw = _feedback(raw_A, 'raw_A')
    if raw.ndim != 2 or raw.shape[0] < 1 or raw.shape[1] not in (8, 16):
        raise ValueError('One or more [S,G8/G16] groups required')
    score = _feedback(canonical_scores, 'canonical_scores', raw.shape)
    c = _feedback(components, 'components', (*raw.shape, 4))
    comf = _feedback(comfort, 'comfort', raw.shape)
    if (not isinstance(high_threshold, (int, float)) or isinstance(high_threshold, bool)
            or not math.isfinite(high_threshold) or not 0 <= high_threshold <= 1):
        raise ValueError('Finite canonical high threshold in [0,1] required')
    if any(not bool(((v >= 0) & (v <= 1)).all()) for v in (score, c, comf)):
        raise ValueError('Official metrics must lie in [0,1]')
    canonical = c[:, :, 0]*c[:, :, 1]*(5*c[:, :, 3]+5*c[:, :, 2]+2*comf)/12
    if not torch.allclose(score, canonical, atol=5e-15, rtol=0):
        raise ValueError('Official canonical score and component feedback disagree')
    feasible = (c[:, :, :3] == 1).all(-1)
    strict4 = feasible & (comf == 1)
    negative_high = strict4 & (score >= high_threshold) & (raw < 0)
    updated = raw.clone(); pairs = []; initial_pairs = 0; triggered_groups = 0
    for group in range(raw.shape[0]):
        neg = sorted(torch.nonzero(negative_high[group]).flatten().tolist(),
                     key=lambda i: (-float(score[group, i]), i))
        pos = sorted(torch.nonzero((raw[group] > 0) & feasible[group]).flatten().tolist(),
                     key=lambda j: (float(score[group, j]), j))
        initial = sum(float(score[group, j]) < float(score[group, i]) for i in neg for j in pos)
        initial_pairs += initial; triggered_groups += int(initial > 0)
        for i in neg:
            for j in pos:
                if not (updated[group, i] < 0 and updated[group, j] > 0
                        and score[group, i] > score[group, j]):
                    continue
                amount = min(-float(updated[group, i]), float(updated[group, j]))
                updated[group, i] += amount
                updated[group, j] -= amount
                pairs.append(dict(group=group, high_candidate=i,
                    lower_score_positive_candidate=j, transfer=amount))
    changed = updated.ne(raw)
    remaining_negative = strict4 & (score >= high_threshold) & (updated < 0)
    remaining_pairs = sum(int(((updated[group] > 0) & feasible[group]
                              & (score[group] < score[group, i])).sum())
                          for group in range(raw.shape[0])
                          for i in torch.nonzero(remaining_negative[group]).flatten().tolist())
    if remaining_pairs:
        raise AssertionError('Opposing canonical-dominance pair remains after cancellation')
    sums_before = raw.sum(1); sums_after = updated.sum(1)
    delta = updated-raw
    return dict(protocol=PROTOCOL, high_threshold=float(high_threshold),
        delta=delta, updated_rawA=updated, pairs=pairs,
        counts=dict(groups=raw.shape[0], candidates=raw.numel(),
            original_negative_high=int(negative_high.sum()),
            original_qualifying_pairs=initial_pairs, triggered_groups=triggered_groups,
            transfers=len(pairs), changed_groups=int(changed.any(1).sum()),
            changed_candidates=int(changed.sum()),
            negative_high_cleared=int((negative_high & ~remaining_negative).sum()),
            remaining_negative_high=int(remaining_negative.sum()),
            remaining_qualifying_pairs=remaining_pairs,
            newly_allzero_groups=int((updated.eq(0).all(1) & raw.ne(0).any(1)).sum())),
        group_sum_before=sums_before, group_sum_after=sums_after,
        max_group_sum_roundoff=float((sums_after-sums_before).abs().max()))

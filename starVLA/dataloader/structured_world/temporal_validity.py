"""Keep native planning eligibility separate from physical-time scene labels."""
import numpy as np


def mask_auxiliary_time_mismatch(arrays, actual_future_times_s, *, tolerance_s=.06):
    """Nominal 0.5-second scene labels accept at most 60 ms annotation jitter.

    Larger jitter is unknown, never interpolated into a fabricated object GT
    and never free. Native VAD six-keyframe ego targets and eligibility remain
    untouched. This is a disclosed nearest-keyframe approximation, not exact
    annotation-time resampling, shared across matched groups.
    """
    times = np.asarray(actual_future_times_s, dtype=np.float64)
    expected = .5*np.arange(1, len(times)+1)
    valid = np.isfinite(times) & (np.abs(times-expected) <= tolerance_s)
    if arrays['occupancy'].shape[0] != len(times)+1: raise ValueError('Time/grid shape mismatch')
    arrays['future_label_time_valid'] = valid.copy()
    for index in np.flatnonzero(~valid)+1:
        arrays['occupancy'][index] = 255
        arrays['occupancy_valid'][index] = False
        arrays['instances'][index] = 255
    return valid

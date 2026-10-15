"""First-event labels at the original trajectory times, with explicit semantics.

NC here means the first at-fault event (including an official 0.5 collision).
TTC is the first official TTC *infraction time*, not a collision timestamp.
Neither event is inferred from a terminal scalar or fabricated at the endpoint.
"""
import numpy as np
from iqe.contracts import require

EVENTS = ('at_fault_collision_event', 'offroad_event', 'ttc_infraction_event')


def event_labels(collision_index, offroad, ttc_index, *, simulator_dt, horizon, dt, components):
    area = np.asarray(offroad)
    require(area.ndim == 1 and area.dtype == np.bool_, 'official per-time offroad bool array required')
    require(simulator_dt > 0 and dt > 0 and horizon > 0, 'positive time contract')
    require(abs((len(area)-1)*simulator_dt-horizon*dt) < 1e-9, 'simulation horizon mismatch')
    require(abs(dt/simulator_dt-round(dt/simulator_dt)) < 1e-9, 'waypoints must align to simulation')
    offroad_ids = np.flatnonzero(area)
    indices = [collision_index, int(offroad_ids[0]) if len(offroad_ids) else np.inf, ttc_index]
    keys = ('no_at_fault_collisions', 'drivable_area_compliance', 'time_to_collision_within_bound')
    times = np.arange(1, horizon+1, dtype=np.float64)*dt
    labels = np.zeros((horizon, len(EVENTS)), dtype=np.bool_)
    first = []
    for col, (index, key) in enumerate(zip(indices, keys)):
        score = float(components[key])
        require(np.isfinite(score) and 0 <= score <= 1, 'invalid official component')
        require(not np.isnan(index) and index >= 0, 'invalid event index')
        event = np.isfinite(index)
        require(event == (score < 1-1e-8), 'event trace and official component disagree')
        if event:
            require(float(index).is_integer() and index < len(area), 'out-of-range event index')
            at = float(index)*simulator_dt
            labels[:, col] = times >= at-1e-12
            first.append(at)
        else:
            first.append(None)
    return dict(event_names=list(EVENTS), times_s=times.tolist(), first_event_s=first,
                cumulative_event=labels.tolist(), valid=np.ones_like(labels).tolist(),
                includes_t0_events_in_first_bin=True,
                NC_0_5_is_an_event=True, TTC_is_official_infraction_not_collision=True)

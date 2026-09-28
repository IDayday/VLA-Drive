"""Independent vehicle-only GT schema. Never pass labels to prediction providers."""
import numpy as np
from starVLA.model.modules.structured_world.targets import CLASSES, make_targets

SCHEMA = "ddpolicy_vehicle_targets_v1"


def make_vehicle_targets(current, future, capacity=32, bounds=(1., -20., 50., 20.),
                         steps=8, current_eligibility=None):
    if capacity < 1 or steps != 8:
        raise ValueError("Vehicle task requires positive capacity and 8 half-second future points")
    anns = current.get("anns")
    if anns is None:
        result = make_targets(current, future, capacity, bounds, steps)
        return result, {"source_current_objects": 0, "source_vehicles": 0,
                        "raw_log_population_available": True, "annotation_present": False}
    names, boxes, tracks = (np.asarray(anns[k]) for k in ("gt_names", "gt_boxes", "track_tokens"))
    if boxes.shape != (len(names), 7) or len(tracks) != len(names):
        raise ValueError("Raw annotation fields are not aligned")
    unknown = set(names) - set(CLASSES)
    if unknown:
        raise ValueError(f"Unknown original NAVSIM category names: {unknown}")
    keep = names == "vehicle"
    vehicle_current = dict(current)
    vehicle_current["anns"] = {"gt_names": names[keep], "gt_boxes": boxes[keep],
                               "track_tokens": [str(x) for x in tracks[keep]]}
    eligible = None
    if current_eligibility is not None:
        eligible = np.asarray(current_eligibility)
        if eligible.shape != (len(names),) or eligible.dtype != np.bool_:
            raise ValueError("Current geometric support must be boolean per source object")
        eligible = eligible[keep]
    result = make_targets(vehicle_current, future, capacity, bounds, steps, eligible)
    if result.current_classes.numel() and not (result.current_classes == CLASSES.index("vehicle")).all():
        raise AssertionError("Nonvehicle escaped vehicle label contract")
    return result, {"source_current_objects": len(names), "source_vehicles": int(keep.sum()),
                    "selected_vehicles": len(result.track_ids), "overflow": result.overflow,
                    "raw_log_population_available": True, "annotation_present": True,
                    "future_valid_points": int(result.future_valid_mask.sum())}

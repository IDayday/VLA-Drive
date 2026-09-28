"""Label-side vehicle coverage and same-track motion; never called by prediction.

Every supervised current GT vehicle gets a row, including misses and export
failures. Matching uses current centers, then a fixed 2m evaluation gate. This
evaluation gate never controls training assignment or motion supervision.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from .prepare_data import atomic_json
from starVLA.model.modules.vehicle_joint.initialization import file_sha256


def evaluate_scene(prediction, target, *, existence=.2, maximum_center_error=2.):
    boxes = target["current_boxes"].numpy()
    xy = target["future_xy_in_ego_t0"].numpy()
    valid = target["future_valid_mask"].numpy().astype(bool)
    keep = target["current_supervision_mask"].numpy().astype(bool)
    indices = np.flatnonzero(keep)
    detected = {}
    selected = {}
    proposals = []
    if prediction is not None:
        for value in prediction.values():
            if not np.isfinite(value).all(): raise ValueError("Nonfinite exported vehicle prediction")
        if "vehicle_boxes" in prediction:
            logits = prediction["vehicle_logits"]
            logits = logits-logits.max(-1, keepdims=True)
            prob = np.exp(logits); prob = prob[:, 0]/prob.sum(-1)
            proposals = np.flatnonzero(prob >= existence)
            if len(proposals) and len(indices):
                costs = np.linalg.norm(prediction["vehicle_boxes"][proposals, None, :2]-boxes[indices][None, :, :2], axis=-1)
                rows, cols = linear_sum_assignment(costs)
                detected = {int(indices[c]):int(proposals[r]) for r,c in zip(rows, cols) if costs[r,c] <= maximum_center_error}
            selected = {int(q):actor-1 for actor,q in enumerate(prediction["selected_query_indices"]) if actor and prediction["active_actor_mask"][actor]}
    rows = []
    for gt in indices:
        points = np.flatnonzero(valid[gt]); query = detected.get(int(gt))
        row = {"gt_index": int(gt), "track_id": str(target["track_ids"][gt]), "distance_m": float(np.linalg.norm(boxes[gt,:2])),
               "valid_points": len(points), "full_horizon": len(points) == 8, "endpoint_valid": bool(valid[gt,-1]),
               "detected": query is not None, "selected": query in selected if query is not None else False}
        if len(points):
            stationary = np.linalg.norm(xy[gt, points]-boxes[gt, :2], axis=-1)
            row.update(stationary_ADE=float(stationary.mean()), stationary_last_valid_FDE=float(stationary[-1]),
                       stationary_FDE=float(stationary[-1]) if valid[gt,-1] else None,
                       motion_group="stationary" if stationary.max() <= .5 else "moving")
        else: row["motion_group"] = "no_valid_future"
        if query is not None:
            row.update(query=query, center_error=float(np.linalg.norm(prediction["vehicle_boxes"][query,:2]-boxes[gt,:2])))
            for name, output in (("head", prediction["vehicle_future_xy"][query]),
                                 ("joint", prediction["vehicle_xy"][selected[query]] if query in selected else None)):
                if output is not None and len(points):
                    error = np.linalg.norm(output[points]-xy[gt, points], axis=-1)
                    row.update({name+"_ADE":float(error.mean()), name+"_last_valid_FDE":float(error[-1]),
                                name+"_FDE":float(error[-1]) if valid[gt,-1] else None})
        rows.append(row)
    return rows, {"target_vehicles":len(indices), "predicted_vehicles":len(proposals), "detected_vehicles":len(detected),
                  "selected_vehicles":len(selected), "unmatched_candidates_including_unannotated_regions":len(proposals)-len(detected)}


def main():
    p = argparse.ArgumentParser(__doc__)
    for key in ("predictions", "vehicle-root", "index", "output"): p.add_argument("--"+key, required=True)
    a = p.parse_args()
    bank, labels, out = Path(a.predictions), Path(a.vehicle_root), Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    label_identity = json.loads((labels/"identity.json").read_text())["identity"]
    identity = {"prediction_identity_sha256":file_sha256(bank/"identity.json"), "label_identity":label_identity,
                "index_sha256":file_sha256(a.index), "existence":.2, "center_match_m":2.,
                "stationary_max_displacement_m":.5, "future_used_for_selection":False}
    atomic_json(out/"identity.json", identity)
    rows, scenes = [], []
    for scene in json.loads(Path(a.index).read_text()):
        token = scene["token"]
        with torch.serialization.safe_globals([np.core.multiarray.scalar, np.dtype, type(np.dtype("U16")), np.str_]):
            payload = torch.load(labels/"targets"/(token+".pt"), weights_only=True, map_location="cpu")
        if payload["identity"] != label_identity: raise ValueError("World label identity mismatch")
        pred, failure = None, None
        try:
            meta = json.loads((bank/"predictions"/(token+".json")).read_text())
            if meta["status"] != "ok": raise ValueError(meta["error"])
            path = bank/"predictions"/(token+".npz")
            if file_sha256(path) != meta["proposal_sha256"]: raise ValueError("Prediction changed")
            with np.load(path) as values: pred = {k:values[k] for k in values.files}
            current, counts = evaluate_scene(pred, payload["targets"])
        except Exception as error:
            failure = repr(error); current, counts = evaluate_scene(None, payload["targets"])
        for row in current: rows.append({"token":token,"log":scene["log"],"export_failure":failure,**row})
        scenes.append({"token":token,"log":scene["log"],"failure":failure,**payload["counts"],**counts})
    for name, values in (("vehicles",rows),("scenes",scenes)):
        fields = sorted(set().union(*(row.keys() for row in values))) if values else ["token"]
        with (out/(name+".csv")).open("w") as stream:
            writer=csv.DictWriter(stream,fields);writer.writeheader();writer.writerows(values)
    summary = {"scenes":len(scenes),"failed_scenes":sum(r["failure"] is not None for r in scenes),
        "source_vehicle_population":sum(r["source_vehicles"] for r in scenes),"supervised_targets":len(rows),
        "detected_targets":sum(r["detected"] for r in rows),"joint_selected_targets":sum(r["selected"] for r in rows),
        "metric_scope":"current ROI/FOV GT vehicles; source population separately retained; misses stay in coverage denominators",
        "cv_neighbor":"NOT_APPLICABLE: no legal current neighbor velocity"}
    for group in ("all", "stationary", "moving"):
        values = rows if group == "all" else [r for r in rows if r["motion_group"] == group]
        summary[group] = {"targets":len(values),"detected":sum(r["detected"] for r in values),"selected":sum(r["selected"] for r in values)}
        for key in ("head_ADE", "head_FDE", "joint_ADE", "joint_FDE", "stationary_ADE"):
            available = [r[key] for r in values if r.get(key) is not None]
            summary[group][key] = float(np.mean(available)) if available else None
            summary[group][key+"_denominator"] = len(available)
    atomic_json(out/"summary.json",summary)
    if summary["failed_scenes"]: raise RuntimeError("Failures retained in vehicle denominators")


if __name__ == "__main__": main()

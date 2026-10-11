"""Offline current-annotation visibility groups for shared instance rasters.

This reconstructs the exact sorted track-ID contract used by the existing
label builder. Future-only tracks are not called invisible at the current time.
Metrics use positive GT track/event pixels: report recall, never precision/IoU
on a region deliberately defined by positive labels.
"""
import numpy as np
import torch
from nuscenes.eval.detection.utils import category_to_detection_name
from starVLA.dataloader.structured_world.adapters import CLASS_NAMES


def current_instance_visibility(nusc, token):
    cursor = nusc.get('sample', token)
    tracks, current = set(), {}
    for t in range(7):
        for annotation_token in cursor['anns']:
            annotation = nusc.get('sample_annotation', annotation_token)
            if category_to_detection_name(annotation['category_name']) not in CLASS_NAMES: continue
            instance = annotation['instance_token']; tracks.add(instance)
            if t == 0: current[instance] = int(annotation['visibility_token'])
        if t != 6:
            if not cursor['next']: raise ValueError('Incomplete native six-keyframe future')
            cursor = nusc.get('sample', cursor['next'])
    return {i+1: current.get(track) for i, track in enumerate(sorted(tracks))}


def positive_counts(probability, mask):
    positives = int(mask.sum())
    true_positive = int(((probability >= .5) & mask).sum())
    return {'positive_GT_pixels': positives, 'true_positive': true_positive,
            'false_negative': positives-true_positive,
            'predicted_probability_sum': float(probability[mask].sum())}


def summarize(counts):
    n = counts['positive_GT_pixels']
    return {**counts, 'recall': counts['true_positive']/n if n else None,
            'mean_predicted_probability': counts['predicted_probability_sum']/n if n else None}


def score_visibility(prediction, labels, mapping):
    instances = labels['instances'].long()
    occupancy = labels['occupancy']
    valid = labels['occupancy_valid'].bool() & (occupancy != 255)
    if 'future_label_time_valid' in labels:
        valid = valid.clone(); valid[1:] &= labels['future_label_time_valid'][:, None, None].bool()
    identified = instances[(occupancy == 1) & valid]
    if identified.numel() and (identified.min() <= 0 or identified.max() > len(mapping)):
        raise ValueError('Native annotation tracks differ from the cache instance-ID contract')
    # -2 free/invalid, -1 no annotation at t0, 0 explicitly unknown visibility,
    # 1..4 native nuScenes annotation bands. Neither -1 nor 0 is "invisible".
    lookup = torch.full((len(mapping)+1,), -2, dtype=torch.int64)
    for index, visibility in mapping.items(): lookup[index] = -1 if visibility is None else visibility
    association = torch.full_like(instances, -2)
    occupied = valid & (occupancy == 1)
    association[occupied] = lookup[instances[occupied]]
    p0, pt = prediction['p0'][0], prediction['pt'][:, 0]
    pair = valid[:1] & valid[1:]
    enter = pair & (occupancy[:1] == 0) & (occupancy[1:] == 1)
    release = pair & (occupancy[:1] == 1) & (occupancy[1:] == 0)
    groups = {}
    for value, name in [(-1, 'not_annotated_at_current_time'), (0, 'unknown_annotation_visibility'),
                        (1, 'v0_40'), (2, 'v40_60'), (3, 'v60_80'), (4, 'v80_100')]:
        current_region = association[0] == value
        future_region = association[1:] == value
        groups[name+'/current_occupancy_positive'] = positive_counts(p0, current_region)
        groups[name+'/future_occupancy_positive'] = positive_counts(pt, future_region)
        groups[name+'/enter_positive'] = positive_counts(prediction['p_enter'][:, 0], enter & future_region)
        groups[name+'/release_positive'] = positive_counts(prediction['p_release'][:, 0], release & current_region[None])
    return groups

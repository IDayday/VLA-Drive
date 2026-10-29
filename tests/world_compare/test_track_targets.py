import ast
import hashlib
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import numpy.typing as npt
import pytest

from starVLA.rl.qwen_world_compare.track_targets import (
    SCHEMA, FutureTrackStore, audit_annotation_frame, build_scene_tracks,
    future_box_in_current_ego, nearest_future_frames, file_sha)


def frame(timestamp, tokens=('one',), centers=((10., 0.),), ego=(0., 0., 0.)):
    x, y, yaw = ego
    boxes = np.array([[a, b, 0., 4., 2., 1.5, .3] for a, b in centers])
    return dict(token=str(timestamp), timestamp=int(timestamp),
        lidar2ego_rotation=[1., 0., 0., 0.], lidar2ego_translation=[0., 0., 0.],
        ego2global_rotation=[np.cos(yaw / 2), 0., 0., np.sin(yaw / 2)],
        ego2global_translation=[x, y, 0.], anns=dict(gt_boxes=boxes,
            gt_names=np.array(['vehicle'] * len(boxes)), track_tokens=list(tokens)))


def current(frame):
    return dict(boxes=[[*box[:6], np.sin(box[6]), np.cos(box[6])]
                       for box in frame['anns']['gt_boxes']],
        classes=[0] * len(frame['anns']['gt_boxes']), valid=[True] * len(frame['anns']['gt_boxes']),
        coverage_valid=False)


def build(frames):
    return build_scene_tracks(frames, 0, {}, current(frames[0]),
                              verify_current_builder=False)


def test_track_identity_keeps_current_order_without_future_selection():
    start = frame(0, tokens=('one', 'two'), centers=((10., 0.), (12., 0.)))
    later = frame(1000000, tokens=('two', 'new', 'one'),
                  centers=((14., 0.), (9., 0.), (11., 0.)))
    row = build([start, later])
    assert row['track_tokens'] == ['one', 'two']
    np.testing.assert_allclose(row['future_states'][:2, 0, 0], [11., 14.])
    assert row['future_valid'][:2, 0].all()
    assert not row['future_valid'][2:].any()
    assert not row['future_valid'][:, 1:].any()
    assert row['future_time_valid'].tolist() == [True, False, False, False]


def test_missing_and_duplicate_future_track_are_invalid_not_zero_targets():
    start = frame(0)
    future = frame(1000000, tokens=('one', 'one'), centers=((11., 0.), (12., 0.)))
    row = build([start, future])
    assert not row['future_valid'].any()
    assert row['duplicate_future_track_matches'] == 1
    assert np.isfinite(row['future_states']).all()
    assert row['future_timestamps_us'].tolist() == [1000000, -1, -1, -1]
    assert row['actual_dt_s'].tolist() == [1., -1., -1., -1.]


def test_timestamp_nearest_matching_uses_measured_times_and_tolerance():
    frames = [frame(0), frame(980000), frame(2060000), frame(3070000), frame(3990000)]
    assert nearest_future_frames(frames, 0) == (1, 2, None, 4)
    row = build(frames)
    np.testing.assert_allclose(row['actual_dt_s'], [.98, 2.06, -1., 3.99])
    assert row['future_valid'][0].tolist() == [True, True, False, True]
    with pytest.raises(ValueError, match='strictly increasing'):
        nearest_future_frames([frame(0), frame(0)], 0)


def test_planar_global_current_transform_with_rotated_translated_ego():
    current_frame = frame(0, ego=(100., 200., np.pi / 2))
    future_frame = frame(1000000, ego=(101., 202., 0.))
    box = np.array([3., 4., 0., 4., 2., 1.5, .1])
    actual = future_box_in_current_ego(box, future_frame, current_frame)
    np.testing.assert_allclose(actual, [6., -4., 4., 2., np.sin(.1 - np.pi / 2),
                                      np.cos(.1 - np.pi / 2)], atol=1e-6)


def test_lidar_coordinate_mismatch_is_a_hard_error():
    value = frame(0)
    value['lidar2ego_translation'] = [1., 0., 0.]
    with pytest.raises(ValueError, match='not equivalent'):
        audit_annotation_frame(value)


def test_duplicate_current_track_refused():
    start = frame(0, tokens=('one', 'one'), centers=((10., 0.), (12., 0.)))
    with pytest.raises(ValueError, match='Ambiguous current track'):
        build([start])


def test_unknown_current_track_keeps_object_but_masks_future():
    start = frame(0, tokens=(None,))
    row = build([start, frame(1000000, tokens=(None,))])
    assert row['current_count'] == 1
    assert row['track_tokens'] == [None]
    assert row['current_objects']['valid'] == [True]
    assert not row['future_valid'].any()


def test_original_official_oriented_box_function_agrees_in_global():
    path = Path(__file__).parents[2] / 'navsim_v1.1/navsim/navsim/planning/scenario_builder/navsim_scenario_utils.py'
    names = {'normalize_angle', 'rotate_state_se2', 'gt_boxes_oriented_box'}
    source = ast.parse(path.read_text())
    extracted = ast.Module(body=[node for node in source.body
                          if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[])
    class State:
        def __init__(self, x, y, heading):
            self.x, self.y, self.heading = x, y, heading
    class Box:
        def __init__(self, center, length, width, height):
            self.center = center
    namespace = dict(np=np, npt=npt, List=list, StateSE2=State, OrientedBox=Box,
                     EgoState=SimpleNamespace, BoundingBoxIndex=SimpleNamespace(
                         X=0, Y=1, Z=2, LENGTH=3, WIDTH=4, HEIGHT=5, HEADING=6))
    exec(compile(extracted, str(path), 'exec'), namespace)
    f = frame(0, centers=((10., 2.),), ego=(100., 200., .7))
    box = f['anns']['gt_boxes'][0]
    native = namespace['gt_boxes_oriented_box']([box], SimpleNamespace(
        rear_axle=State(100., 200., .7)))[0].center
    origin = frame(0)
    global_output = future_box_in_current_ego(box, f, origin)
    np.testing.assert_allclose(global_output[:2], [native.x, native.y], atol=1e-5)
    np.testing.assert_allclose(global_output[4:], [np.sin(native.heading), np.cos(native.heading)], atol=1e-7)


def test_shard_reader_hash_population_and_lru(tmp_path):
    tokens = ['a', 'b']
    shards = {}
    for token in tokens:
        p = tmp_path / (token + '.pkl')
        p.write_bytes(pickle.dumps({token: {'token': token}}))
        shards[p.name] = dict(tokens=[token], sha256=file_sha(p))
    manifest = dict(schema=SCHEMA, count=2, shards=shards,
                    token_index={'a': 'a.pkl', 'b': 'b.pkl'})
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    (tmp_path / 'COMPLETE').write_text(json.dumps(dict(status='COMPLETE',
        manifest_sha256=file_sha(tmp_path / 'manifest.json'))))
    reader = FutureTrackStore(tmp_path, max_cached_logs=1)
    assert reader.get('a')['token'] == 'a'
    assert reader['b']['token'] == 'b'
    assert list(reader.cache) == ['b.pkl']
    (tmp_path / 'a.pkl').write_bytes(b'changed')
    with pytest.raises(ValueError, match='shard has changed'):
        reader['a']


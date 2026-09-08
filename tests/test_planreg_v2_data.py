import pytest
import torch
from navsim.agents.EpisodeDrive.planreg_v2.motion import CandidateKinematicsCodec, GTLogMotionBuilder, IntervalMotionEncoder
from navsim.agents.EpisodeDrive.planreg_v2.normalizers import measured_statistics, TrajectoryNormalizer, wrap_angle
from navsim.agents.EpisodeDrive.planreg_v2.targets import long_target


def test_causal_kinematics_and_gt_identity():
    times = torch.tensor([.5, 1., 1.5, 2., 2.5, 3., 3.5, 4.])
    p = torch.zeros(2, 64, 8, 3)
    p[..., 0] = times * 3
    state = torch.tensor([[3., 0., 0., 0.]]).expand(2, 4)
    codec = CandidateKinematicsCodec()
    out = codec(p, state, times, torch.ones(8, dtype=torch.bool))
    torch.testing.assert_close(out['physical_motion'][..., 6:], torch.zeros(2, 64, 8, 2))
    torch.testing.assert_close(out['motion_sequence'][:, 0], out['motion_sequence'][:, 1])
    p[..., 4:, :] += 100
    changed = codec(p, state, times, torch.ones(8, dtype=torch.bool))
    torch.testing.assert_close(changed['motion_sequence'][..., :4, :], out['motion_sequence'][..., :4, :])


def test_acceleration_midpoint_and_irregular_times():
    t = torch.tensor([.3, .7, 1.5, 4.])
    p = torch.zeros(1, 1, 4, 3)
    p[..., 0] = 2*t + t*t
    out = CandidateKinematicsCodec()(p, torch.tensor([[2., 0., 2., 0.]]), t, torch.ones(4, dtype=torch.bool))
    torch.testing.assert_close(out['physical_motion'][..., 6], torch.full((1,1,4), 2.))


def test_normalizer_roundtrip_floor_and_serialization(tmp_path):
    x = torch.randn(8, 3)
    x[:, 2] = torch.linspace(3., 4., 8)
    x[:, 2] = wrap_angle(x[:, 2])
    records = [('a', x, torch.ones(8, dtype=torch.bool)), ('a', x, torch.ones(8, dtype=torch.bool)),
               ('b', torch.zeros(8, 3), torch.ones(8, dtype=torch.bool))]
    n = measured_statistics(records, 'train', 'synthetic-unit-only')
    assert n.metadata['count'] == 2 and n.mean.shape == (8,3)
    torch.testing.assert_close(n.inverse(n.normalize(x)), x)
    n.save(tmp_path/'stats.json')
    restored = TrajectoryNormalizer.load(tmp_path/'stats.json').bfloat16()
    assert restored.mean.dtype == torch.float32
    torch.testing.assert_close(restored.mean, n.mean)
    with pytest.raises(ValueError):
        n.indices([.6])
    with pytest.raises(ValueError):
        measured_statistics(records, 'navtest', 'bad')


def test_logged_vector_frame_explicit_and_intervals():
    poses = torch.zeros(9,3); poses[:, 0] = torch.arange(9)*.5
    out = GTLogMotionBuilder('ego').build(poses, torch.ones(9,2), torch.zeros(9,2), torch.arange(9)*.5, torch.ones(9,dtype=torch.bool))
    torch.testing.assert_close(out['motion_sequence'][:,4:6], torch.ones(9,2))
    encoder = IntervalMotionEncoder(16)
    encoded, valid = encoder(out['motion_sequence'][None,1:], out['timestamps'][None,1:], out['valid_mask'][None,1:], [.5,1.5,4.])
    assert encoded.shape == (1,3,16) and valid.all()
    with pytest.raises(ValueError):
        GTLogMotionBuilder('guess')


def test_long_five_seconds_and_missing_not_extrapolated():
    t = torch.arange(11)*.5
    p = torch.zeros(11,3); p[:,0] = t
    long, valid = long_target(p,t,torch.ones(11,dtype=torch.bool))
    assert valid and long[-1,0] == 5
    _, valid = long_target(p[:9],t[:9],torch.ones(9,dtype=torch.bool))
    assert not valid


def test_logged_timestamp_jitter_is_not_false_missing_action():
    times=torch.tensor([[.4997,.9992,1.4988,1.9985,2.4985,2.9987,3.4988,3.9988]])
    encoder=IntervalMotionEncoder(16)
    output,coverage=encoder(torch.zeros(1,8,8),times,torch.ones(1,8,dtype=torch.bool),[.5,1.5,4.])
    assert coverage.all() and output.shape==(1,3,16)
    times[:,-1]=3.
    _,coverage=encoder(torch.zeros(1,8,8),times,torch.ones(1,8,dtype=torch.bool),[.5,1.5,4.])
    assert not coverage[0,-1]

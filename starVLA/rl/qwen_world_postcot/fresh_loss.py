"""Sealed from-generic-initialization loss screen, separate from the63k forks."""
import hashlib
import json
from pathlib import Path

from .contracts import fingerprint, file_sha
from .coordinate_objective import CoordinateObjective
from .loss_designs import LossDesign

ARMS = ('PHYSICAL', 'TIME_RESIDUAL', 'DELTA', 'HUBER')


def student_initialization_digest(model):
    """Hash all new student tensors; official Qwen files have a separate seal."""
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        if name.startswith('qwen.'):
            continue
        t = value.detach().contiguous().cpu()
        h.update(json.dumps([name, list(t.shape), str(t.dtype)]).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def objective(config):
    config = dict(config)
    if config.pop('kind') == 'PHYSICAL':
        return CoordinateObjective(**config)
    return LossDesign(**dict(kind=config.pop('loss_kind'), **config))


def read_registration(path, arm, config_path):
    row = json.loads(Path(path).read_text()); body = dict(row)
    if body.pop('sha256', None) != fingerprint(body):
        raise ValueError('Changed fresh-loss registration')
    if (row['schema'] != 'world_fresh_loss_IL_v1' or arm not in ARMS
            or row['seed'] != 42 or row['evaluation_seed'] != 2101
            or row['pilot_updates'] != 10000 or row['max_updates'] != 100000
            or row['interface_updates'] != 1000 or row['global_batch'] != 32
            or row['initialization'] != 'official_generic_Qwen_plus_new_World_seed42'
            or row['initial_driving_checkpoint'] is not None):
        raise ValueError('Fresh generic-init IL contract changed')
    if file_sha(config_path) != row['configs'][arm]['sha256']:
        raise ValueError('Fresh arm resolved configuration changed')
    for name in ARMS:
        objective(row['arms'][name])
    if row['selection'] != dict(max_survivors=2, primary='full_NAVTEST_10000',
            exclude_if_delta_points_lte=-.5, exclude_if_paired_CI_upper_lt=0.,
            ties='fixed_arm_order', no_decision_from_DEV=True):
        raise ValueError('Unregistered loss-screen selection change')
    return row


def verify_selection(path, registration, arm, resume):
    if not path or not resume:
        raise ValueError('Full continuation requires a selection receipt and native resume')
    row = json.loads(Path(path).read_text()); body = dict(row)
    if body.pop('sha256', None) != fingerprint(body):
        raise ValueError('Changed screen selection receipt')
    if (row['registration_sha256'] != registration['sha256']
            or arm not in row['selected'] or row['status'] != 'LOCKED'):
        raise ValueError('Arm is not selected for full continuation')
    native = row['parents'][arm]
    if file_sha(Path(native['checkpoint'])/'files.json') != native['files_sha256']:
        raise ValueError('Selected10k parent inventory changed')
    seal = json.loads((Path(resume)/'COMPLETE').read_text())
    identity = json.loads((Path(resume)/'identity.json').read_text())
    if (seal['updates'] < registration['pilot_updates']
            or identity['fresh_loss_experiment']['registration_sha256'] != registration['sha256']
            or identity['fresh_loss_experiment']['arm'] != arm):
        raise ValueError('Continuation is not this selected native lineage')


def select_survivors(results, registration):
    """Selection uses complete paired Test; never training loss or partial scores."""
    if set(results) != set(ARMS):
        raise ValueError('All four arms must complete the common screen')
    eligible = []
    for arm in ARMS:
        r = results[arm]
        if r['scenes'] != 12146 or r['invalid_scenes'] != 0:
            raise ValueError('Incomplete Test cannot select continuation')
        if r['delta_points'] <= -.5 and r['paired_CI_points'][1] < 0:
            continue
        eligible.append(arm)
    return sorted(eligible, key=lambda arm: (-results[arm]['PDMS'], ARMS.index(arm)))[:2]

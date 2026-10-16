"""Finite follow-up identities shared by existing dispatch/controllers.

This does not supply an actor, alter a recipe or authorize arbitrary run roots.
The two historical variants remain on their original dispatcher path.
"""
import json
from pathlib import Path


FOLLOWUPS = {
    'R_CANONICAL_X1': dict(
        root='/mnt/project/simwam-x-anchor-20261015',
        status='AUTHORIZED_BY_USER_GO_ON', config='canonical_x1'),
    'R_CANONICAL_LAST_ODE': dict(
        root='/mnt/project/simwam-terminal-ode-20261016',
        status='REGISTERED_UNDER_ACTIVE_USER_GOAL', config='canonical_last_ode'),
}


def require_registration(variant, root=None):
    spec = FOLLOWUPS[variant]
    expected = Path(spec['root'])
    root = expected if root is None else Path(root)
    if root.resolve() != expected.resolve():
        raise ValueError('Follow-up run root differs from registration')
    record = json.loads((root / 'AUTHORIZATION.json').read_text())
    if (record['status'] != spec['status'] or record['proposed_name'] != variant
            or Path(record['output_root']).resolve() != root.resolve()
            or record['budget'] != dict(optimizer_updates=20000,
                fresh_global_rollouts=5000, scene_exposures=40000,
                new_candidates=320000, training_world=8)):
        raise ValueError('Missing exact finite follow-up registration')
    return root


def add_arguments(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--followup-x1', action='store_true')
    group.add_argument('--followup-last-ode', action='store_true')


def selected_variant(args):
    if args.followup_x1:
        return 'R_CANONICAL_X1'
    if args.followup_last_ode:
        return 'R_CANONICAL_LAST_ODE'
    return None

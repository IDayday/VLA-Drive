"""Train-only, immutable candidate-group replay and frozen feature contracts."""
import hashlib
import json
import numpy as np
import torch


def array_hash(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def validate_train_tokens(tokens, manifest):
    if manifest.get('split') != 'trainval':
        raise ValueError('Replay requires an explicit trainval token manifest; Navtest labels cannot train scorer')
    allowed = set(manifest['tokens'])
    if len(tokens) != len(set(tokens)) or set(tokens) - allowed:
        raise ValueError('Replay contains duplicate or non-trainval tokens')
    if not allowed:
        raise ValueError('Empty trainval token manifest')


def group_identity(token, proposals, evaluator_sha256, checkpoint_sha256, scene_sha256):
    if np.asarray(proposals).shape != (64, 8, 3):
        raise ValueError('Replay must preserve the original complete 64-candidate group')
    if not np.isfinite(proposals).all():
        raise ValueError('Non-finite replay coordinates')
    payload = {'token': token, 'candidate_group_sha256': array_hash(proposals),
               'evaluator_sha256': evaluator_sha256, 'checkpoint_sha256': checkpoint_sha256,
               'scene_sha256': scene_sha256}
    payload['group_key'] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def validate_group(identity, token, proposals):
    if token != identity['token'] or array_hash(proposals) != identity['candidate_group_sha256']:
        raise RuntimeError('Stale replay labels: scene/candidate coordinates changed')


def categorize_group(predicted, truth, component_predictions):
    selected = int(np.argmax(predicted))
    official = truth[:, -1]
    result = []
    if predicted[selected] > .9 and official[selected] < .5:
        result.append('high_predicted_low_true')
    if official.max() > .9 and official[selected] < .5:
        result.append('high_oracle_low_selected')
    if ((component_predictions[:, 2] > .8) & (truth[:, 3] == 0)).any():
        result.append('ttc_hard_negative')
    ep_choice = component_predictions[:, 3].argmax()
    if truth[:, 2].max() - truth[ep_choice, 2] > .3:
        result.append('ep_large_misordering')
    return result or ['ordinary']


class FrozenScorerCacheContract:
    """Only upstream-frozen probes may cache representations. No formal cache."""
    def __init__(self, upstream_modules):
        self.modules = dict(upstream_modules)
        self.assert_frozen()
        self.versions = {f'{module_name}.{name}': (id(p), p._version)
                         for module_name, module in self.modules.items()
                         for name, p in module.named_parameters()}

    def assert_frozen(self):
        for module_name, module in self.modules.items():
            for name, p in module.named_parameters():
                if p.requires_grad:
                    raise RuntimeError(f'Cached scorer features invalid: upstream unfrozen {module_name}.{name}')

    def validate(self):
        self.assert_frozen()
        actual = {f'{module_name}.{name}': (id(p), p._version)
                  for module_name, module in self.modules.items() for name, p in module.named_parameters()}
        if actual != self.versions:
            raise RuntimeError('Cached scorer features invalid: upstream parameters changed')


def exact_cached_scorer_forward(action_head, scene, ego, proposals):
    embedded = action_head.pos_embed(proposals.detach().flatten(-2))
    decoded = action_head.scorer_attention(embedded, scene.detach()) + ego.detach()
    return action_head.scorer(proposals.detach(), decoded)[0]

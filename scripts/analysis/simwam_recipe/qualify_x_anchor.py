"""CPU contract check for the separately registered X1 follow-up."""
import argparse
import copy
import json
from pathlib import Path
import tempfile

from omegaconf import OmegaConf

from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.simwam_recipe.contracts import load_recipe, fingerprint
from scripts.analysis.simwam_recipe.diagnose_anchor_channels import qualify


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    configs = root / 'configs/flow_rl'
    source = load_recipe(configs / 'simwam_source_ddp.yaml')
    control = load_recipe(configs / 'simwam_canonical_ddp.yaml')
    candidate = load_recipe(configs / 'simwam_canonical_x1_ddp.yaml')
    expected = copy.deepcopy(control)
    expected['variant'] = 'R_CANONICAL_X1'
    expected['grpo']['train']['kl_dim_weights'] = [1., 1., .5, .5]
    assert candidate == expected
    rejected = []
    mutations = [('learning_rate', 1e-4), ('seed', 43), ('max_steps', 100),
                 ('gradient_clipping_effective', 1.), ('DDP_static_graph', True)]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'invalid.yaml'
        for key, value in mutations:
            changed = copy.deepcopy(candidate)
            changed[key] = value
            OmegaConf.save(OmegaConf.create(changed), path)
            try:
                load_recipe(path)
            except ValueError:
                rejected.append(key)
            else:
                raise AssertionError('Unregistered mutation accepted: ' + key)
        # The historical canonical contract must still reject the new weights.
        changed = copy.deepcopy(candidate)
        changed['variant'] = 'R_CANONICAL'
        OmegaConf.save(OmegaConf.create(changed), path)
        try:
            load_recipe(path)
        except ValueError:
            rejected.append('historical_canonical_anchor_change')
        else:
            raise AssertionError('Original contract was weakened')
    report = dict(status='PASS', original_source_identity=fingerprint(source),
                  original_control_identity=fingerprint(control),
                  candidate_identity=fingerprint(candidate),
                  only_registered_diff=True, rejected_mutations=rejected,
                  anchor_arithmetic=qualify(), new_candidates=0, optimizer_updates=0)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    atomic_json(output, report)
    print(json.dumps(report))


if __name__ == '__main__':
    main()

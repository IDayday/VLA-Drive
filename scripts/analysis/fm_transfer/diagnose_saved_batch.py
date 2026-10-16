"""Reuse the source gradient readout with the run's own new SFT and screen.

Read-only milestone analysis: no new chains, rewards, optimizer updates, or
counterfactual recipe. The original diagnostic entry retains its legacy base.
"""
import copy
import json
from pathlib import Path
import sys

from starVLA.rl.fm_transfer.contracts import require_dispatch
from starVLA.rl.simwam_recipe.contracts import file_sha, fingerprint, load_recipe


def validate_identity(identity, hard, collection):
    """Reject a historical base, intermediate SFT, or mismatched scene screen."""
    if (identity['schema'] != 'fm_transfer_source_RL_v1' or
            identity['purpose'] != 'formal' or identity['world'] != 8):
        raise ValueError('Only an actual eight-rank formal transfer run')
    cfg = copy.deepcopy(identity['recipe'])
    transfer = cfg.pop('supervised_transfer')
    if cfg != load_recipe('configs/flow_rl/simwam_source_ddp.yaml'):
        raise ValueError('Changed source recipe')
    parent = transfer['SFT_identity']
    expected_endpoint = {'A_DDP_TIME': 20000, 'B_SOURCE_ACTION_DIT': 100000}[transfer['variant']]
    if (parent['qualification'] or parent['updates'] != expected_endpoint or
            parent['config']['max_updates'] != expected_endpoint or
            parent['config']['variant'] != transfer['variant'] or
            parent['config']['architecture'] != transfer['architecture'] or
            parent['model_sha256'] != identity['SFT_sha256']):
        raise ValueError('Not this route\'s complete permanent SFT reference')
    if collection * 4 not in (5000, 10000, 15000, 20000):
        raise ValueError('Only the registered fixed RL milestones')
    body = dict(hard)
    seal = body.pop('sha256')
    screen = hard['screen_identity']
    screen_body = dict(screen)
    screen_seal = screen_body.pop('sha256')
    expected = dict(SFT_model_sha256=parent['model_sha256'],
                    SFT_identity_sha256=parent['identity_sha256'],
                    architecture=transfer['architecture'], seed=2101,
                    grid='native', threshold=.9,
                    statistic='one canonical v1.1 score per scene')
    if (fingerprint(body) != seal or fingerprint(screen_body) != screen_seal or
            hard['status'] != 'COMPLETE_FIXED_SFT_SINGLE_NATIVE_ODE_HARD_SUBSET' or
            hard['screen_contract'] != expected or screen['screen_contract'] != expected or
            screen['new_SFT'] != parent or screen['qualification_only'] or
            identity['data_stream_identity']['screen_contract'] != expected):
        raise ValueError('Saved behavior needs its own fixed SFT screen')
    return cfg, parent, screen


class TransferDiagnostic:
    require_dispatch = staticmethod(require_dispatch)

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('--hard-manifest', required=True)
        parser.add_argument('--sft-screen', required=True)

    def prepare(self, args, identity):
        if args.gradient_strata_registration:
            raise ValueError('Historical gradient-stratum registrations do not apply to new SFTs')
        hard = json.loads(Path(args.hard_manifest).read_text())
        if file_sha(args.hard_manifest) != identity['hard_manifest_sha256']:
            raise ValueError('Run hard population changed')
        self.cfg, self.parent, self.screen = validate_identity(identity, hard, args.collection)
        self.root = Path(args.sft_screen)
        if json.loads((self.root/'manifest.json').read_text()) != self.screen:
            raise ValueError('Different physical SFT screen')
        self.rows = {row['token']: row for row in hard['rows']}
        if set(self.rows) != set(hard['tokens']) or len(self.rows) != len(hard['rows']):
            raise ValueError('Malformed fixed hard population')
        self.behavior_update = args.collection*4-4
        self.training_source_files = identity['source_files']
        self.evidence = dict(schema='fm_transfer_saved_gradient_v1',
            own_SFT=self.parent, hard_manifest_sha256=file_sha(args.hard_manifest),
            SFT_screen_manifest_sha256=file_sha(self.root/'manifest.json'),
            adapter_source_sha256=file_sha(__file__), verified_scene_scores=[],
            interpretation='Source loss gradients after inner4, not Adam parameter displacement; one saved global batch only')

    def make_actor(self, cfg):
        from starVLA.rl.fm_transfer.rl_adapter import make_transfer_actor
        actor, parent = make_transfer_actor(self.parent['checkpoint'], self.cfg)
        if parent != self.parent:
            raise ValueError('Loaded reference differs from the training identity')
        repository = Path(__file__).resolve().parents[3]
        imported = {Path(module.__file__).resolve() for module in sys.modules.values()
                    if getattr(module, '__file__', None)}
        checked = {}
        for relative, expected in self.training_source_files.items():
            path = repository/relative
            if path.resolve() in imported:
                actual = file_sha(path)
                if actual != expected:
                    raise ValueError('Imported training dependency changed: '+relative)
                checked[relative] = actual
        self.evidence['imported_training_dependencies_sha256'] = checked
        return actor

    def validate_batch(self, rank, row):
        if row['behavior_version']['update'] != self.behavior_update or row['token'] not in self.rows:
            raise ValueError('Wrong behavior version or non-TRAIN scene')

    def baseline_metrics(self, token):
        path = self.root/'scenes'/token[:2]/(token+'.json')
        row = json.loads(path.read_text())
        fixed = self.rows[token]
        if (row['token'] != token or row['identity_sha256'] != self.screen['sha256'] or
                file_sha(path) != fixed['score_sha256'] or row['metrics'] != fixed['metrics'] or
                file_sha(path.with_suffix('.pt')) != row['payload_sha256']):
            raise ValueError('Own SFT scene score/payload changed')
        self.evidence['verified_scene_scores'].append(dict(token=token, score_sha256=file_sha(path)))
        return row['metrics']


if __name__ == '__main__':
    from scripts.analysis.simwam_recipe.diagnose_saved_batch import main
    main(runtime=TransferDiagnostic())

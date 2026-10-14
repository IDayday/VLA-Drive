"""Qualified batch-score revision with an explicit, preserved execution ancestry."""
import argparse
import json
import os
from pathlib import Path
import sys

from starVLA.rl.constrained_flow_noise.contracts import file_sha, fingerprint
from starVLA.rl.flow_grpo.transactions import atomic_json
from . import train_efficient as implementation
from . import train_efficient_cast as parent
from .batched_canonical_score import initialize_worker


_parent_identity = parent.source_identity


def source_identity():
    value = _parent_identity()
    value['parent_execution_sha256'] = value.pop('sha256')
    value['schema'] = 'cfn_execution_independent_canonical_batch_v3'
    value['sources'].update({name: file_sha(Path(__file__).with_name(name)) for name in
                             ('train_efficient_batch.py', 'batched_canonical_score.py')})
    value['scoring'] = 'Official batch simulator/submetrics; unchanged official aggregate on each [fixed reference, one candidate] pair'
    value['sha256'] = fingerprint(value)
    return value


def main():
    if any(x in sys.argv[1:] for x in ('-h', '--help')):
        parent.main()
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--acceptance')
    known, args = parser.parse_known_args()
    identity, previous = source_identity(), _parent_identity()
    if '--qualification' not in args:
        if not known.acceptance:
            raise ValueError('Batch scoring requires native upgrade acceptance')
        proof = json.loads(Path(known.acceptance).read_text())
        if (proof.get('status') != 'PASS' or proof.get('execution_sha256') != identity['sha256']
                or proof.get('parent_execution_upgrade_exact') is not True):
            raise ValueError('Unqualified batch-score execution transition')
    output = Path(args[args.index('--output') + 1])
    if int(os.environ.get('RANK', '0')) == 0:
        record = output / 'execution_identity.json'
        if record.exists():
            old = json.loads(record.read_text())
            if old != identity:
                if old != previous or '--resume' not in args:
                    raise ValueError('Unexpected previous execution; never silently relabel')
                history = output / 'execution_history' / old['sha256']
                history.mkdir(parents=True, exist_ok=True)
                for name in ('execution_identity.json', 'execution_acceptance.json'):
                    source = output / name
                    if source.exists():
                        target = history / name
                        if target.exists() and target.read_bytes() != source.read_bytes():
                            raise ValueError('Execution history already differs')
                        if not target.exists():
                            target.write_bytes(source.read_bytes())
                atomic_json(output / 'execution_upgrade.json', dict(
                    previous=old['sha256'], current=identity['sha256'],
                    native_resume=args[args.index('--resume') + 1],
                    acceptance=known.acceptance, scope='Execution only; all scientific checkpoint checks retained'))
                atomic_json(record, identity)
    native_resume = implementation.train.resume_checkpoint
    def resume(*pos, **kw):
        state = native_resume(*pos, **kw)
        old = state.get('execution_adapter')
        if old == previous:
            state.setdefault('execution_ancestry', []).append(old)
            state['execution_adapter'] = identity
        elif old is not None and old != identity:
            raise ValueError('Checkpoint execution ancestry is not the qualified parent')
        return state
    implementation.train.resume_checkpoint = resume
    implementation.initialize_worker = initialize_worker
    parent.source_identity = source_identity
    parent.main()


if __name__ == '__main__':
    main()

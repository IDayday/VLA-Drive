"""Qualified selected-parent RL50k, followed by its full native Navtest."""
import json
from pathlib import Path
import subprocess
import sys

from starVLA.rl.qwen_world_postcot.checkpoint import atomic_json
from starVLA.rl.qwen_world_postcot.contracts import file_sha, fingerprint
from tools.qwen_world_postcot.start_scope_rl import main as train


def main():
    registration = Path(sys.argv[sys.argv.index('--registration')+1])
    cfg = json.loads(registration.read_text()); root = Path(cfg['root'])
    train()  # Existing real16-update acceptance, mid-inner resume, then fresh50k.
    checkpoint = Path(cfg['formal_run'])/'checkpoints/update050000_inner0'
    seal = json.loads((checkpoint/'COMPLETE').read_text())
    identity = json.loads((checkpoint/'identity.json').read_text())
    if seal['updates'] != 50000 or seal['phase'] != 'READY' or seal['next_inner'] != 0:
        raise ValueError('Navtest must follow the committed full RL50k endpoint')
    parent = json.loads(Path(cfg['parent_selection']).read_text())
    evaluation = root/'NAVTEST_RL050000'
    evaluation.mkdir(exist_ok=False)
    pin = dict(label='MEMORY_IL070000_RL050000',kind='RL',updates=50000,
               checkpoint=str(checkpoint),parent_checkpoint=parent['selected']['checkpoint'],
               files_sha256=seal['files_sha256'],native_identity_sha256=fingerprint(identity),
               user_instruction=parent['user_instruction'])
    atomic_json(evaluation/'PINNED.json',dict(schema='world_checkpoint_sweep_NAVTEST_v1',
        models={pin['label']:pin},max_concurrent_models=1,seed=2101,full_population=12146,
        reference_evaluation=cfg['parent_evaluation'],observed_Test=True))
    atomic_json(root/'PIPELINE_STATE.json',dict(status='RL_COMPLETE_NAVTEST_RUNNING',
        RL_updates=50000,evaluation=str(evaluation),registration_sha256=file_sha(registration)))
    subprocess.run(['/usr/bin/python3','-u','-m','tools.qwen_world_postcot.latest_snapshot_owner',
                    '--root',str(evaluation),'--report-dir',cfg['evaluation_report']],check=True)
    if not (evaluation/'COMPLETE.json').exists():
        raise RuntimeError('RL complete, but Navtest failed; inspect native evidence')
    atomic_json(root/'PIPELINE_STATE.json',dict(status='COMPLETE',RL_updates=50000,
        evaluation=str(evaluation),all_training_and_evaluation_workers_exited=True))


if __name__ == '__main__':
    main()

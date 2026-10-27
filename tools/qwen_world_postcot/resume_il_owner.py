"""IL-only continuation using the immutable, qualified World campaign helpers.

The native trainer is unchanged. This owner stops at IL100k and never invokes
checkpoint selection, new RL training, or additional inference seeds.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time


def reconcile_journal(path, boundary, archive):
    """Retain the exact abandoned attempt before publishing its legal prefix."""
    path, archive = Path(path), Path(archive)
    # A preparation crash can occur between two ranks. Reuse an archived
    # original only after proving that the current file is its exact prefix.
    reused = archive.exists()
    raw = archive.read_bytes() if reused else path.read_bytes()
    prefix, suffix, malformed = [], [], []
    for line in raw.splitlines(keepends=True):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if len(prefix) != boundary:
                raise ValueError('Corrupt journal before the saved boundary')
            # Quota failures can leave damaged rows inside the abandoned
            # suffix. Preserve their exact bytes; they are never active data.
            malformed.append(line.hex())
            continue
        if row['updates'] <= boundary:
            if row['updates'] != len(prefix) + 1 or suffix:
                raise ValueError('Journal prefix is not continuous')
            if row['cursor'] != row['updates'] * 32:
                raise ValueError('Scene exposure does not match the saved update')
            prefix.append(line)
        else:
            suffix.append(row)
    if len(prefix) != boundary:
        raise ValueError('Missing complete saved-boundary journal prefix')
    expected = b''.join(prefix)
    if reused:
        if path.read_bytes() != expected:
            raise ValueError('Current journal does not match the archived legal prefix')
    else:
        archive.parent.mkdir(parents=True, exist_ok=True)
        os.link(path, archive)
        temporary = path.with_name(path.name + '.resume-prefix.' + str(os.getpid()))
        with temporary.open('xb') as stream:
            stream.write(expected); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    return dict(original_sha256=hashlib.sha256(raw).hexdigest(), archive=str(archive),
                retained_updates=boundary, aborted_updates=[row['updates'] for row in suffix],
                aborted_observed_worker_seconds=sum(row.get('seconds', 0.) for row in suffix),
                corrupt_aborted_rows=malformed, previous_reconciliation_reused=reused,
                physical_attempt_evidence_preserved=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--registration', required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.registration).read_text())
    root, source = Path(cfg['root']), Path(cfg['source'])
    # The command is executed with PYTHONPATH=<original frozen snapshot>.
    from tools.qwen_world_postcot import campaign as c
    if Path(c.__file__).resolve() != source / 'tools/qwen_world_postcot/campaign.py':
        raise ValueError('Controller helpers must come from the original snapshot')
    lock = (c.ROOT / 'campaign/owner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / 'OWNER_LAUNCH.json').exists():
        raise FileExistsError('Do not repeat an already launched continuation')
    if c.file_sha(Path(cfg['checkpoint']) / 'files.json') != cfg['checkpoint_files_sha256']:
        raise ValueError('Registered recovery inventory changed')
    previous = json.loads((root / 'PREVIOUS_TRAINING_RESULT.json').read_text())
    if (previous['returncode'] == 0 or previous['cwd'] != str(source)
            or cfg['command'] != previous['command'] + ['--resume', cfg['checkpoint']]
            or 'tools.qwen_world_postcot.train_il' not in cfg['command']):
        raise ValueError('Continuation must be the original IL command plus exact native resume')
    if (c.file_sha(root / 'PREFLIGHT.json') != cfg['preflight_sha256']
            or json.loads((root / 'PREFLIGHT.json').read_text())['status'] != 'PASS'):
        raise ValueError('Missing sealed preflight qualification')
    okay, receipt = c.accepted(str(source))
    if not okay:
        raise ValueError('Original training acceptance no longer holds: ' + str(receipt))
    state = json.loads((root / 'PREVIOUS_CAMPAIGN_STATE.json').read_text())
    c._same_source(str(source), state)
    c.poll_jobs(state)
    if c.active(state, name='formal_IL'):
        raise ValueError('Another IL invocation still owns its training processes')
    before = c.probe(cfg['host']); time.sleep(2); after = c.probe(cfg['host'])
    expected = cfg['gpu_uuids']
    if any([gpu['uuid'] for gpu in sample['gpus']] != expected or sample['owners']
           or min(gpu['free_MiB'] for gpu in sample['gpus']) < cfg['minimum_free_MiB']
           or any(c.gpu_activity_issue(gpu, sample['owners']) for gpu in sample['gpus'])
           for sample in (before, after)):
        raise ValueError('Idle eight-GPU admission changed; no occupant is evicted')
    route = dict(host=cfg['host'], gpus=list(range(8)), uuids=expected,
                 share=None, minimum_free_MiB=cfg['minimum_free_MiB'])
    invocation = c.dispatch(state, str(source), 'formal_IL', cfg['command'], route, role='training')
    owner = dict(pid=os.getpid(), host=c.socket.gethostname(),
                 start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
                 root=str(root), source=str(source), at=time.time(), invocation=invocation,
                 state_file=str(root / 'STATE.json'), terminal_file=str(root / 'OWNER_RESULT.json'),
                 original_campaign_not_restarted=True, automatic_RL=False, evaluation_seed=2101)
    c.atomic_json(root / 'OWNER_LAUNCH.json', owner)
    c.atomic_json(c.ROOT / 'ACTIVE_IL_OWNER.json', owner)
    state.update(resume_owner=owner, controller_scope='IL_only_through100k_no_automatic_RL')
    while True:
        c.poll_jobs(state)
        training = state['jobs'][invocation]
        state['stage'] = 'IL_RUNNING' if training['status'] == 'RUNNING' else training['status']
        if training['status'] in ('FAILED', 'REFUSED_RESOURCE'):
            c.atomic_json(root / 'STATE.json', state)
            c.atomic_json(root / 'OWNER_RESULT.json', dict(status='REVIEW_REQUIRED',
                          training=training, at=time.time(), automatic_retry=False))
            return
        # These are the original asynchronous data and diagnostics routines.
        # In particular do not silently freeze the30k candidate bank to100k.
        for name, function in [('candidate_refresh', c.advance_refreshes),
                               ('DEV_diagnostics', lambda st, src, rec: c.advance_dev_window(st, src, rec, 'IL'))]:
            try:
                function(state, str(source), receipt)
                state.pop(name + '_controller_error', None)
            except Exception as error:
                state[name + '_controller_error'] = repr(error)
        state['last_tick_unix'] = time.time()
        c.atomic_json(root / 'STATE.json', state)
        if training['status'] == 'COMPLETE':
            header = c.checkpoint_header(c.progress(c.IL_RUN)['checkpoint'], 'IL')
            if header['updates'] != 100000:
                raise ValueError('IL exited before the registered100k endpoint')
            c.atomic_json(root / 'OWNER_RESULT.json', dict(status='IL_COMPLETE100000',
                          checkpoint=header['checkpoint'], at=time.time(), automatic_RL=False,
                          remaining_auxiliary_jobs=c.active(state)))
            return
        time.sleep(60)


if __name__ == '__main__':
    main()

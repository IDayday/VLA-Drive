"""Add evaluation observations without changing a live training supervisor.

Native rolling checkpoints are temporarily pinned with same-filesystem hardlinks.
Their identities and payloads are unchanged; only the evaluation storage route
differs. The existing dispatcher remains the sole GPU/scoring queue owner.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
from types import SimpleNamespace
import uuid


SOURCE = '0c54f1c49e7ed2a76b22826b654d6c72625f5b6d'
SCHEMA = 'structured_world_navsim_navtest_cadence_v1'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def require(value, message):
    if not value:
        raise ValueError(message)


def load_policy(path):
    policy = read(path)
    require(policy['schema'] == SCHEMA and policy['population'] == 'navtest'
            and policy['expected_scenes'] == 12146
            and policy['sampling_seed'] == 42
            and policy['training_recipe_change'] is False,
            'Invalid cadence evaluation contract')
    targets = policy['observation_updates']
    require(targets == [25000, 50000, 60000, 70000, 80000, 90000, 100000],
            'Unexpected registered observation cadence')
    require(sha(policy['base_policy_path']) == policy['base_policy_sha256'],
            'Base evaluation policy changed')
    require(len(policy['runs']) == 4 and len({r['group'] for r in policy['runs']}) == 4,
            'Four distinct registered G runs required')
    registered = {r['training_identity']: r['group'] for r in policy['runs']}
    seen = set()
    for extra in policy.get('additional_observations', []):
        identity, update = extra.get('training_identity'), extra.get('updates')
        require(identity in registered and extra.get('group') == registered[identity]
                and type(update) is int and 0 < update <= policy.get('training_endpoint', 100000)
                and bool(extra.get('user_instruction')) and bool(extra.get('requested_utc')),
                'Invalid explicit additional observation')
        require((identity, update) not in seen and update not in targets,
                'Duplicate explicit additional observation')
        seen.add((identity, update))
    return policy


def observation_updates(policy, identity):
    """A user-requested extra checkpoint belongs only to its registered run."""
    return sorted(set(policy['observation_updates']) | {
        extra['updates'] for extra in policy.get('additional_observations', [])
        if extra['training_identity'] == identity})


def eligible(policy, request):
    """Keep completed history; suppress new off-cadence requests for these runs."""
    registered = {r['training_identity'] for r in policy['runs']}
    return (request['training_identity'] not in registered
            or request['updates'] in observation_updates(policy, request['training_identity']))


def checkpoint(run, update, identity):
    for prefix in ('milestone', 'latest'):
        folder = run / 'checkpoints' / f'{prefix}_{update:07d}'
        complete_path = folder / 'COMPLETE.json'
        if not complete_path.exists():
            continue
        try:
            complete = read(complete_path)
        except FileNotFoundError:
            continue  # Native rolling retention can race this read.
        require(complete['completed'] == update and complete['identity'] == identity
                and complete['tag'] == folder.name and complete['ranks'] == 8,
                'Native checkpoint boundary/identity differs')
        files = complete['file_sha256']
        require(len(files) == 17 and all(Path(name).name == name and name.endswith('.pt')
                                        for name in files), 'Native checkpoint inventory differs')
        if all((folder / name).is_file() for name in files):
            return folder, complete
    return None


def pin(policy, registration, update, folder, complete):
    key = registration['group'] + f'_{update:07d}'
    root = Path(policy['pins_root']) / key
    native_run = root / 'native_run'
    tag = folder.name
    if (root / 'PIN.json').exists():
        record = read(root / 'PIN.json')
        require(record['source_run'] == registration['run'] and record['source_tag'] == tag
                and record['checkpoint_complete_sha256'] == sha(folder / 'COMPLETE.json')
                and not (root / 'RELEASED.json').exists(), 'Existing pin differs or was released')
        return native_run, tag
    require(not root.exists(), 'Unregistered checkpoint pin exists')
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = root.with_name('.' + key + '.' + uuid.uuid4().hex + '.partial')
    destination = staging / 'native_run' / 'checkpoints' / tag
    destination.mkdir(parents=True)
    metadata = Path(registration['run']) / 'identity.json'
    complete_bytes = (folder / 'COMPLETE.json').read_bytes()
    require(json.loads(complete_bytes) == complete, 'Native boundary changed before pin')
    linked = {}
    try:
        (staging / 'native_run' / 'identity.json').write_bytes(metadata.read_bytes())
        require(sha(staging / 'native_run' / 'identity.json') == registration['identity_sha256'],
                'Source training registration changed')
        for name in complete['file_sha256']:
            source = folder / name
            require(not source.is_symlink(), 'Native payload must be a regular file')
            os.link(source, destination / name)  # Fail on EXDEV; never silently copy weights.
            new = (destination / name).stat()
            try:
                old = source.stat()
            except FileNotFoundError:
                old = new  # A successful link retains the inode after source pruning.
            require((old.st_dev, old.st_ino, old.st_size) == (new.st_dev, new.st_ino, new.st_size),
                    'Hardlink payload identity differs')
            linked[name] = {'device': new.st_dev, 'inode': new.st_ino, 'bytes': new.st_size,
                            'sha256': complete['file_sha256'][name]}
        # Validate retained inodes, not source paths that the trainer may prune.
        for name, expected in complete['file_sha256'].items():
            require(sha(destination / name) == expected, 'Pinned payload hash differs: ' + name)
        (destination / 'COMPLETE.json').write_bytes(complete_bytes)
        atomic(staging / 'PIN.json', {
            'schema': 'navtest_native_hardlink_pin_v1', 'key': key,
            'source_run': registration['run'], 'source_tag': tag,
            'training_identity': registration['training_identity'],
            'source_identity_sha256': registration['identity_sha256'],
            'checkpoint_complete_sha256': hashlib.sha256(complete_bytes).hexdigest(),
            'updates': update, 'files': linked, 'additional_payload_bytes_copied': 0,
            'release_condition': 'valid full q0/q_final Navtest completion after owned workers exit',
            'created_utc': datetime.now(timezone.utc).isoformat()})
        staging.rename(root)
    except BaseException:
        # Only these new hardlinks are removed; original checkpoint paths are untouched.
        shutil.rmtree(staging)
        raise
    return native_run, tag


def release_pin(policy, key, request, output):
    root = Path(policy['pins_root']) / key
    record_path = root / 'PIN.json'
    if not record_path.exists() or (root / 'RELEASED.json').exists():
        return
    record = read(record_path)
    proof_path = output / 'jobs' / key / 'OWNED_WORKERS_EXITED.json'
    if not proof_path.exists():
        return  # Preserve payloads after an interrupted controller until exit is proved.
    proof = read(proof_path)
    require(proof['key'] == key and proof['training_identity'] == request['training_identity']
            and proof['checkpoint_files_sha256'] == request['checkpoint_files_sha256']
            and proof['all_gpu_launchers_exited'] is True and proof['scorer_exited'] is True,
            'Owned worker exit proof differs')
    expected_run = root / 'native_run'
    require(Path(request['run']) == expected_run and request['tag'] == record['source_tag']
            and request['training_identity'] == record['training_identity'], 'Refusing foreign pin release')
    folder = expected_run / 'checkpoints' / request['tag']
    for name, saved in record['files'].items():
        path = folder / name
        if path.exists():
            st = path.stat()
            require((st.st_dev, st.st_ino, st.st_size) ==
                    (saved['device'], saved['inode'], saved['bytes']), 'Pinned inode changed')
    # Remove only this cadence job's temporary FP32 export, retaining its receipt.
    export_dir = output / 'FP32_exports' / key
    export_file = export_dir / 'model_fp32.pt'
    released_export = False
    if export_file.exists():
        receipt_path = export_dir / 'FP32_EXPORT_COMPLETE.json'
        require(receipt_path.exists(), 'FP32 export identity receipt absent')
        receipt = read(receipt_path)
        require(receipt['training_identity'] == request['training_identity']
                and receipt['checkpoint'] == request['tag']
                and receipt['checkpoint_files_sha256'] == request['checkpoint_files_sha256'],
                'Refusing foreign FP32 export cleanup')
        require(not export_file.is_symlink(), 'FP32 export is not a regular owned file')
    for name in record['files']:
        (folder / name).unlink(missing_ok=True)
    if export_file.exists():
        export_file.unlink()
        released_export = True
    atomic(root / 'RELEASED.json', {'key': key, 'released_utc': datetime.now(timezone.utc).isoformat(),
        'native_hardlinks_released': len(record['files']), 'temporary_FP32_export_released': released_export,
        'source_checkpoint_paths_modified': False, 'score_complete': str(output / 'jobs' / key / 'scores/COMPLETE.json'),
        'metadata_and_results_retained': True})


def poll(policy, output, enqueue, valid_completion):
    """Called by the sole queue owner; it never launches or signals training."""
    output = Path(output)
    observations = []
    for registration in policy['runs']:
        run = Path(registration['run'])
        require(sha(run / 'identity.json') == registration['identity_sha256'], 'Formal run identity changed')
        source = read(run / 'identity.json')
        require(source['identity'] == registration['training_identity']
                and source['training_source_sha'] == SOURCE
                and source['scope'] == 'formal' and source['dataset'] == 'navsim'
                and source['config']['structured_world']['group'] == registration['group'],
                'Unexpected cadence training source')
        latest = run / 'checkpoints' / 'latest'
        latest_complete = run / 'checkpoints' / latest.read_text().strip() / 'COMPLETE.json' if latest.exists() else None
        try:
            saved_update = read(latest_complete)['completed'] if latest_complete else 0
        except FileNotFoundError:
            saved_update = 0
        for update in observation_updates(policy, registration['training_identity']):
            key = registration['group'] + f'_{update:07d}'
            request_path = output / 'requests' / (key + '.json')
            completion = output / 'completed' / (key + '.json')
            status = {'key': key, 'updates': update}
            if completion.exists():
                require(request_path.exists(), 'Completed job lacks original request')
                request = read(request_path)
                require(read(completion).get('valid') is True, 'Invalid completion marker')
                pin_root = Path(policy['pins_root']) / key
                if (pin_root / 'PIN.json').exists() and not (pin_root / 'RELEASED.json').exists():
                    require(valid_completion(output / 'jobs' / key, request), 'Invalid full score bank')
                    release_pin(policy, key, request, output)
                status['status'] = 'COMPLETE'
            elif request_path.exists():
                status['status'] = 'QUEUED_OR_RUNNING'
            elif update in (25000, 50000, 100000):
                status['status'] = 'ORIGINAL_SUPERVISOR_OWNS_REQUEST'
            else:
                pin_root = Path(policy['pins_root']) / key
                recovered_pin = (pin_root / 'PIN.json').exists()
                if recovered_pin:
                    saved_pin = read(pin_root / 'PIN.json')
                    require(saved_pin['source_run'] == registration['run']
                            and saved_pin['source_identity_sha256'] == registration['identity_sha256']
                            and not (pin_root / 'RELEASED.json').exists(), 'Invalid interrupted pin')
                    found = checkpoint(pin_root / 'native_run', update, source['identity'])
                    require(found is not None and sha(found[0] / 'COMPLETE.json') ==
                            saved_pin['checkpoint_complete_sha256'], 'Interrupted pin payloads missing')
                else:
                    found = checkpoint(run, update, source['identity'])
                if found is None:
                    status['status'] = ('PAST_CHECKPOINT_UNAVAILABLE' if saved_update > update
                                        else 'WAITING_FOR_NATIVE_CHECKPOINT')
                    status['latest_saved_update'] = saved_update
                else:
                    folder, complete = found
                    if recovered_pin:
                        pinned_run, tag = pin_root / 'native_run', folder.name
                    else:
                        try:
                            pinned_run, tag = pin(policy, registration, update, folder, complete)
                        except FileNotFoundError:
                            status['status'] = 'CHECKPOINT_PRUNED_DURING_PIN'
                            observations.append(status)
                            continue
                    enqueue(SimpleNamespace(run=pinned_run, tag=tag, output=output, priority=100))
                    status['status'] = 'QUEUED_PINNED_NATIVE_CHECKPOINT'
                    status['source_checkpoint'] = str(folder)
                    status['pinned_run'] = str(pinned_run)
            observations.append(status)
    atomic(output / 'CADENCE_STATE.json', {'schema': SCHEMA,
        'observed_utc': datetime.now(timezone.utc).isoformat(),
        'observation_updates': policy['observation_updates'], 'observations': observations,
        'historical_off_cadence_results_preserved': True, 'training_processes_changed': False})


class Observer:
    """One serial CPU capture loop, owned by the dispatcher lock holder.

Long GPU inference or resource qualification cannot delay checkpoint capture.
This thread only publishes requests; it never launches a second GPU dispatcher.
"""
    def __init__(self, policy, output, enqueue, valid_completion):
        self.error = None
        self.stopped = threading.Event()

        def run():
            try:
                while not self.stopped.is_set():
                    poll(policy, output, enqueue, valid_completion)
                    self.stopped.wait(30)
            except BaseException as error:
                self.error = error
                atomic(Path(output) / 'CADENCE_OBSERVER_ERROR.json', {
                    'error': repr(error), 'observed_utc': datetime.now(timezone.utc).isoformat(),
                    'training_processes_changed': False})

        self.thread = threading.Thread(target=run, name='navtest-checkpoint-capture', daemon=True)
        self.thread.start()

    def close(self):
        self.stopped.set()
        self.thread.join()  # Keep the dispatch lock until an in-flight pin is published/rolled back.

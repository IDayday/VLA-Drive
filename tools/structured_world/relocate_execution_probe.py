"""Copy only the new zt4 recollected native triplets; optionally release their copies.

Standard library + rsync/SSH, no tensor loading. Hash integrity is not numerical
acceptance. Failed attempts are retained, never retried or cleaned automatically.
"""
import argparse
import ctypes
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import stat
import subprocess
import sys
import uuid

SOURCE_HOST = 'training-vlawm-zt4'
SOURCE_ROOT = Path('/var/tmp/structured-world-fgtr-execution-20261011/native_zt4_recollected_p0_v1')
ARCHIVE_PARENT = Path('/var/tmp')
DEFAULT_ARCHIVE = ARCHIVE_PARENT/'structured-world-fgtr-execution-20261011/native_recollected_archive'
ARCHIVE_NAMES = ('native_recollected_archive', 'native_zt4_recollected_archive_p0_v1')
PARENT_HOST = 'training-vla-zt-worker-0'
TRAINING_SOURCE = 'be5e72248a46afd43e1bf279c21134c391c5edad'
ORIGINS = {'G1_mid': 25000, 'G3_mid': 50000}
ROLES = ('A', 'AA', 'B')
POINTS = (1, 32, 100, 250, 500)
PAYLOADS = frozenset({'mp_rank_00_model_states.pt'} |
    {f'rng_rank{r}.pt' for r in range(8)} |
    {f'zero_pp_rank_{r}_mp_rank_00_optim_states.pt' for r in range(8)})
RESERVE_BYTES = 145*(1 << 30)
GROUP_UPPER_BYTES = 678*(1 << 30)
ARCHIVE_UPPER_BYTES = 1380*(1 << 30)
HEX = re.compile(r'[0-9a-f]{64}\Z')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def no_links(path):
    """Reject lexical traversal, symlink ancestors and non-directory ancestors."""
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts and '.' not in path.parts,
            'Absolute path without traversal required')
    for p in [*reversed(path.parents), path]:
        if not p.exists() and not p.is_symlink():
            continue
        s = p.lstat()
        require(not stat.S_ISLNK(s.st_mode), 'Symlink forbidden: '+str(p))
        if p != path:
            require(stat.S_ISDIR(s.st_mode), 'Non-directory ancestor: '+str(p))
    return path


def source_path(group):
    require(group in ORIGINS, 'Only G1_mid/G3_mid are authorized')
    return no_links(SOURCE_ROOT/group)


def archive_path(root, group):
    require(group in ORIGINS, 'Unknown group')
    root = no_links(root)
    require(root.is_relative_to(ARCHIVE_PARENT) and root != ARCHIVE_PARENT and
            root.name in ARCHIVE_NAMES, 'Archive must be the local /var/tmp private recollected archive')
    require(not root.is_relative_to(SOURCE_ROOT) and not SOURCE_ROOT.is_relative_to(root),
            'Source/archive overlap forbidden')
    return root/group


def file_hash(path, *, durable=False, stamps=None):
    path = no_links(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, 'Expected private regular file: '+str(path))
        h = hashlib.sha256()
        while True:
            block = os.read(fd, 8 << 20)
            if not block:
                break
            h.update(block)
        if durable:
            os.fsync(fd)
        after = os.fstat(fd)
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), 'File changed while hashing')
        if stamps is not None:
            stamps[str(path)] = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        return {'sha256': h.hexdigest(), 'bytes': before.st_size}
    finally:
        os.close(fd)


def read(path):
    no_links(path)
    return json.loads(Path(path).read_text())


def read_text(path):
    no_links(path)
    return Path(path).read_text()


def sync_dir(path):
    fd = os.open(no_links(path), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def owned_archive_bytes(root):
    total = 0
    for p in Path(root).rglob('*'):
        no_links(p)
        if p.is_file():
            total += p.stat().st_size
    return total


def write_new(path, value):
    """No overwrite; marker visible before a destructive action and fsynced."""
    no_links(path)
    with open(path, 'x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    sync_dir(Path(path).parent)


def inspect_group(root, group, *, durable=False, stamps=None):
    """The tests use small bytes but the full real 3 x 5 x 17 inventory."""
    root = no_links(root)
    require(group in ORIGINS and root.is_dir(), 'Missing authorized group')
    require(not any((root/r/'RELOCATED.json').exists() for r in ROLES), 'Source payload release already begun; no retry')
    require(not any((root/r/'RELOCATION_FAILED.json').exists() for r in ROLES), 'Previous failed relocation; no retry')
    require({p.name for p in root.iterdir() if p.is_dir()} == set(ROLES), 'Exactly the three new role runs required')
    identities, banks, payload_paths = {}, {}, set()
    for role in ROLES:
        run = root/role
        no_links(run)
        ident, done, ex, proc, boundary = [read(run/n) for n in
            ('identity.json', 'TRAINING_COMPLETE.json', 'PROCESS_EXIT.json', 'PROCESS.json', 'RESUME_BOUNDARY_VERIFIED.json')]
        identity = ident['identity']; start = ORIGINS[group]; end = start+500
        require(HEX.fullmatch(identity) and json_hash({k:v for k,v in ident.items() if k != 'identity'}) == identity,
                'Run identity self hash mismatch')
        require(ident['scope'] == 'profile' and ident['dataset'] == 'navsim' and
                ident['training_source_sha'] == TRAINING_SOURCE and
                ident['world_size'] == 8 and ident['effective_global_batch'] == 32 and
                ident['microbatch'] == 4 and ident['FM_repeat'] == 8,
                'Not the bound be5 full-population native pilot')
        require(ident['schedule_horizon'] == 100000 and ident['schedule_warmup'] == 5000 and
                ident['scenes_per_epoch'] == 101592 and ident.get('engineering_training_selection') is None and
                ident['config']['structured_world']['group'] ==
                ('G1_FULL_UNIFORM' if group == 'G1_mid' else 'G3_EVENT_LOCAL'), 'Wrong registered NAV group/population/schedule')
        require(ident['execution_mode'] == ('engineering_fast' if role == 'B' else 'io_preserving_v1') and
                ident['benchmark_no_end_checkpoint'] is False, 'Wrong role/mode/save contract')
        require(ex['returncode'] == 0 and ex.get('monitor_error') is None and
                ex['pid'] == proc['pid'] and math.isfinite(ex['wall_seconds']) and ex['wall_seconds'] > 0,
                'Actual successful PROCESS_EXIT required')
        require(done['identity'] == identity and done['updates'] == end,
                'Complete continuous 500-update endpoint required')
        ancestry = ident['engineering_continue_origin']
        require(ancestry['completed'] == start and ancestry['tag'] == f'milestone_{start:07d}' and
                HEX.fullmatch(ancestry['identity']) and HEX.fullmatch(ancestry['complete_sha256']), 'Wrong native origin')
        require(boundary['identity'] == identity and boundary['passed'] is True and
                boundary['completed'] == start and boundary['all_task_RNG'] == 'exact all ranks', 'Missing exact origin boundary')
        plan = ident['engineering_native_capture']
        require(plan['origin_completed'] == start and plan['end_update'] == end and
                plan['relative_updates'] in [list(POINTS[:-1]), list(POINTS)] and
                plan['absolute_updates'] == [start+x for x in plan['relative_updates']] and
                plan['stop_at_capture'] is False, 'Wrong continuous five-capture plan')
        metrics = [json.loads(x) for x in read_text(run/'metrics.jsonl').splitlines() if x.strip()]
        require([x['updates'] for x in metrics] == list(range(start+1, end+1)) and
                metrics[-1]['scene_exposure'] == done['exposure'], 'Actual 500 update metrics required')
        previous = boundary['exposure']
        for m in metrics:
            require(0 < m['scene_exposure']-previous <= 32, 'Invalid actual scene exposure')
            previous = m['scene_exposure']
        tags = [f'milestone_{start+x:07d}' for x in POINTS]
        require(read_text(run/'checkpoints/latest').strip() == tags[-1], 'Latest must be the complete500 endpoint')
        require({p.name for p in (run/'checkpoints').iterdir() if p.is_dir()} == set(tags), 'Exactly five retained banks required')
        role_banks = []
        for point, tag in zip(POINTS, tags):
            folder = run/'checkpoints'/tag
            complete = read(folder/'COMPLETE.json')
            require(complete['identity'] == identity and complete['completed'] == start+point and
                    complete['tag'] == tag and complete['ranks'] == 8 and
                    complete['status'] == ('COMPLETE' if point == 500 else 'RUNNING'), 'Bank step/tag/identity/status mismatch')
            require(set(complete['file_sha256']) == PAYLOADS and
                    all(isinstance(h,str) and HEX.fullmatch(h) for h in complete['file_sha256'].values()),
                    'Every bank must bind the exact17 native payloads')
            require({p.name for p in folder.iterdir()} == PAYLOADS | {'COMPLETE.json'}, 'Unexpected/missing bank entry')
            payload_paths.update(str((folder/n).relative_to(root)) for n in PAYLOADS)
            role_banks.append({'tag':tag, 'completed':start+point, 'identity':identity,
                               'file_sha256':complete['file_sha256']})
        identities[role] = ident
        banks[role] = role_banks
    require(all(identities[r]['engineering_continue_origin'] == identities['A']['engineering_continue_origin']
                for r in ROLES), 'Triplet origins differ')
    require(all(identities[r]['config'] == identities['A']['config'] for r in ROLES), 'Triplet configurations differ')
    files = {}
    for directory, dirs, names in os.walk(root, followlinks=False):
        for name in dirs:
            require(not (Path(directory)/name).is_symlink(), 'Directory symlink forbidden')
        for name in names:
            p = Path(directory)/name
            no_links(p)
            if (p.parent == root and name in ('ARCHIVE_VERIFIED.json', 'RELEASE_RECEIPT.json')) or (
                    p.parent in [root/r for r in ROLES] and name in ('SOURCE_COPY_RECEIPT.json',)):
                continue
            require(p.relative_to(root).parts[0] in ROLES, 'Unexpected group entry')
            files[str(p.relative_to(root))] = file_hash(p, durable=durable, stamps=stamps)
        if durable:
            sync_dir(directory)
    require(len(payload_paths) == 255, 'Real triplet must contain15 complete banks/255 payloads')
    for role in ROLES:
        for bank in banks[role]:
            for name, h in bank['file_sha256'].items():
                item = files[f'{role}/checkpoints/{bank["tag"]}/{name}']
                require(item['bytes'] > 0 and item['sha256'] == h, 'Actual native payload SHA256 mismatch: '+name)
    return {'schema':'native_recollected_copy_inventory_v1', 'group':group,
            'run_identities':{r:x['identity'] for r,x in identities.items()},
            'banks':banks, 'files':files, 'payload_paths':sorted(payload_paths),
            'total_bytes':sum(x['bytes'] for x in files.values()), 'banks_count':15,
            'payload_count':255, 'tensor_numerical_comparison':False}


def rename_new(source, destination):
    """Linux atomic no-replace rename; never clobber even an empty collision."""
    no_links(source); no_links(destination)
    libc = ctypes.CDLL(None, use_errno=True)
    fn = getattr(libc, 'renameat2', None)
    require(fn is not None, 'Atomic no-replace rename unavailable')
    fn.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    fn.restype = ctypes.c_int
    if fn(-100, os.fsencode(source), -100, os.fsencode(destination), 1):
        e = ctypes.get_errno()
        raise OSError(e, os.strerror(e), str(destination))
    sync_dir(Path(destination).parent)


def release_payloads(root, group, ticket):
    """Recheck all hashes before the first unlink; keep original COMPLETE archives."""
    require(ticket['destination_verified'] is True and ticket['group'] == group and
            HEX.fullmatch(ticket['archive_receipt_sha256']), 'Verified destination ticket required')
    archive_path(ticket['archive_root'], group)
    require(all(read(root/r/'SOURCE_COPY_RECEIPT.json') == ticket for r in ROLES), 'Unbound source receipt')
    stamps = {}
    current = inspect_group(root, group, stamps=stamps)
    require(current == ticket['inventory'], 'Source changed after verified copy; nothing released')
    marker = {'schema':'native_probe_relocated_v1', 'status':'RELEASE_STARTED', 'created_UTC':utc(),
              'group':group, 'archive':ticket['archive'], 'archive_receipt_sha256':ticket['archive_receipt_sha256'],
              'inventory_sha256':json_hash(current), 'restorable_here':False,
              'original_COMPLETE_files':'COMPLETE.original.json; historical capture only, not a recovery completion'}
    for role in ROLES:
        write_new(root/role/'RELOCATED.json', marker)
        for bank in current['banks'][role]:
            folder = root/role/'checkpoints'/bank['tag']
            write_new(folder/'RELOCATED.json', marker)
            rename_new(folder/'COMPLETE.json', folder/'COMPLETE.original.json')
    deleted = []
    try:
        for relative in current['payload_paths']:
            folder, name = Path(relative).parent, Path(relative).name
            require(name in PAYLOADS, 'Unlink inventory escaped fixed native names')
            parent = no_links(root/folder)
            fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                s = os.stat(name, dir_fd=fd, follow_symlinks=False)
                require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1, 'Payload type changed before unlink')
                require((s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns) == stamps[str(root/relative)],
                        'Payload changed after final hash; refusing unlink')
                os.unlink(name, dir_fd=fd)
                os.fsync(fd)
                deleted.append(relative)
            finally:
                os.close(fd)
    except BaseException as error:
        for role in ROLES:
            write_new(root/role/'RELOCATION_FAILED.json', {**marker, 'status':'FAILED_DURING_RELEASE',
                'error':repr(error), 'deleted_payloads':deleted, 'automatic_retry':False})
        raise
    receipt = {**marker, 'status':'RELEASED', 'completed_UTC':utc(),
               'deleted_payload_count':len(deleted), 'released_bytes':sum(current['files'][p]['bytes'] for p in deleted),
               'run_metadata_and_original_COMPLETE_preserved':True}
    for role in ROLES:
        write_new(root/role/'RELEASE_RECEIPT.json', receipt)
    return receipt


def record_source_receipt(root, group, ticket):
    require(inspect_group(root, group) == ticket['inventory'], 'Source changed during copy')
    result = {}
    for role in ROLES:
        path = root/role/'SOURCE_COPY_RECEIPT.json'
        write_new(path, ticket)
        result[role] = file_hash(path)['sha256']
    return {'source_receipts_sha256':result}


def remote_action(action, group, ticket=None):
    require(socket.gethostname() in (SOURCE_HOST, SOURCE_HOST+'-worker-0'), 'Source host identity mismatch')
    root = source_path(group)
    if action == 'inspect':
        return inspect_group(root, group)
    require(action in ('receipt', 'release'), 'Unknown source action')
    require(ticket['source_host'] == SOURCE_HOST and ticket['source'] == str(root), 'Ticket source route mismatch')
    if action == 'receipt':
        return record_source_receipt(root, group, ticket)
    return release_payloads(root, group, ticket)


class SSHSource:
    def __init__(self):
        self.code = Path(__file__).read_text()
    def call(self, action, group, ticket=None):
        bootstrap = ('import sys,json,hashlib; x=json.load(sys.stdin); '
            'assert hashlib.sha256(x["code"].encode()).hexdigest()==x["code_sha256"]; '
            'n={"__name__":"bound_relocation_helper"}; exec(compile(x["code"],"<bound relocation helper>","exec"),n); '
            'print(json.dumps(n["remote_action"](x["action"],x["group"],x["ticket"])))')
        request = {'code':self.code,'code_sha256':hashlib.sha256(self.code.encode()).hexdigest(),
                   'action':action,'group':group,'ticket':ticket}
        p = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',
            '-o','ServerAliveInterval=30','-o','ServerAliveCountMax=4',SOURCE_HOST,
            shlex.join(['python3','-c',bootstrap])], input=json.dumps(request), text=True, capture_output=True)
        require(p.returncode == 0, 'Source '+action+' failed; no retry: '+p.stderr[-1000:])
        return json.loads(p.stdout)
    def copy(self, group, stage, inventory):
        p = subprocess.run(['rsync','-rlt','--protect-args','--from0','--files-from=-',
            '--chmod=Du=rwx,Dgo=,Fu=rw,Fgo=','--',SOURCE_HOST+':'+str(source_path(group))+'/',str(stage)+'/' ],
            input=b''.join(os.fsencode(n)+b'\0' for n in sorted(inventory['files'])), capture_output=True)
        require(p.returncode == 0, 'rsync failed; partial retained, no retry: '+p.stderr.decode(errors='replace')[-1000:])


def relocate(group, archive_root=DEFAULT_ARCHIVE, *, release=False, release_only=False, backend=None, helper_binding=None):
    require(socket.gethostname() == PARENT_HOST, 'Run only on the registered parent archive host')
    final = archive_path(archive_root, group); root = final.parent
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    require(root.stat().st_uid == os.getuid() and root.stat().st_mode & 0o077 == 0, 'Private owned archive root required')
    backend = backend or SSHSource()
    lock = root/('.'+group+'.lock')
    fd = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    stage = None
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if release_only:
            require(release, 'Release-only requires explicit --release-source-payloads')
            receipt = read(final/'ARCHIVE_VERIFIED.json')
            inv = inspect_group(final, group, durable=True)
            require(inv == receipt['inventory'], 'Existing archive hash/metadata mismatch; no release')
            ticket = receipt['source_ticket']
            ticket['archive_receipt_sha256'] = file_hash(final/'ARCHIVE_VERIFIED.json')['sha256']
        else:
            require(not final.exists() and not final.is_symlink(), 'Archive collision; no overwrite/delete')
            require(not list(root.glob('.writing_'+group+'_*')) and not (root/('.'+group+'_FAILED.json')).exists(),
                    'Prior partial/failed attempt retained; no automatic retry')
            print(json.dumps({'stage':'SOURCE_FULL_HASH','group':group}), flush=True)
            inv = backend.call('inspect', group)
            require(inv['group'] == group and inv['banks_count'] == 15 and inv['payload_count'] == 255, 'Incomplete real triplet')
            require(inv['total_bytes'] <= GROUP_UPPER_BYTES and
                    owned_archive_bytes(root)+inv['total_bytes'] <= ARCHIVE_UPPER_BYTES,
                    'Registered owned archive pool/run cap exceeded')
            free_before = shutil.disk_usage(root).free
            require(free_before >= inv['total_bytes']+RESERVE_BYTES, 'Archive physical free reserve145GiB insufficient; quota not assumed')
            stage = root/('.writing_'+group+'_'+uuid.uuid4().hex)
            stage.mkdir(mode=0o700)
            print(json.dumps({'stage':'COPY_ONLY_NEW_WRITING','group':group,'bytes':inv['total_bytes']}), flush=True)
            backend.copy(group, stage, inv)
            copied = inspect_group(stage, group, durable=True)
            require(copied == inv, 'Destination hashes/metadata differ; partial retained')
            ticket = {'schema':'native_probe_source_copy_receipt_v1','group':group,'source_host':SOURCE_HOST,
                'source':str(source_path(group)),'archive_root':str(root),'archive':str(final),
                'destination_host':socket.gethostname(),'destination_verified':True,
                'created_UTC':utc(),'inventory':inv,'relocation_helper_binding':helper_binding}
            receipt = {'schema':'native_probe_archive_verified_v1','destination_full_hash_verified':True,
                'inventory':inv,'source_ticket':ticket,'payload_fsync_complete':True,
                'no_numerical_acceptance':True,'physical_free_bytes_before_copy':free_before,
                'reserve_bytes':RESERVE_BYTES,'quota_status':'not independently exposed; physical free is not exclusive/quota proof'}
            receipt['owned_storage_plan'] = {'group_upper_bytes':GROUP_UPPER_BYTES,
                'archive_upper_bytes':ARCHIVE_UPPER_BYTES,'kind':'bounded internal owned archive plan, not exclusive filesystem reservation'}
            write_new(stage/'ARCHIVE_VERIFIED.json', receipt)
            rename_new(stage, final)
            stage = None
            ticket = {**ticket,'archive_receipt_sha256':file_hash(final/'ARCHIVE_VERIFIED.json')['sha256']}
            backend.call('receipt', group, ticket)
        if release:
            print(json.dumps({'stage':'EXPLICIT_SOURCE_PAYLOAD_RELEASE','group':group}), flush=True)
            result = backend.call('release', group, ticket)
            write_new(final/'RELEASE_RECEIPT.json', result)
        return {'status':'RELEASED' if release else 'COPY_ONLY','group':group,'archive':str(final),
                'banks':15,'payload_files':255,'source_restorable_here':not release,
                'archive_receipt_sha256':file_hash(final/'ARCHIVE_VERIFIED.json')['sha256'],
                'numerical_acceptance':False}
    except BaseException as error:
        failure = {'status':'FAILED','group':group,'error':repr(error),'partial':str(stage) if stage else None,
                   'archive':str(final),'explicit_release_requested':release,
                   'possible_partial_release':release,'automatic_retry':False,'created_UTC':utc()}
        if stage is not None and stage.exists():
            write_new(stage/'RELOCATION_FAILED.json', failure)
        failure_path = root/('.'+group+'_FAILED.json')
        if not failure_path.exists():
            write_new(failure_path, failure)
        raise
    finally:
        os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=tuple(ORIGINS), required=True)
    parser.add_argument('--archive-root', type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument('--execute', action='store_true', help='Otherwise print bounded routes only; no SSH/copy/release')
    parser.add_argument('--release-source-payloads', action='store_true', help='Explicitly release only verified new zt4 payload copies')
    parser.add_argument('--release-only', action='store_true', help='Use an existing verified copy, without recopy/retry')
    args = parser.parse_args()
    if not args.execute:
        require(not args.release_source_payloads and not args.release_only, 'Releasing requires --execute')
        print(json.dumps({'status':'PLAN_ONLY','source_host':SOURCE_HOST,'source':str(source_path(args.group)),
                          'archive':str(archive_path(args.archive_root,args.group)),'default':'copy_only'})); return 0
    # Parent commits/freezes this helper before any real data operation. The
    # executable file itself must equal its committed blob, independent of the
    # be5 model/trainer source being archived.
    script = Path(__file__).resolve()
    checkout = script.parents[2]
    frozen = subprocess.check_output(['git','show','HEAD:'+str(script.relative_to(checkout))],cwd=checkout)
    require(frozen == script.read_bytes(), 'Relocation helper must be cold committed/frozen before --execute')
    binding = {'helper_sha256':hashlib.sha256(frozen).hexdigest(),
               'helper_source_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip()}
    result = relocate(args.group,args.archive_root,release=args.release_source_payloads,release_only=args.release_only,
                      helper_binding=binding)
    print(json.dumps(result)); return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        print(json.dumps({'status':'FAILED','error':str(error),'automatic_retry':False}),file=sys.stderr)
        sys.exit(2)

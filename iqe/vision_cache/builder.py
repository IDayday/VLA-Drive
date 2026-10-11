"""Independent frozen-front builder; no optimizer, loss, backward or LLM calls."""
from pathlib import Path
import signal
import time
import torch
from ..contracts import require
from ..io import atomic_json, digest, file_hash, read_json, lock
from ..performance.vision_cache import FrozenVisionPayload
from ..training.execution import FrozenVisionAhead
from .identity import payload_key, tensor_state_hash, visual_identity
from .inputs import current_identity, current_numeric_fields
from .plan import capacity_check, load_plan
from .storage import ShardWriter, commit_cache, recover_shard, shard_lease, validate_payload


def split_payloads(value, grid, image_token_id):
    ids, mask, positions, slots, visual, deep = value
    require(grid.shape == (len(ids)*3, 3), 'three-view visual grid packing')
    offset = 0; outputs = []
    for i in range(len(ids)):
        length = int(mask[i].sum()); start = ids.shape[1]-length
        require(bool((mask[i, :start] == 0).all()) and bool((mask[i, start:] == 1).all()), 'only left padding supported')
        count = int(ids[i].eq(image_token_id).sum())
        cpu = lambda t: t.detach().cpu().clone()
        p = FrozenVisionPayload(cpu(ids[i, start:]), cpu(positions[:, i, start:]),
            {n: cpu(v[i]-start) for n,v in slots.items()}, cpu(visual[offset:offset+count]),
            tuple(cpu(v[offset:offset+count]) for v in deep), cpu(grid[i*3:i*3+3]))
        outputs.append(validate_payload(p)); offset += count
    require(offset == len(visual), 'visual placeholder count mismatch')
    return outputs


class FrozenFront:
    def __init__(self, model):
        self.model = model; self.visual = model.qwen_vl_interface.model.model.visual
        require(not self.visual.training and all(not p.requires_grad for p in self.visual.parameters()), 'frozen eval front required')
        self.grid = None; self.calls = 0
        def capture(module, args, kwargs):
            grid = kwargs.get('grid_thw', args[1] if len(args)>1 else None)
            require(grid is not None, 'grid capture missing')
            self.grid = grid.detach().cpu().clone(); self.calls += 1
        self.hook = self.visual.register_forward_pre_hook(capture, with_kwargs=True)
        language = model.qwen_vl_interface.model.model.language_model
        def forbidden(*args): raise RuntimeError('cache builder must never execute trainable LLM')
        self.language_guard = language.register_forward_pre_hook(forbidden)

    def __call__(self, observations):
        from starVLA.model.framework.DDPForesight import ROBOT_HISTORY_TOKEN
        current = [{k: o[k] for k in ('image', 'lang', 'state')} for o in observations]
        instructions = [o['lang']+' '+ROBOT_HISTORY_TOKEN+''.join(self.model.foresight_tokens)+
                        ''.join(self.model.act_query_tokens) for o in current]
        device = next(self.visual.parameters()).device
        with torch.no_grad(), torch.autocast(device.type, enabled=False):
            value = self.model._build_qwen_batch(current, instructions)
        return split_payloads(value, self.grid, self.model.qwen_vl_interface.model.config.image_token_id)

    def close(self):
        self.hook.remove(); self.language_guard.remove()


def build_access(plan, identity, root):
    """Hash actual image bytes once, never require decoded PIL images on hits."""
    path = Path(root)/'ACCESS_STAGING.json'
    if path.exists():
        value = read_json(path)
        require(value['plan_hash'] == plan['plan_hash'] and value['identity_hash'] == digest(identity), 'stale cache admission staging')
        require(value['content_hash'] == digest(value['access']), 'access staging checksum')
        # JSON canonicalization sorts object keys. Reconstruct the immutable
        # planned order, never derive shard membership from dict iteration.
        return {r['scene_id']:value['access'][r['scene_id']] for r in plan['rows']}
    access = {}
    for row in plan['rows']:
        require(file_hash(row['current_ref']) == row['current_sha256'], 'changed current metadata')
        current = read_json(row['current_ref']); require(current_identity(current) == row['observation_hash'], 'changed observation')
        state, lang = current_numeric_fields(current)
        hashes = [file_hash(p) for p in row['image_paths']]
        key = payload_key(hashes, lang, identity)
        access[row['scene_id']] = {**row, 'key': key, 'image_sha256_ordered': hashes, 'state_hash':digest(state.tolist()),
            'source_images': [{'bytes':Path(p).stat().st_size, 'mtime_ns':Path(p).stat().st_mtime_ns} for p in row['image_paths']]}
    atomic_json(path, {'plan_hash':plan['plan_hash'], 'identity_hash':digest(identity), 'access':access,
                      'content_hash':digest(access)}, immutable=True)
    return access


def projected_shard_bytes(written, completed, record_count, records_per_shard):
    require(0 < record_count <= records_per_shard, 'invalid shard record count')
    return max(written/max(1, completed), 2*1024**3) * record_count/records_per_shard


def build_cache(model, contract, plan_path, root, *, worker=0, workers=1, max_shards=1,
                max_gib=4., reserve_gib=500., rate_mib_s=256., qualification=None, hold_file=None,reuse_roots=()):
    """Each worker leases independent stable shards. No rank is in a payload key."""
    require(0 <= worker < workers <= 8 and max_shards > 0 and max_gib > 0 and rate_mib_s > 0, 'explicit resource/write budget required')
    plan = load_plan(plan_path); root = Path(root); root.mkdir(parents=True, exist_ok=True)
    identity = visual_identity(model, contract)
    require(identity['visual_state_sha256'] == plan['visual_state_sha256'], 'visual changed since locked parent')
    from .reuse import inspect_reuse,link_reuse
    sources=inspect_reuse(reuse_roots,identity,plan['rows'])
    reused_projection=sum(s['index']['shards'][name]['bytes'] for s in sources for name in s['index']['shards'])
    if plan['scope'] == 'representative':
        from .qualification import validate_pilot
        require(qualification, 'representative build requires real FP32 pilot equivalence')
        validate_pilot(qualification,identity,plan['parent_sha256'])
    if plan['scope'] == 'full':
        require(qualification, 'full build needs real pilot numerical and representative IO certificate')
        from .qualification import validate_cache_qualification
        qualified = validate_cache_qualification(qualification, identity=identity, fit_hash=plan['fit_sha256'], for_build=True)
        measured = qualified['representative_record_max_bytes']
        require(type(measured) is int and measured > 0, 'measured record capacity evidence required')
        retention = plan['checkpoint_retention_budget']
        # Original Base continues writing; preserve its entire declared remaining
        # checkpoint schedule plus the parallel22k context budget and safety.
        protected = retention['parent_bytes']*retention['remaining_count'] + (180+200)*1024**3
        reserve_gib = max(reserve_gib, protected/1024**3)
        complete_bytes = sum(p.stat().st_size for p in root.glob('shard_*.bin') if Path(str(p)+'.COMPLETE.json').exists())
        full_capacity = capacity_check(root, max(0,int(measured*plan['requested_scenes']*1.05)-complete_bytes-reused_projection),
            reserve_bytes=int(reserve_gib*1024**3), temporary_bytes=workers*2*1024**3)
        atomic_json(root/'FULL_CAPACITY_PLAN.json',full_capacity)
        require(full_capacity['sufficient'], 'BLOCKED_SHARED_CAPACITY: complete locked population cannot fit; no partial full-build launch')
    atomic_json(root/'IDENTITY.json', identity, immutable=True)
    capacity = capacity_check(root, 0, reserve_bytes=int(reserve_gib*1024**3), temporary_bytes=2*1024**3)
    require(capacity['sufficient'], 'BLOCKED_SHARED_CAPACITY: protected reserve and atomic shard space unavailable')
    with lock(root/'ACCESS_STAGING_BUILD.lock'):
        access = build_access(plan, identity, root)
    with lock(root/'REUSE_BUILD.lock'):
        reused_keys,reused_ids,reused_bytes=link_reuse(root,sources,identity,access)
    # Dedup is by content+prompt+numeric identity, while access keeps each role/source.
    unique = {}; [unique.setdefault(row['key'], row) for row in access.values() if row['key'] not in reused_keys]
    items = list(unique.items()); size = plan['records_per_shard']
    groups = [items[i:i+size] for i in range(0, len(items), size)]
    ids = []
    for i, group in enumerate(groups):
        keys = [key for key,_ in group]
        legacy = f'shard_{i:06d}'
        receipt_path = root/f'{legacy}.bin.COMPLETE.json'
        if receipt_path.exists() and set(read_json(receipt_path)['records']) == set(keys):
            ids.append(legacy)
        else:
            # Content-bound membership also protects recovery across builder
            # assignments. Conflicting historical shards remain untouched.
            ids.append(f'{legacy}_{digest(keys)[:16]}')
    if (root/'COMPLETE.json').exists():
        receipt = read_json(root/'COMPLETE.json'); require(receipt['plan_hash'] == plan['plan_hash'], 'cache root already bound to another plan')
        return {**receipt, 'reused':True}
    stop = [False]; previous = signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    front = FrozenFront(model); began = time.monotonic(); written = 0; completed = 0
    torch_rng = torch.get_rng_state().clone(); cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    try:
        from starVLA.dataloader.foresight_dataset import current_observation
        # Dispatch, when installed by another CPU test, retains this exact online path.
        original_observation = getattr(current_observation, '_online_original', current_observation)
        for index in range(worker, len(groups), workers):
            if hold_file:
                hold=read_json(hold_file)
                if hold.get('mode') not in {'conditional','CONDITIONAL','RELEASED','released','CONDITIONAL_CAPACITY','CONDITIONAL_UNTIL_CAPACITY_SUFFICIENT'}:
                    stop[0]=True
            control=read_json(root/'CONTROL.json') if (root/'CONTROL.json').exists() else {}
            if control.get('pause',False):stop[0]=True
            effective_rate=min(rate_mib_s,float(control.get('rate_mib_s_per_worker',rate_mib_s)))
            require(effective_rate>0,'invalid builder IO throttle')
            if stop[0] or completed >= max_shards: break
            sid = ids[index]
            with shard_lease(root, sid):
                old = recover_shard(root, sid, digest(identity))
                if old is not None: continue
                projected = projected_shard_bytes(written, completed, len(groups[index]), size)
                require(written+projected <= max_gib*1024**3, 'per-invocation write budget reached; resume with an explicit budget')
                space = capacity_check(root, 0, reserve_bytes=int(reserve_gib*1024**3), temporary_bytes=int(projected))
                require(space['sufficient'], 'BLOCKED_SHARED_CAPACITY before new shard')
                writer = ShardWriter(root, sid, digest(identity))
                try:
                    group = groups[index]
                    for at in range(0, len(group), 4):
                        entries = group[at:at+4]; observations = []
                        for key, row in entries:
                            require(file_hash(row['current_ref']) == row['current_sha256'], 'source metadata changed after planning')
                            for path, expected in zip(row['image_paths'], row['image_sha256_ordered']):
                                require(file_hash(path) == expected, 'source image content changed')
                            observations.append(original_observation(read_json(row['current_ref'])))
                        for (key, row), payload in zip(entries, front(observations)): writer.add(key, payload)
                    receipt = writer.commit(); written += receipt['bytes']; completed += 1
                except BaseException:
                    writer.abort(); raise
            elapsed = time.monotonic()-began
            atomic_json(root/f'PROGRESS_{worker}.json', {'status':'BUILDING', 'worker':worker,
                'completed_shards_this_attempt':completed, 'bytes_written_this_attempt':written, 'last_shard':sid,
                'seconds':elapsed, 'visual_calls':front.calls, 'updated_unix':time.time()})
            delay = written/(effective_rate*1024**2)-elapsed
            if delay > 0:
                # Signal-responsive throttle, never a new training task.
                deadline = time.monotonic()+delay
                while not stop[0] and time.monotonic()<deadline: time.sleep(min(.25, deadline-time.monotonic()))
        require(torch.equal(torch_rng, torch.get_rng_state()), 'frozen builder consumed training CPU RNG')
        if cuda_rng: require(all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all())), 'frozen builder consumed CUDA RNG')
        require(tensor_state_hash(front.visual.state_dict()) == identity['visual_state_sha256'], 'frozen visual buffers/weights mutated')
    finally:
        front.close(); signal.signal(signal.SIGTERM, previous)
    all_done = all((root/f'{sid}.bin.COMPLETE.json').exists() for sid in ids+reused_ids)
    if all_done:
        # Global receipt requires every requested observation; no subset COMPLETE.
        with lock(root/'GLOBAL_COMMIT.lock'):
            return commit_cache(root, identity, access, ids+reused_ids, plan_hash=plan['plan_hash'])
    return {'status':'OPERATOR_PAUSED' if stop[0] else 'PARTIAL', 'bytes_written':written,
            'completed_shards_this_attempt':completed, 'total_shards':len(ids),
            'completed_shards':sum((root/f'{sid}.bin.COMPLETE.json').exists() for sid in ids)}

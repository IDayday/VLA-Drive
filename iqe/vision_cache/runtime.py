"""Narrow qualified FP32 production runtime on the existing performance runner."""
from dataclasses import asdict, dataclass
from pathlib import Path
import fcntl
import os
import torch
import torch.distributed as dist
import yaml
from ..contracts import require
from ..io import atomic_json, digest, file_hash, read_json
from ..training.execution import FrozenVisionAhead, collective_check
from ..training.resume import resolve_resume
from .identity import visual_identity
from .inputs import CacheOnlyDataset, CachedVisionAhead
from .storage import CacheReader


@dataclass(frozen=True)
class ProductionOptions:
    schema_version: int = 1
    training_visual_source: str = 'cached_fp32'
    inference_visual_source: str = 'online_fp32'
    visual_numeric_contract: str = 'legacy_fp32'
    cache_root: str | None = None
    max_open_shards: int = 8
    no_image_decode_on_hit: bool = True
    lineage_root: str = '/mnt/project/iqe-fp32-cache-production-runtime-3931fed/lineages'
    lineage_id: str = 'original_query_base_continuation'
    comparison_id: str | None = None
    auto_stop_old: bool = False
    auto_from_scratch: bool = False

    @classmethod
    def load(cls, path):
        with open(path) as f: value = cls(**yaml.safe_load(f))
        require(value.schema_version == 1 and value.training_visual_source in {'cached_fp32','online_fp32'}, 'production source/schema')
        require(value.inference_visual_source == 'online_fp32' and value.visual_numeric_contract == 'legacy_fp32', 'FP32 deployment path is immutable')
        require(value.no_image_decode_on_hit is True and value.auto_stop_old is False and value.auto_from_scratch is False,
                'production may not decode cache hits, stop old training or restart from scratch')
        require(1 <= value.max_open_shards <= 128 and value.lineage_id and '/' not in value.lineage_id, 'invalid lineage/reader bounds')
        require((value.training_visual_source == 'cached_fp32') == (value.cache_root is not None), 'cache path/source mismatch')
        return value


class ProductionRuntime:
    def __init__(self, config, *, resume_from_run=False, allow_fallback=False):
        self.options = ProductionOptions.load(config)
        self.resume_from_run, self.allow_fallback = resume_from_run, allow_fallback
        self.reader = None; self.lineage_lock = None
        self.gradient_audit = None

    def identity(self):
        values = asdict(self.options)
        # Locks/experiment names locate runs; they do not change the numeric path.
        for key in ('lineage_root','lineage_id','comparison_id'): values.pop(key)
        if self.options.cache_root:
            values['cache_index_sha256'] = read_json(Path(self.options.cache_root)/'COMPLETE.json')['index_sha256']
        values['source_hashes'] = {p.name:file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
        return values

    def validate_options(self, opts, args):
        require(opts.optimizer == 'fused' and opts.gradient_as_bucket_view and opts.bucket_cap_mb == 50 and
                opts.checkpoint_policy == 'none' and opts.visual_precision == 'legacy_fp32' and
                opts.execution_weights == 'fp32' and opts.query_training_semantics == 'legacy_repeat8' and
                opts.vision_cache is None and opts.deterministic, 'production requires the qualified combined_fp32 execution recipe')
        require(not args.build_vision_cache and not args.mask_probe, 'production builder and training are separate')
        require(args.mode != 'bounded' or self.options.comparison_id, 'bounded experiment needs explicit comparison identity')
        require(args.mode != 'continuation' or (args.qualification and self.options.comparison_id is None), 'formal continuation must bind qualification and stable lineage')
        self.runtime_options = opts.identity(); self.mode = args.mode

    def resolve_checkpoint(self, args, rank, device):
        path = args.checkpoint; error = None
        if self.resume_from_run and rank == 0:
            try: path = resolve_resume(args.output, 'auto', allow_fallback=self.allow_fallback)
            except Exception as e: error = repr(e)
        collective_check(error, device=device)
        values = [path]; dist.broadcast_object_list(values, src=0); args.checkpoint = values[0]

    def lock_lineage(self, args):
        root = Path(self.options.lineage_root); root.mkdir(parents=True, exist_ok=True)
        key = digest({'lineage':self.options.lineage_id, 'comparison':self.options.comparison_id})
        handle = (root/f'{key}.lock').open('a+b'); fcntl.flock(handle, fcntl.LOCK_EX|fcntl.LOCK_NB)
        record = root/f'{key}.json'; output = str(Path(args.output).resolve())
        if record.exists():
            previous = read_json(record)
            require(previous['output'] == output, 'duplicate lineage: resume existing output; timestamps cannot fork a production run')
        else:
            atomic_json(record, {'lineage':self.options.lineage_id,'comparison':self.options.comparison_id,
                'output':output,'parent':str(Path(args.checkpoint).resolve()),'parent_sha256':read_json(args.checkpoint+'.COMPLETE.json')['checksum']}, immutable=True)
        self.lineage_lock = handle

    def validate_state(self, state, identity, args):
        require(state.get('kind') == 'iqe_query_base' and state['optimizer_step'] > 0, 'complete trained Base state required')
        require(not state.get('terminal', False), 'terminal Base: no additional optimizer updates permitted')
        if state.get('performance_identity'):
            require(state['performance_identity'] == identity, 'exact resume runtime changed')
        if args.mode == 'continuation':
            from .qualification import validate_cache_qualification
            validate_cache_qualification(args.qualification, fit_hash=identity['fit_hash'], runtime_identity=identity, for_build=False)
            require(state.get('learned_driving_weights_loaded') is False, 'initialization provenance missing/foreign; never default missing to False')
            args.save = True

    def attach_dataset(self, dataset, model, contract, fit):
        if self.options.training_visual_source == 'online_fp32': return dataset
        identity = visual_identity(model, contract)
        self.reader = CacheReader(self.options.cache_root, identity=identity,
                                  max_open_shards=self.options.max_open_shards)
        access = self.reader.index['access']
        require(set(access) <= set(fit), 'cache access list includes data outside parent fit population')
        require(all(r['role'] == 'incremental_fit' for r in access.values()), 'training cache role leak')
        if self.mode == 'continuation': require(set(access) == set(fit), 'production continuation requires full locked fit coverage')
        return CacheOnlyDataset(dataset, self.reader)

    def vision(self, model):
        if not self.reader: return FrozenVisionAhead(model)
        vision=CachedVisionAhead(model,self.reader)
        from PIL import Image
        from .inputs import _CURRENT_READER
        original_open=Image.open;original_close=vision.close
        def guarded_open(*args,**kwargs):
            require(_CURRENT_READER.get() is None,'Image.open forbidden on a cache hit')
            return original_open(*args,**kwargs)
        Image.open=guarded_open
        def guard_visual(*args):
            if vision.active is not None:
                vision.calls['visual_calls_on_hit']+=1
                raise RuntimeError('visual.forward forbidden on a cache hit')
        handle=vision.visual.register_forward_pre_hook(guard_visual)
        def close():
            try:original_close()
            finally:Image.open=original_open;handle.remove()
        vision.close=close
        return vision

    def checkpoint_metadata(self, state, parent, checksum):
        return {'training_visual_source':self.options.training_visual_source, 'inference_visual_source':'online_fp32',
                'visual_numeric_contract':'legacy_fp32', 'cache_execution_dependency':self.identity(),
                'base_provenance':state.get('base_provenance', {'parent_checkpoint':parent,'parent_sha256':checksum,
                    'parent_step':state['optimizer_step'],'learned_driving_weights_loaded':state.get('learned_driving_weights_loaded'),
                    'unknown_provenance_is_not_false':True}),
                'execution_migration':{'parent':parent,'parent_sha256':checksum,'optimizer_schedule_preserved':True,
                    'cross_historical_backend_bitwise_claim':False}}

    def report(self, vision):
        import resource
        return {'options':asdict(self.options), 'cache_reader':dict(self.reader.stats) if self.reader else None,
                'calls':getattr(vision,'calls',None),'gradient_audit':self.gradient_audit,
                'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'open_fds':len(list(Path('/proc/self/fd').iterdir())), 'read_path':'shared_per_record_pread' if self.reader else 'online_fp32'}

    def capture_gradient(self, model):
        from .identity import tensor_state_hash
        parameters={n:p for n,p in model.named_parameters() if p.requires_grad}
        gradients={n:p.grad for n,p in parameters.items() if p.grad is not None}
        groups={}
        for name,p in parameters.items():
            if 'embed_tokens' in name:group='input_token_embeddings'
            elif 'language_model' in name:group='language'
            elif '.visual.' in name:group='ILLEGAL_trainable_visual'
            elif 'foresight_queries' in name:group='foresight_queries'
            elif 'action_input_model' in name:group='history_embedding'
            elif 'action_model' in name:group='query_'+name.split('action_model.',1)[-1].split('.')[0]
            else:group=name.split('.')[0]
            row=groups.setdefault(group,{'parameters':0,'with_gradient':0,'finite':True,'squared_norm':0.})
            row['parameters']+=p.numel()
            if p.grad is not None:
                row['with_gradient']+=p.numel();row['finite'] &= bool(torch.isfinite(p.grad).all())
                row['squared_norm']+=float(p.grad.detach().float().square().sum())
        self.gradient_audit={'full_named_gradient_sha256':tensor_state_hash(gradients),
            'unused_names':sorted(set(parameters)-set(gradients)), 'groups':groups,
            'scope':'every element of every available named gradient, post identical clipping'}

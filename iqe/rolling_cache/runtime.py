"""Opt-in storage adapter. The existing trainer and numeric graph are unchanged."""
from dataclasses import dataclass
from pathlib import Path
import torch.distributed as dist
import yaml
from ..contracts import require
from ..io import file_hash, digest
from ..vision_cache.runtime import ProductionOptions, ProductionRuntime
from ..vision_cache.inputs import CacheOnlyDataset
from ..vision_cache.identity import visual_identity
from .plan import load_session
from .pool import Pool
from .reader import RollingReader, validate_rolling_certificate


@dataclass(frozen=True)
class RollingOptions:
    root: str
    lineage_root: str
    lineage_id: str
    comparison_id: str | None = None
    max_open_shards: int = 4
    wait_seconds: int = 900


class RollingRuntime(ProductionRuntime):
    def __init__(self, config, *, resume_from_run=False, allow_fallback=False):
        with open(config) as handle: self.rolling = RollingOptions(**yaml.safe_load(handle))
        self.session, self.full = load_session(self.rolling.root)
        self.options = ProductionOptions(cache_root=self.rolling.root,
            lineage_root=self.rolling.lineage_root, lineage_id=self.rolling.lineage_id,
            comparison_id=self.rolling.comparison_id, max_open_shards=self.rolling.max_open_shards)
        self.resume_from_run, self.allow_fallback = resume_from_run, allow_fallback
        self.reader = self.lineage_lock = self.gradient_audit = None

    def identity(self):
        return {'schema': 'iqe_rolling_runtime_v1', 'session_hash': self.session['content_hash'],
            'wait_seconds': self.rolling.wait_seconds, 'max_open_shards': self.rolling.max_open_shards,
            'training_visual_source': 'remote_rolling_fp32', 'inference_visual_source': 'online_fp32',
            'visual_numeric_contract': 'legacy_fp32', 'no_image_decode_on_hit': True,
            'source_hashes': {p.name:file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
            'qualified_core_hashes': {p.name:file_hash(p) for p in sorted(
                (Path(__file__).parent.parent / 'vision_cache').glob('*.py'))}}

    def validate_state(self, state, identity, args):
        require(state['kind'] == 'iqe_query_base' and not state.get('terminal', False), 'live Query state required')
        require(digest(state['fit_scene_ids']) == self.session['fit_sha256'], 'rolling population changed')
        if state.get('performance_identity'):
            require(state['performance_identity'] == identity, 'exact resume runtime changed')
        else:
            require(state['optimizer_step'] == self.session['parent_step'] and
                    state['epoch'] == self.session['epoch'] and state['offset'] == self.session['offset'],
                    'new rolling lineage must start at its bound complete parent')
            from ..io import read_json
            require(read_json(args.checkpoint+'.COMPLETE.json')['checksum'] == self.session['parent_sha256'],
                    'rolling parent checkpoint changed')
        if args.mode == 'continuation':
            validate_rolling_certificate(args.qualification, identity, self.session)
            require(state.get('learned_driving_weights_loaded') is False, 'unknown initialization provenance')
            args.save = True
        self.cursor = {k:state[k] for k in ('optimizer_step','epoch','offset')}

    def attach_dataset(self, dataset, model, contract, fit):
        from tools.ddpolicy_vehicle.training_state import epoch_batches
        from ..training.execution import collective_check
        require(fit == [r['scene_id'] for r in self.full['rows']], 'rolling fit order mismatch')
        pool = None; error = None; device = next(model.parameters()).device
        try: pool = Pool(self.rolling.root, self.session, self.full, epoch_batches)
        except Exception as exc: error = repr(exc)
        collective_check(error, device=device)
        request = [None]; error = None
        if dist.get_rank() == 0:
            try: request[0] = pool.start_attempt(self.cursor['optimizer_step'])
            except Exception as exc: error = repr(exc)
        collective_check(error, device=device)
        dist.broadcast_object_list(request, src=0)
        self.reader = RollingReader(pool, request[0], rank=dist.get_rank(),
            epoch=self.cursor['epoch'], offset=self.cursor['offset'], step=self.cursor['optimizer_step'],
            identity=visual_identity(model, contract), wait_seconds=self.rolling.wait_seconds,
            max_open_shards=self.rolling.max_open_shards)
        return CacheOnlyDataset(dataset, self.reader)

    def vision(self, model):
        value = super().vision(model)
        original_prepare, original_close = value.prepare, value.close
        def prepare(observations, micro_batch):
            result = original_prepare(observations, micro_batch)
            self.reader.prepared([o.scene_id for o in observations])
            return result
        def close():
            try: original_close()
            finally: self.reader.pool.close()
        value.prepare, value.close = prepare, close
        return value

    def checkpoint_metadata(self, state, parent, checksum):
        result = super().checkpoint_metadata(state, parent, checksum)
        result['training_visual_source'] = 'remote_rolling_fp32'
        result['rolling_prefetch_is_not_resume_cursor'] = True
        return result

"""Optional metadata adapters. No trainer dispatch or scope admission here.

The immutable source reader inherits the existing shard/header/record checks.
The provider inherits the existing all-times target assembly, masks, RNG isolation
and bounded tensor LRU. Only population-sized metadata containers are replaced.
"""
from collections import OrderedDict
import threading

from .st_world_history import file_sha256
from .st_world_metadata_index import MetadataIndex
from .st_world_ondemand import BoundedTeacherProvider
from .st_world_schema import TeacherFeatureContract, identity_hash
from .st_world_targets import TeacherCacheReader, validate_target_manifest


class IndexedTeacherCacheReader(TeacherCacheReader):
    def __init__(self, index, *, expected_identity, expected_contract=None, max_verified_shards=16):
        if not isinstance(index, MetadataIndex):
            raise TypeError('Use a verified immutable MetadataIndex')
        if type(max_verified_shards) is not int or max_verified_shards < 1:
            raise ValueError('Reader verification LRU must be bounded and positive')
        index.check()
        d = index.receipt
        if d['cache_identity'] != expected_identity or d['cache_header'] is None:
            raise ValueError('Indexed source requires the original COMPLETE cache identity')
        from pathlib import Path
        self.root = Path(d['cache_root'])
        self.identity = dict(d['cache_header'], records=index.records)
        self.contract = TeacherFeatureContract.from_dict(d['contract'])
        if expected_contract is not None and self.contract.identity != expected_contract.identity:
            raise ValueError('Wrong teacher/transform/precision/grid')
        self.manifest, self.scenes = index.manifest, index.scenes
        self.locations, self.receipt_hashes = index.locations, index.receipt_hashes
        self.verified, self.verified_versions = OrderedDict(), {}
        self.max_verified_shards = max_verified_shards
        self._verification_lock = threading.RLock()
        self.metadata_index = index

    def verify_population(self, requested_manifest):
        _verify_population(self.metadata_index, requested_manifest)

    def __getstate__(self):
        raise TypeError('Indexed teacher metadata is process-owned; construct per rank with loader_workers=0')


def _verify_population(index, requested):
    index.check()
    if isinstance(requested, MetadataIndex):
        requested.check()
        if requested.receipt['request_identity'] != index.receipt['request_identity']:
            raise ValueError('Indexed request population differs')
    else:
        requested = validate_target_manifest(requested)
        if requested['identity'] != index.receipt['request_identity']:
            raise ValueError('Requested teacher population differs')


class IndexedBoundedTeacherProvider(BoundedTeacherProvider):
    """New execution identity; no reuse of an old provider PCA/channel scope.

    The caller still needs a separately admitted fixed8 executor and explicit
    trainer/PCA registration. This class does not grant PROFILE/FORMAL approval.
    Each rank owns its SQLite connection and tensor LRU; no background threads.
    """
    def __init__(self, index, contract, sources=(), *, executor=None,
                 execution_identity=None, max_cached_bytes, max_scene_bytes,
                 max_batch_images, max_manifest_bytes=1 << 20):
        if not isinstance(index, MetadataIndex):
            raise TypeError('Use a verified immutable MetadataIndex')
        index.check()
        if TeacherFeatureContract.from_dict(index.receipt['contract']).identity != contract.identity:
            raise ValueError('Indexed request/teacher contract differs')
        sources = tuple(sources)
        if any(not isinstance(source, IndexedTeacherCacheReader) for source in sources):
            raise ValueError('Bounded metadata path requires indexed COMPLETE sources')
        indices = {id(index): index}
        indices.update({id(s.metadata_index): s.metadata_index for s in sources})
        if sum(i.max_metadata_cache_bytes for i in indices.values()) > 256*(1 << 20):
            raise ValueError('Per-rank combined metadata cache budget exceeds 256 MiB')
        # Reuse every existing source, executor, lineage and budget check with a
        # bounded header. The complete original population was independently
        # streamed/validated by build_index and pinned by MetadataIndex.
        header = dict(index.receipt['manifest_header'], scenes=[])
        header['identity'] = identity_hash({k:v for k,v in header.items() if k != 'identity'})
        super().__init__(header, contract, sources, executor=executor,
            execution_identity=execution_identity, max_cached_bytes=max_cached_bytes,
            max_scene_bytes=max_scene_bytes, max_batch_images=max_batch_images,
            max_manifest_bytes=max_manifest_bytes)
        self.manifest, self.scenes, self.metadata_index = index.manifest, index.scenes, index
        self.identity.update(schema='st_indexed_bounded_teacher_provider_v1',
            request_manifest_identity=index.receipt['request_identity'],
            metadata_index_identity=index.receipt['identity'],
            metadata_adapter_sha256=file_sha256(__file__),
            max_metadata_cache_bytes=index.max_metadata_cache_bytes,
            source_metadata_index_identities=[getattr(s, 'metadata_index', None).receipt['identity']
                if isinstance(s, IndexedTeacherCacheReader) else None for s in self.sources],
            metadata_policy='immutable_index_bounded_rows', training_admission=False)
        self.identity['identity'] = identity_hash({k:v for k,v in self.identity.items() if k != 'identity'})

    def verify_population(self, manifest):
        _verify_population(self.metadata_index, manifest)

    def memory_report(self):
        result = super().memory_report()
        # Include all unique SQLite readers; tensor LRU has its separate budget.
        indices = {id(self.metadata_index): self.metadata_index}
        indices.update({id(s.metadata_index): s.metadata_index for s in self.sources
                        if isinstance(s, IndexedTeacherCacheReader)})
        result['metadata_cache_budget_bytes'] = sum(i.max_metadata_cache_bytes for i in indices.values())
        result['metadata_connections'] = len(indices)
        result['metadata_population_materialized'] = False
        return result

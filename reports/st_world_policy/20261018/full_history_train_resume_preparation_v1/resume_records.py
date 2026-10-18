"""Explicit same-input partial-history resume; original builder remains the writer.

Only already committed, independently pinned metadata records can be reused.
Raw-log hashes and current-record content are still checked by the original
builder on every invocation. Uncovered scenes run the unchanged implementation.
"""
from contextlib import ExitStack, contextmanager
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import stat
from unittest.mock import patch


class CommittedHistoryRecords:
    def __init__(self, root, budget, census, *, bounded, history, schema, record_bytes):
        self.root = Path(root).resolve()
        self.budget, self.bounded = budget, bounded
        self.history, self.schema = history, schema
        self.rows = {}
        self.verified_images = {}
        self.reused = self.new = 0
        self.saved_history_image_reference_bytes = 0
        self.rehashed_image_bytes = 0
        if (census.get('schema') != 'st_partial_history_metadata_census_v1' or
                census.get('status') != 'CENSUS_COMPLETE_NOT_INPUT_QUALIFICATION' or
                census.get('root') != str(self.root) or
                census.get('source_images_logs_models_read') is not False or
                census.get('shards') != len(census.get('rows', []))):
            raise ValueError('Explicit committed-metadata census required')
        expected_names = {row['name'] for row in census['rows']}
        actual_names = {p.name for p in (self.root / 'history_shards').iterdir()}
        if actual_names != expected_names:
            raise ValueError('Shard population changed since resume census')
        with bounded.BudgetedReader(self.root / 'SELECTED_SCENES.json', budget) as stream:
            selected = json.loads(stream.read())
        groups = OrderedDict()
        for row in selected['scenes']:
            groups.setdefault(row['log'], []).append(row['token'])
        expected_prefix = [(token, log) for log, tokens in groups.items() for token in tokens]
        if len(expected_prefix) != census['expected_scenes'] or len({t for t, _ in expected_prefix}) != len(expected_prefix):
            raise ValueError('Original selected scene population changed')
        decoder = json.JSONDecoder()
        for ordinal, item in enumerate(census['rows']):
            name = item['name']
            if '/' in name or not name.startswith(f'{ordinal:06d}_') or not name.endswith('.jsonseq'):
                raise ValueError('Only the original contiguous committed prefix is reusable')
            path = self.root / 'history_shards' / name
            info = path.lstat()
            if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or
                    not 0 < info.st_size == item['bytes'] <= 4 << 20):
                raise ValueError('Unsafe or changed committed shard')
            with bounded.BudgetedReader(path, budget) as stream:
                raw = stream.read((4 << 20) + 1)
            if len(raw) != info.st_size or hashlib.sha256(raw).hexdigest() != item['sha256']:
                raise ValueError('Committed shard checksum mismatch')
            # Original JSON uses ensure_ascii=True, so byte and text offsets agree.
            text = raw.decode('ascii')
            pos, count = 0, 0
            while pos < len(text):
                record, end = decoder.raw_decode(text, pos)
                if text[end:end + 1] != '\n':
                    raise ValueError('Incomplete committed record')
                end += 1
                history.validate_history_record(record)
                payload = raw[pos:end]
                if payload != record_bytes(record):
                    raise ValueError('Committed record serialization differs from original writer')
                token, log = record['token'], record['log']
                if token in self.rows or hashlib.sha256(log.encode()).hexdigest()[:16] != name[7:-8]:
                    raise ValueError('Duplicate scene or wrong shard log')
                if len(self.rows) >= len(expected_prefix) or (token, log) != expected_prefix[len(self.rows)]:
                    raise ValueError('Committed shards are not the original ordered scene prefix')
                self.rows[token] = dict(path=path, offset=pos, length=end - pos,
                    sha256=hashlib.sha256(payload).hexdigest(), log=log)
                count += 1
                pos = end
                budget.check()
            if count != item['scenes'] or not 1 <= count <= 128:
                raise ValueError('Committed record count changed')
        if len(self.rows) != census['scenes']:
            raise ValueError('Committed scene coverage changed')

    @staticmethod
    def _version(info):
        return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
                info.st_ctime_ns, info.st_mode, info.st_nlink)

    def _verify_image(self, ref, roots):
        p = Path(ref['path'])
        if p != p.resolve() or not any(p.is_relative_to(root) for root in roots):
            raise ValueError('Committed image escapes original sensor roots')
        before = p.stat()
        if (not stat.S_ISREG(before.st_mode) or before.st_size != ref['bytes'] or
                before.st_mtime_ns != ref['mtime_ns']):
            raise ValueError('Committed image version changed; no stale reuse/fallback')
        key = str(p)
        old = self.verified_images.get(key)
        if old is not None:
            if old != (ref, self._version(before)):
                raise ValueError('Image changed after this invocation verified its SHA')
            return
        # Old records lack inode/ctime. Never invent their original values or
        # trust size/mtime alone: rehash each distinct image on this invocation.
        digest = hashlib.sha256()
        amount = 0
        with self.bounded.BudgetedReader(p, self.budget) as stream:
            while block := stream.read(1 << 20):
                amount += len(block)
                digest.update(block)
        after = p.stat()
        if (self._version(before) != self._version(after) or amount != ref['bytes'] or
                digest.hexdigest() != ref['sha256']):
            raise ValueError('Committed image SHA/version changed')
        self.verified_images[key] = (dict(ref), self._version(after))
        self.rehashed_image_bytes += amount

    def get(self, token, log, current, raw_sha, *, offsets, tolerance, sensor_roots):
        row = self.rows.get(token)
        if row is None:
            self.new += 1
            return None
        with self.bounded.BudgetedReader(row['path'], self.budget) as stream:
            stream.seek(row['offset'])
            raw = stream.read(row['length'])
        if hashlib.sha256(raw).hexdigest() != row['sha256']:
            raise ValueError('Committed record changed during resume')
        record = self.history.validate_history_record(json.loads(raw))
        if (record['token'] != token or record['log'] != log or row['log'] != log or
                record['source_raw_sha256'] != raw_sha or
                record['current_record_identity'] != self.schema.identity_hash(current) or
                record['current_timestamp_us'] != current['timestamp'] or
                record['offsets_s'] != list(offsets) or
                record['timestamp_tolerance_s'] != tolerance):
            raise ValueError('Committed history differs from actual source/current/causal contract')
        roots = [Path(p).resolve() for p in sensor_roots]
        for frame in record['frames']:
            for view in frame['views']:
                ref = view['image']
                if ref is None:
                    continue
                self._verify_image(ref, roots)
                # Original image_reference charges a full-file header bound and
                # actual full-file SHA. Neither is repeated on this resume path.
                self.saved_history_image_reference_bytes += 2 * ref['bytes']
        self.reused += 1
        self.budget.check()
        return record

    def summary(self):
        return dict(schema='st_committed_history_resume_v1',
            committed_scenes=len(self.rows), reused_scenes=self.reused,
            new_scenes=self.new,
            old_registered_image_reference_bound=self.saved_history_image_reference_bytes,
            rehashed_unique_image_count=len(self.verified_images),
            rehashed_unique_image_bytes=self.rehashed_image_bytes,
            saved_registered_image_reference_bytes=self.saved_history_image_reference_bytes-self.rehashed_image_bytes,
            source_raw_hash_and_current_identity_rechecked=True,
            current_teacher_image_path_unchanged=True,
            image_reuse_contract='each unique image rehashed against prior SHA; repeat uses same-invocation dev/ino/size/mtime/ctime')


@contextmanager
def resume_original_writer(*, writer, original, bounded, history, schema, record_bytes,
                           census, expected_command_args):
    """Narrow invocation overlay, explicitly recorded in the independent admission.

    Registration/claim/source bytes stay unchanged. Its exact parser argv is
    validated first; only the original native ``resume`` boolean is enabled.
    Original config identity deliberately excludes that operational boolean.
    """
    factory, budget_type = writer.parser, bounded.PreparationBudget
    build = original.build_history_record
    state = {}

    def parser():
        p = factory()
        parse = p.parse_args
        def parse_args(argv=None, namespace=None):
            if argv != expected_command_args:
                raise ValueError('Resume cannot alter the registered builder arguments')
            args = parse(argv, namespace)
            if args.resume or args.record_layout != 'packed_log_v1':
                raise ValueError('Only explicit legacy packed partial resume')
            args.resume = True
            return args
        p.parse_args = parse_args
        return p

    class ResumeBudget(budget_type):
        def __init__(self, root, limits):
            super().__init__(root, limits)
            if state:
                raise ValueError('Exactly one original writer invocation')
            state['records'] = CommittedHistoryRecords(root, self, census,
                bounded=bounded, history=history, schema=schema, record_bytes=record_bytes)

    def build_or_reuse(frames, token, log, current_record, sensor_roots, *,
                       raw_log_sha256, offsets_s=(-1.0, -0.5), tolerance_s=.05):
        records = state['records']
        result = records.get(token, log, current_record, raw_log_sha256,
            offsets=offsets_s, tolerance=tolerance_s, sensor_roots=sensor_roots)
        if result is None:
            return build(frames, token, log, current_record, sensor_roots,
                raw_log_sha256=raw_log_sha256, offsets_s=offsets_s,
                tolerance_s=tolerance_s)
        return result

    with ExitStack() as stack:
        stack.enter_context(patch.object(writer, 'parser', parser))
        stack.enter_context(patch.object(bounded, 'PreparationBudget', ResumeBudget))
        stack.enter_context(patch.object(original, 'build_history_record', build_or_reuse))
        yield state

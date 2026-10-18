"""Read pinned JSON metadata only to estimate strict deduplicated image hashing."""
import hashlib
import json
import os
from pathlib import Path
import resource
import time

HERE = Path(__file__).resolve().parent


def main():
    census = json.loads((HERE / 'COMMITTED_METADATA_CENSUS.json').read_text())
    root = Path(census['root'])
    started, read = time.monotonic(), 0
    images, occurrences, occurrence_bytes = {}, 0, 0
    decoder = json.JSONDecoder()
    for row in census['rows']:
        read += row['bytes']
        if read > 1280 << 20 or time.monotonic() - started > 120:
            raise ValueError('Metadata-only cost audit exceeded budget')
        delay = read / (32 << 20) - (time.monotonic() - started)
        if delay > 0:
            time.sleep(delay)
        raw = (root / 'history_shards' / row['name']).read_bytes()
        if len(raw) != row['bytes'] or hashlib.sha256(raw).hexdigest() != row['sha256']:
            raise ValueError('Pinned metadata changed')
        text, pos = raw.decode('ascii'), 0
        while pos < len(text):
            record, end = decoder.raw_decode(text, pos)
            pos = end + 1
            for frame in record['frames']:
                for view in frame['views']:
                    ref = view['image']
                    if ref is None:
                        continue
                    previous = images.setdefault(ref['path'], ref)
                    if previous != ref:
                        raise ValueError('Same image path has conflicting original evidence')
                    occurrences += 1
                    occurrence_bytes += ref['bytes']
    unique_bytes = sum(r['bytes'] for r in images.values())
    report = dict(schema='st_history_resume_source_io_estimate_v1',
        committed_scenes=census['scenes'], missing_scenes=census['missing_scene_count'],
        committed_history_image_occurrences=occurrences,
        committed_unique_history_image_paths=len(images),
        committed_history_occurrence_bytes=occurrence_bytes,
        old_registered_history_image_read_bound_bytes=2 * occurrence_bytes,
        strict_once_per_unique_image_hash_bytes=unique_bytes,
        unique_image_hash_floor_seconds_at_32_MiB_s=unique_bytes / (32 << 20),
        registration_bytes_saved_for_committed_history=2 * occurrence_bytes - unique_bytes,
        current_teacher_reads_and_uncovered_scenes_not_measured=True,
        no_source_image_raw_model_opened=True, metadata_bytes_read=read,
        elapsed_seconds=time.monotonic() - started,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        cpu_affinity=sorted(os.sched_getaffinity(0)))
    print(json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    main()

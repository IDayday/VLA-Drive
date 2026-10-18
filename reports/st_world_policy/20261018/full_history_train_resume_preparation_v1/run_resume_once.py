#!/usr/bin/env python3
"""One independently admitted same-input resume using the original CPU monitor."""
import time
ENTRY = time.monotonic()
with open('/proc/self/io') as _entry_io:
    ENTRY_IO = {k: int(v) for k, v in (line.split(':') for line in _entry_io)}
from pathlib import Path
import sys
import types

from storage_binding import source_bytes
from resume_records import resume_original_writer

HERE = Path(__file__).resolve().parent
BASE_WRAPPER = Path('/mnt/project/VLA-Drive-st-world-policy-20261016/reports/st_world_policy/20261018/full_causal_history_train_attempt2_preparation_v1/run_full_history_once.py')
BASE_WRAPPER_SHA = '09b526a9375c5c1c4df6a2a67e07a5f5ab91ca6ea3a061365f0460c435922063'
PLAN_SHA = '00746c8aa133558369bee39dc4d011ccc5a36bc0d2a352fc0c7ccf8f7e6c5d34'
OLD_PLAN_SHA = 'a6ea02aa66ff98a51284272f024136d51867e1e999a04a78f9f03ecee4219ed3'


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('Pinned history controller adaptation changed')
    return source.replace(old, new)


def load_controller():
    source = source_bytes(BASE_WRAPPER, BASE_WRAPPER_SHA).decode()
    source = replace_once(source, OLD_PLAN_SHA, PLAN_SHA)
    # The retained implementation wrapper is itself an explicit dependency.
    source = replace_once(source,
        "BOUND_EXTRA_SOURCES={str(Path(__file__).resolve()), str(HERE/'storage_binding.py'), str(HISTORY32)}",
        "BOUND_EXTRA_SOURCES={str(Path(__file__).resolve()), str(HERE/'storage_binding.py'), str(HISTORY32), str(BASE_WRAPPER), str(HERE/'resume_records.py')}")
    source = replace_once(source,
        "required_extra = {str(Path(__file__).resolve()), str(HERE/'storage_binding.py'), str(HISTORY32)}",
        "required_extra = {str(Path(__file__).resolve()), str(HERE/'storage_binding.py'), str(HISTORY32), str(BASE_WRAPPER), str(HERE/'resume_records.py')}")
    module = types.ModuleType('same_input_history_resume_controller')
    module.__dict__.update(__file__=str(Path(__file__).resolve()), BASE_WRAPPER=BASE_WRAPPER)
    exec(compile(source, str(BASE_WRAPPER), 'exec'), module.__dict__)
    module.ENTRY, module.ENTRY_IO = ENTRY, ENTRY_IO
    original_bound_source, original_controller, original_validate = module.bound_source, module.controller, module.validate

    def bound_source(raw):
        source = original_bound_source(raw)
        source = replace_once(source,
            "core.publish_guarded(m,b,guard,start,{'admission':args.admission,'sha256':args.admission_sha256,'automatic_retry':False})",
            'BOUND_PRESERVE_START(start)')
        source = replace_once(source,
            'with nonblocking_original_ledger(b):result=m.run_registered_metadata(**guard)',
            'with nonblocking_original_ledger(b):result=BOUND_RESUME_CALL(m,guard,reg)')
        source = replace_once(source,
            "'start_marker':str(start),'ledger_result_write_confirmed':False,'ledger_result_exists':result_path.exists(),'scientific_result':False}",
            "'start_marker':str(start),'ledger_result_write_confirmed':False,'ledger_result_exists':result_path.exists(),'scientific_result':False, 'preserved_child_failure':BOUND_FAILURE_SUMMARY(observed)}")
        return source

    def validate(a, cp, role, spec, inherited, *, now, child=False):
        cpus = original_validate(a, cp, role, spec, inherited, now=now, child=child)
        plan = inherited.pinned(a['plan'])
        if role != 'train' or a.get('resume_overlay') != plan['resume_overlay']:
            raise ValueError('Explicit independent same-input resume admission required')
        if a.get('reviewer') in ('/root/temporal_profile_prepare', '/root'):
            raise ValueError('Resume author cannot admit itself')
        for pin in plan['versioned_attempt']['old_failures']:
            inherited.pinned(pin)
        inherited.pinned(plan['resume_overlay']['committed_metadata_census'])
        return cpus

    def controller(raw, a, cp, role, spec, inherited, cpus):
        result = original_controller(raw, a, cp, role, spec, inherited, cpus)
        overlay = a['resume_overlay']
        def preserve_start(path):
            if str(path) != overlay['preserved_start_marker']['path']:
                raise ValueError('Original start marker path changed')
            inherited.pinned(overlay['preserved_start_marker'])
            if (Path(spec['output_root']) / 'COMPLETE.json').exists():
                raise ValueError('Original history is already complete; do not repeat')
        def resume_call(metadata, guard, reg):
            # Native source validation, ledger admission, locks, storage checks,
            # budgets and original finalization all still execute unchanged.
            from tools.st_world_policy import prepare_causal_history as writer
            from tools.st_world_policy import prepare_history as original
            from tools.st_world_policy import prepare_future_metadata as bounded
            from tools.st_world_policy.history_shards import record_bytes
            from starVLA.dataloader import st_world_history as history, st_world_schema as schema
            census = inherited.pinned(overlay['committed_metadata_census'])
            with resume_original_writer(writer=writer, original=original, bounded=bounded,
                    history=history, schema=schema, record_bytes=record_bytes, census=census,
                    expected_command_args=reg['command'][3:]) as state:
                value = metadata.run_registered_metadata(**guard)
                summary = state['records'].summary()
                if (summary['reused_scenes'] != overlay['expected_committed_scenes'] or
                        summary['new_scenes'] != overlay['expected_new_scenes']):
                    raise ValueError('Actual resume consumption differs from pinned prefix/population')
                value['same_input_resume'] = summary
                return value
        result.BOUND_PRESERVE_START = preserve_start
        result.BOUND_RESUME_CALL = resume_call
        result.BOUND_FAILURE_SUMMARY = failure_summary
        return result
    module.bound_source, module.controller, module.validate = bound_source, controller, validate
    return module


def failure_summary(observed):
    """Keep causal child/monitor evidence even if shared-ledger publication fails."""
    if observed is None:
        return None
    child = observed.get('child_accounting') or {}
    return dict(returncode=observed.get('returncode'), failure=observed.get('failure'),
        monitor_error=observed.get('monitor_error'), child_error=child.get('error'),
        child_counters=child.get('counters'), read_accounting=observed.get('read_accounting'),
        cleanup=observed.get('cleanup'))


if __name__ == '__main__':
    raise SystemExit(load_controller().main())

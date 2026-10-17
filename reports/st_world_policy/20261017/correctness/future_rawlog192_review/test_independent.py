"""Small CPU/source-bound counterexamples; never open a real raw log."""
from pathlib import Path
import subprocess

import pytest
from tools.st_world_policy import prepare_future_metadata as f


def args():
    return f.parser().parse_args(['plan','--student-root','/fixture/student','--partition','/fixture/partition',
        '--current-population','/fixture/current','--raw-log-root','/fixture/raw','--output','/fixture/output',
        '--sensor-root','/fixture/images','--split','train','--scope','full_population',
        '--max-scenes','101592','--workers','1'])


def limits():
    a=args()
    return {k:getattr(a,k) for k in ('workers','read_mib_per_second','max_seconds','max_source_read_bytes',
        'max_metadata_bytes','max_rss_bytes','max_raw_log_bytes')}


def test_parser_old_default_and_unrelated_limits_stay_fixed():
    a=args(); values=limits()
    assert a.max_raw_log_bytes==128*f.MiB
    assert values['max_rss_bytes']==4*f.GiB and values['workers']==1
    assert values['max_metadata_bytes']==2*f.GiB and values['read_mib_per_second']==64
    assert f.validate_limits(values)==values
    for observed in (147814233,137629626,171848564,192*f.MiB):
        assert f.validate_limits(dict(values,max_raw_log_bytes=observed))['max_raw_log_bytes']==observed


@pytest.mark.parametrize('bad',[192*(1<<20)+1,True,192.0*(1<<20)])
def test_overmaximum_or_noninteger_rejected(bad):
    with pytest.raises(ValueError): f.validate_limits(dict(limits(),max_raw_log_bytes=bad))


@pytest.mark.parametrize('registered,actual',[(128*(1<<20),171848564),
                                           (192*(1<<20),192*(1<<20)+1)])
def test_actual_build_rejects_oversize_before_hash_or_unpickle(monkeypatch,registered,actual):
    p={'source':{'student_root':'/fixture/student'},'raw_log_root':'/fixture/raw',
       'limits':dict(limits(),max_raw_log_bytes=registered)}
    monkeypatch.setattr(f,'version',lambda path:(1,2,actual,4,5))
    def forbidden(*args,**kwargs):pytest.fail('oversize source reached raw IO/deserialization')
    monkeypatch.setattr(f,'budget_sha',forbidden)
    monkeypatch.setattr(f,'BudgetedReader',forbidden)
    monkeypatch.setattr(f,'TrustedNumpyUnpickler',forbidden)
    with pytest.raises(f.BudgetExceeded,match='RAW_LOG_SIZE_BUDGET'):
        f.build_log(p,{},'fixture_log',[],Path('/fixture/output'),object())


def test_only_production_change_is_hard_limit():
    prior=subprocess.check_output(['git','show','fc6c6c70^:tools/st_world_policy/prepare_future_metadata.py'],text=True)
    current=Path(f.__file__).read_text()
    before="'max_rss_bytes':4*GiB,'max_raw_log_bytes':128*MiB}"
    after="'max_rss_bytes':4*GiB,'max_raw_log_bytes':192*MiB}"
    assert prior.count(before)==1 and current==prior.replace(before,after)

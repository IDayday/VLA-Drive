import json
import pytest
from scripts.analysis.fm_transfer.analyze_sft_training import scan_metrics


def row(update):
    return dict(rank=0,updates=update,cursor=update*4,scene_tokens=['train_a','train_b'],
        loss=.1,seconds_data=.2,seconds_update=.8,wall_seconds_this_invocation=float(update))


def scan(tmp_path, rows):
    p=tmp_path/'journal.jsonl';p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return scan_metrics([p],rank=0,updates=2,per_rank_batch=2,global_batch=4,allowed_tokens={'train_a','train_b'})


def test_scene_exposure_is_not_noise_repeat_or_rank_count(tmp_path):
    result,tokens=scan(tmp_path,[row(1),row(2)])
    assert result['updates']==2 and result['rank_scene_exposures']==4
    assert tokens=={'train_a','train_b'}
    assert result['data_seconds_sum']==.4 and result['update_seconds_sum']==1.6


@pytest.mark.parametrize('failure',['overlap','missing','cursor','foreign_rank','dev_token','nonfinite'])
def test_reject_unproved_effective_update_or_population(tmp_path,failure):
    rows=[row(1),row(2)]
    if failure=='overlap':rows.append(row(2))
    elif failure=='missing':rows.pop()
    elif failure=='cursor':rows[1]['cursor']=7
    elif failure=='foreign_rank':rows[1]['rank']=1
    elif failure=='dev_token':rows[1]['scene_tokens'][1]='dev'
    elif failure=='nonfinite':rows[1]['loss']=float('nan')
    with pytest.raises(ValueError):scan(tmp_path,rows)
